#!/usr/bin/env python3
"""1단계 모델의 파이썬 참조 구현 — 브라우저 전처리를 검증할 기준.

브라우저(onnxruntime-web)와 이 스크립트가 같은 영상에 대해 같은 텐서를 만들고
같은 로짓을 내야 한다. 전처리는 어긋나도 오류가 나지 않고 조용히 틀리므로,
"화면에 그럴듯한 값이 나온다"는 검증이 되지 않는다. 수치로 비교해야 한다.

    python3 data/reference_infer.py                    # 시연 영상 20장 추론
    python3 data/reference_infer.py --dump-tensor s1-os.jpg   # 텐서를 .npy 로 저장

가중치는 저장소에 없다. 먼저 받아 두어야 한다(README 팀원 안내 참고):
    hf download HEROJ137/WINTER-retina-models \\
        stage1_odir_convnextv2_tiny_int8.onnx --local-dir data/model

전처리 단계의 "순서"는 preprocessing.json 에 명시돼 있지 않아 아래 순서를 가정했다.
이 가정이 학습 때와 다르면 추론이 조용히 틀린다 — 학습 스크립트와 대조해 확인할 것.
"""
import argparse, json, os, sys
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('opencv-python 이 필요합니다: pip install opencv-python')
try:
    import onnxruntime as ort
except ImportError:
    sys.exit('onnxruntime 이 필요합니다: pip install onnxruntime')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'data', 'model')
STEM = 'stage1_odir_convnextv2_tiny'
IMAGES = os.path.join(ROOT, 'data', 'images')


def load_meta():
    def j(suffix):
        p = os.path.join(MODEL_DIR, '%s_%s.json' % (STEM, suffix))
        return json.load(open(p, encoding='utf-8'))
    labels = open(os.path.join(MODEL_DIR, '%s_labels.txt' % STEM), encoding='utf-8').read().split()
    return j('preprocessing'), j('postprocessing'), j('color_reference'), labels


# ── 전처리 ──────────────────────────────────────────────────────────────
def fov_crop(bgr, thr):
    """안저 원판의 바운딩 박스로 자른다. 검은 배경이 정규화 통계를 오염시킨다."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > thr)
    if len(xs) == 0:
        return bgr
    return bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def square_pad(bgr):
    """정사각형으로 0 패딩. 비율을 유지해야 병변 형태가 왜곡되지 않는다."""
    h, w = bgr.shape[:2]
    n = max(h, w)
    top, left = (n - h) // 2, (n - w) // 2
    out = np.zeros((n, n, 3), bgr.dtype)
    out[top:top + h, left:left + w] = bgr
    return out


def match_histogram(bgr, ref, blend, thr):
    """채널별 CDF 매칭. ref 는 학습셋 512장에서 만든 기준 CDF 다.

    안저 영상은 카메라·조명에 따라 색조가 크게 흔들린다. 학습셋 색분포로
    끌어당겨야 추론 분포가 학습 분포와 맞는다. 마스크(>thr) 밖의 검은 배경은
    통계에서 제외한다 — 넣으면 0 이 대량으로 들어가 매핑이 무너진다.
    """
    ref_cdf = np.asarray(ref['cdf'], np.float64)          # (3, 256)
    ref_val = np.asarray(ref['values'], np.float64)       # (3, 256)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > thr
    out = bgr.astype(np.float32).copy()
    # cv2 는 BGR, 기준 CDF 는 RGB 순서다.
    for ci, bgr_i in enumerate((2, 1, 0)):
        ch = bgr[:, :, bgr_i]
        vals = ch[mask]
        if vals.size == 0:
            continue
        hist = np.bincount(vals, minlength=256).astype(np.float64)
        src_cdf = np.cumsum(hist) / hist.sum()
        lut = np.interp(src_cdf, ref_cdf[ci], ref_val[ci])          # 0..255
        mapped = lut[ch]
        out[:, :, bgr_i] = (1 - blend) * ch + blend * mapped
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(bgr, clip, grid, green_only):
    """CLAHE. green 채널만 거는 것이 기본이다 — 안저에서 병변 대비가 가장 큰 채널."""
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid))
    out = bgr.copy()
    if green_only:
        out[:, :, 1] = c.apply(out[:, :, 1])
    else:
        for i in range(3):
            out[:, :, i] = c.apply(out[:, :, i])
    return out


def preprocess(path, pre, ref):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit('영상을 읽을 수 없습니다: %s' % path)
    thr = ref.get('mask_threshold', 12)
    if pre.get('fov_crop'):
        bgr = fov_crop(bgr, thr)
    if pre.get('square_pad'):
        bgr = square_pad(bgr)
    n = pre['img_size']
    bgr = cv2.resize(bgr, (n, n), interpolation=cv2.INTER_LINEAR)
    if pre.get('histogram_match'):
        bgr = match_histogram(bgr, ref, float(pre.get('color_match_blend', 1.0)), thr)
    if pre.get('clahe'):
        bgr = apply_clahe(bgr, float(pre['clahe_clip_limit']), pre['clahe_grid'],
                          bool(pre.get('clahe_green_only')))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(pre['imagenet_mean'], np.float32)) / np.asarray(pre['imagenet_std'], np.float32)
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)      # NCHW


# ── 후처리 ──────────────────────────────────────────────────────────────
def postprocess(logits, post):
    """클래스별 temperature 로 나눈 뒤 sigmoid. threshold 도 클래스별이다."""
    t = np.asarray(post['temperature'], np.float32)
    p = 1.0 / (1.0 + np.exp(-logits / t))
    return p, np.asarray(post['thresholds'], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-tensor', metavar='영상파일명',
                    help='data/images/ 의 파일 하나를 전처리해 텐서를 .npy 로 저장')
    ap.add_argument('--out', default=None, help='--dump-tensor 의 저장 경로')
    args = ap.parse_args()

    pre, post, ref, labels = load_meta()
    onnx = os.path.join(MODEL_DIR, '%s_int8.onnx' % STEM)
    if not os.path.exists(onnx):
        sys.exit('가중치가 없습니다: %s\nREADME 의 팀원 안내대로 먼저 받으세요.' % onnx)

    if args.dump_tensor:
        x = preprocess(os.path.join(IMAGES, args.dump_tensor), pre, ref)
        out = args.out or (os.path.splitext(args.dump_tensor)[0] + '.npy')
        np.save(out, x)
        print('%s → %s  shape=%s  min=%.5f max=%.5f mean=%.5f'
              % (args.dump_tensor, out, x.shape, x.min(), x.max(), x.mean()))
        return

    sess = ort.InferenceSession(onnx, providers=['CPUExecutionProvider'])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    # 정답 라벨과 나란히 봐야 예측이 맞는지 알 수 있다.
    truth = {}
    csvp = os.path.join(ROOT, 'data', 'scenarios.csv')
    if os.path.exists(csvp):
        import csv
        for r in csv.DictReader(open(csvp, encoding='utf-8')):
            truth[r['os_image']] = r['os_labels']
            truth[r['od_image']] = r['od_labels']

    files = sorted(f for f in os.listdir(IMAGES) if f.lower().endswith('.jpg'))
    print('%-12s %-8s %-10s %s' % ('영상', '정답', '검출', '  '.join('%5s' % l for l in labels)))
    hit = tot = 0
    for f in files:
        x = preprocess(os.path.join(IMAGES, f), pre, ref)
        logits = sess.run([oname], {iname: x})[0][0]
        p, thr = postprocess(logits, post)
        det = [labels[i] for i in range(len(labels)) if p[i] >= thr[i]] or ['-']
        gt = truth.get(f, '?')
        if gt != '?':
            tot += 1
            hit += set(det) == set(gt.split('|'))
        print('%-12s %-8s %-10s %s' % (f, gt, '|'.join(det),
                                       '  '.join('%5.3f' % v for v in p)))
    print('\nthreshold  %s' % '  '.join('%5.2f' % v for v in np.asarray(post['thresholds'])))
    if tot:
        print('눈별 라벨셋 완전일치 %d/%d' % (hit, tot))
        print('※ 20장 표본이며 홀드아웃 전체 정확도가 아니다. 전처리 순서 가정이 검증되지 않은 상태의 값이다.')


if __name__ == '__main__':
    main()
