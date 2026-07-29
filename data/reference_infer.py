#!/usr/bin/env python3
"""1단계 모델의 파이썬 참조 구현 — 브라우저 전처리를 검증할 기준.

    python3 data/reference_infer.py                         # 시연 영상 20장 추론
    python3 data/reference_infer.py --dump-tensor s1-os.jpg # 최종 텐서를 .npy 로 저장
    python3 data/reference_infer.py --dump-stages out/      # 단계별 텐서를 .npy 로 저장

가중치는 저장소에 없다. 먼저 받아 두어야 한다(README 팀원 안내 참고):
    hf download HEROJ137/WINTER-retina-models \\
        stage1_odir_convnextv2_tiny_int8.onnx --local-dir data/model

## 이 파일은 전처리를 직접 구현하지 않는다

이전 버전은 `cv2` 로 전처리를 다시 구현했고, 그래서 **학습 파이프라인과 달랐다.**
"브라우저 vs 이 스크립트" 비교가 성립하려면 이 스크립트가 학습과 같아야 하는데
그렇지 않았으므로, 그 비교로 얻은 편차값(최대 0.085)은 잘못된 기준에 대한 값이다.

학습 저장소(`model/`)에서 실측한 이탈량 — `scripts/preproc_parity.py`,
데모 20장, ImageNet 정규화 공간의 mean RMS 를 gray level 로 환산:

| 이전 버전이 틀렸던 것 | 이탈량 |
| --- | --- |
| `cv2.resize(INTER_LINEAR)` — 축소 시 antialias 없음. 학습은 PIL bilinear(있음) | 3.17 gray lv |
| `match_histogram` 이 통계는 마스크로 내면서 **매핑을 배경까지 덮어씀** | 2.50 gray lv |
| `cv2.createCLAHE` — 학습은 `src/augment.py` 의 numpy 구현(단일패스 재분배) | 미측정(이 환경에 cv2 없음) |
| 판정에 `>=` 사용 — 학습은 strict `>` (`apply_thresholds`) | 1 ULP |

정사각 패딩 후 배경은 화면의 약 21% 다. 무시할 수 있는 영역이 아니다.

그래서 이제 **학습 코드를 그대로 import 한다.** vendoring(복사)하지 않는 이유는
복사본이 조용히 낡기 때문이고, 그게 애초에 고치려던 실패 모드다.
`model/` 저장소 경로는 `--model-repo` 또는 환경변수 `WINTER_MODEL_REPO` 로 지정하며,
기본값은 형제 디렉터리 `../model` 이다. 찾지 못하면 **cv2 로 되돌아가지 않고 실패한다** —
조용히 틀린 기준을 만드는 것보다 멈추는 것이 낫다.

전처리 순서는 `preprocessing.json` v2 의 `order` 필드에 명시돼 있다(v1 에는 없었다).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'data', 'model')
STEM = 'stage1_odir_convnextv2_tiny'
IMAGES = os.path.join(ROOT, 'data', 'images')
DEFAULT_MODEL_REPO = os.path.join(os.path.dirname(ROOT), 'model')


def load_training_pipeline(model_repo):
    """학습 저장소의 전처리 코드를 import 한다. 실패하면 멈춘다."""
    repo = os.path.abspath(model_repo)
    if not os.path.isdir(os.path.join(repo, 'src')):
        sys.exit(
            '학습 저장소를 찾을 수 없습니다: %s\n'
            '이 스크립트는 전처리를 직접 구현하지 않고 학습 코드를 그대로 import 합니다.\n'
            '  --model-repo <경로>  또는  export WINTER_MODEL_REPO=<경로>\n'
            '로 model/ 저장소를 지정하세요 (기본값은 형제 디렉터리 ../model).' % repo
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from src.augment import CLAHE, ColorReference, HistogramMatch, build_transforms6
        from src.data import FundusPrepare
    except ImportError as exc:
        sys.exit(
            'model/ 저장소에서 전처리 모듈을 import 하지 못했습니다: %s\n'
            'pip install torch torchvision pillow numpy 가 필요합니다.' % exc
        )
    return {
        'build_transforms6': build_transforms6,
        'ColorReference': ColorReference,
        'HistogramMatch': HistogramMatch,
        'CLAHE': CLAHE,
        'FundusPrepare': FundusPrepare,
    }


def load_meta():
    def j(suffix):
        p = os.path.join(MODEL_DIR, '%s_%s.json' % (STEM, suffix))
        return json.load(open(p, encoding='utf-8'))
    labels_path = os.path.join(MODEL_DIR, '%s_labels.txt' % STEM)
    labels = open(labels_path, encoding='utf-8').read().split()
    return j('preprocessing'), j('postprocessing'), j('color_reference'), labels


def build_transform(pre, ref_payload, api):
    """학습의 eval-time transform 을 그대로 만든다.

    `build_transforms6(train=False)` 가 계약의 `order` 를 정의하는 당사자다.
    여기서 순서를 다시 쓰지 않는 것이 요점이다 — 다시 쓰면 다시 갈라진다.
    """
    reference = api['ColorReference'].from_dict(ref_payload)
    return api['build_transforms6'](
        int(pre['img_size']),
        train=False,
        clahe=bool(pre.get('clahe', True)),
        color_reference=reference,
        color_match_blend=float(pre.get('color_match_blend', 1.0)),
    )


def preprocess(path, transform):
    from PIL import Image
    with Image.open(path) as image:
        tensor = transform(image.convert('RGB'))
    return tensor.unsqueeze(0).numpy().astype(np.float32)      # NCHW


def stage_tensors(path, pre, ref_payload, api):
    """단계별 중간값. 브라우저가 같은 지점을 덤프해 순서대로 비교하면
    처음 어긋나는 단계가 곧 고쳐야 할 단계다."""
    from PIL import Image
    from torchvision import transforms as T

    reference = api['ColorReference'].from_dict(ref_payload)
    size = int(pre['img_size'])
    blend = float(pre.get('color_match_blend', 1.0))

    with Image.open(path) as image:
        rgb = image.convert('RGB')
        prepared = api['FundusPrepare'](fov_crop=True, square_pad=True)(rgb)
    resized = T.Resize((size, size))(prepared)
    matched = api['HistogramMatch'](reference, blend=blend)(resized)
    claheed = api['CLAHE'](clip_limit=2.0, grid=(8, 8), green_only=True)(matched)
    normalised = T.Normalize(pre['imagenet_mean'], pre['imagenet_std'])(
        T.ToTensor()(claheed))
    return {
        '1_after_fov_crop_pad': np.asarray(prepared, dtype=np.float32),
        '2_after_resize': np.asarray(resized, dtype=np.float32),
        '3_after_histogram_match': np.asarray(matched, dtype=np.float32),
        '4_after_clahe_green': np.asarray(claheed, dtype=np.float32),
        '5_after_normalize': normalised.numpy(),
    }


def postprocess(logits, post):
    """클래스별 temperature 로 나눈 뒤 sigmoid. threshold 도 클래스별이다."""
    t = np.asarray(post['temperature'], np.float32)
    p = 1.0 / (1.0 + np.exp(-logits / t))
    return p, np.asarray(post['thresholds'], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-repo',
                    default=os.environ.get('WINTER_MODEL_REPO', DEFAULT_MODEL_REPO),
                    help='학습 저장소(model/) 경로. 전처리를 여기서 import 한다')
    ap.add_argument('--dump-tensor', metavar='영상파일명',
                    help='data/images/ 의 파일 하나를 전처리해 최종 텐서를 .npy 로 저장')
    ap.add_argument('--dump-stages', metavar='디렉터리',
                    help='단계별 중간 텐서를 .npy 로 저장 (--dump-tensor 와 함께 파일 지정)')
    ap.add_argument('--out', default=None, help='--dump-tensor 의 저장 경로')
    args = ap.parse_args()

    api = load_training_pipeline(args.model_repo)
    pre, post, ref, labels = load_meta()
    transform = build_transform(pre, ref, api)

    if pre.get('version', 1) < 2 or 'order' not in pre:
        print('주의: preprocessing.json v%s 에는 order 필드가 없습니다. 이 스크립트는 '
              'src/augment.py 의 순서를 쓰므로 안전하지만, 브라우저 구현자는 순서를 '
              '추측해야 합니다 — 계약을 v2 로 올리세요.' % pre.get('version', 1),
              file=sys.stderr)

    if args.dump_stages:
        name = args.dump_tensor or sorted(
            f for f in os.listdir(IMAGES) if f.lower().endswith('.jpg'))[0]
        os.makedirs(args.dump_stages, exist_ok=True)
        stages = stage_tensors(os.path.join(IMAGES, name), pre, ref, api)
        stem = os.path.splitext(name)[0]
        for stage, array in stages.items():
            np.save(os.path.join(args.dump_stages, '%s__%s.npy' % (stem, stage)), array)
        print('%s: 단계 %d 개를 %s 에 저장' % (name, len(stages), args.dump_stages))
        for stage, array in stages.items():
            print('  %-24s shape=%-18s min=%8.3f max=%8.3f mean=%8.4f'
                  % (stage, array.shape, array.min(), array.max(), array.mean()))
        return

    if args.dump_tensor:
        x = preprocess(os.path.join(IMAGES, args.dump_tensor), transform)
        out = args.out or (os.path.splitext(args.dump_tensor)[0] + '.npy')
        np.save(out, x)
        print('%s → %s  shape=%s  min=%.5f max=%.5f mean=%.5f'
              % (args.dump_tensor, out, x.shape, x.min(), x.max(), x.mean()))
        return

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit('onnxruntime 이 필요합니다: pip install onnxruntime')

    onnx = os.path.join(MODEL_DIR, '%s_int8.onnx' % STEM)
    if not os.path.exists(onnx):
        sys.exit('가중치가 없습니다: %s\nREADME 의 팀원 안내대로 먼저 받으세요.' % onnx)
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
    print('%-12s %-8s %-10s %s'
          % ('영상', '정답', '검출', '  '.join('%5s' % l for l in labels)))
    hit = tot = 0
    for f in files:
        x = preprocess(os.path.join(IMAGES, f), transform)
        logits = sess.run([oname], {iname: x})[0][0]
        p, thr = postprocess(logits, post)
        # strict `>` — 학습의 apply_thresholds 와 같아야 한다. tune_thresholds_recall
        # 이 nextafter 로 임계값을 잡는 이유가 비교가 strict 라는 전제다.
        det = [labels[i] for i in range(len(labels)) if p[i] > thr[i]] or ['-']
        gt = truth.get(f, '?')
        if gt != '?':
            tot += 1
            hit += set(det) == set(gt.split('|'))
        print('%-12s %-8s %-10s %s'
              % (f, gt, '|'.join(det), '  '.join('%5.3f' % v for v in p)))
    print('\nthreshold  %s'
          % '  '.join('%5.2f' % v for v in np.asarray(post['thresholds'])))
    if tot:
        print('눈별 라벨셋 완전일치 %d/%d' % (hit, tot))
        print('※ 20장 표본이며 홀드아웃 전체 정확도가 아니다. 배점용 수치는 '
              'model/ 의 src.evaluate6 를 ODIR test split 전체에 돌려 얻는다.')


if __name__ == '__main__':
    main()
