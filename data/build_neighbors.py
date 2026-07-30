#!/usr/bin/env python3
"""유사 학습 사례(이웃) 빌드: 홀드아웃 임베딩 → kNN → 영상 복사 → neighbors.json.

    python3 data/build_neighbors.py                 # 저장소 루트에서

무엇을 하는가.

  1. 이웃 후보를 고른다 — 홀드아웃에서 6클래스 밖(C·M)과 시연 환자를 뺀 눈들.
  2. 헤드리스 Chrome 으로 data/neighbors_embed.html 을 열어 후보와 시연 20눈의
     임베딩(768차원)을 뽑는다.
  3. 코사인 유사도로 시연 눈마다 이웃 3개를 고른다.
  4. 화면에 필요한 이웃 영상만 data/neighbors/ 로 복사하고 neighbors.json 을 쓴다.

왜 임베딩을 브라우저에서 뽑는가. kNN 은 뱅크와 쿼리가 같은 전처리에서 나와야 성립한다.
파이썬으로 전처리를 다시 구현하면 화면과 갈라지고, 갈라진 두 공간을 비교하는 kNN 은
의미가 없다. 그래서 화면이 쓰는 frontend/project/preproc.js 를 그대로 로드해 돌린다.

왜 시연 환자를 후보에서 빼는가. 안 빼면 자기 사진이 유사도 100% 로 "확정 진단 사례"
로 올라온다. 같은 환자의 반대쪽 눈도 뺀다 — 같은 사람 눈은 이웃이 아니라 자기 자신에
가깝고, 이웃 라벨 일치도라는 신호가 무의미해진다.

전제: 임베딩 출력이 추가된 모델이 필요하다. 원본은 logits 만 출력한다.
    python3 data/make_embed_model.py
"""
import base64
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
SRC_LABELS = os.path.join(ROOT, 'test_dataset_seed42', 'test_labels.csv')
SRC_IMAGES = os.path.join(ROOT, 'archive', 'preprocessed_images')
EMBED_MODEL = os.path.join(DATA, 'model', 'stage1_odir_convnextv2_tiny_int8_embed.onnx')
OUT_IMAGES = os.path.join(DATA, 'neighbors')
OUT_JSON = os.path.join(DATA, 'neighbors.json')
LIST_JSON = os.path.join(DATA, '_neighbors_list.json')
PAGE = os.path.join(DATA, 'neighbors_embed.html')

K = 3                      # 눈당 이웃 수. 화면 컴포넌트 5-7 / 8-10 이 3개씩이다.
DIM = 768
CHROME = ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          '/Applications/Chromium.app/Contents/MacOS/Chromium',
          'google-chrome', 'chromium')


def load_build():
    """build.py 의 키워드→클래스 매핑을 재사용한다. 두 곳에 두면 갈라진다."""
    spec = importlib.util.spec_from_file_location('winter_build', os.path.join(DATA, 'build.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_chrome():
    for c in CHROME:
        if os.path.exists(c):
            return c
        w = shutil.which(c)
        if w:
            return w
    sys.exit('Chrome/Chromium 을 찾을 수 없습니다. 임베딩 추출에 필요합니다.')


def build_list(b):
    scen = list(csv.DictReader(open(os.path.join(DATA, 'scenarios.csv'), encoding='utf-8')))
    demo_ids = {r['patient_id'] for r in scen}
    rows = list(csv.DictReader(open(SRC_LABELS, encoding='utf-8-sig')))

    items, skipped = [], {'out_of_set': 0, 'demo': 0, 'no_image': 0}
    for r in rows:
        eyes = {}
        for side, tag in (('Left', 'os'), ('Right', 'od')):
            dz, _, _ = b.eye_labels(r, side)
            eyes[tag] = dz
        if (eyes['os'] | eyes['od']) & b.OUT_OF_SET:
            skipped['out_of_set'] += 1
            continue
        if r['ID'] in demo_ids:
            skipped['demo'] += 1
            continue
        for side, tag in (('left', 'os'), ('right', 'od')):
            p = os.path.join(SRC_IMAGES, '%s_%s.jpg' % (r['ID'], side))
            if not os.path.exists(p):
                skipped['no_image'] += 1
                continue
            items.append({
                'kind': 'bank', 'id': '%s_%s' % (r['ID'], side), 'pid': r['ID'],
                'file': '%s_%s.jpg' % (r['ID'], side),
                'url': '../' + os.path.relpath(p, ROOT).replace(os.sep, '/'),
                'labels': '|'.join(sorted(eyes[tag])) or 'N',
                'age': r['Patient Age'], 'sex': r['Patient Sex'],
                'keywords': '; '.join(b.keywords(r, side.capitalize())),
            })
    for r in scen:
        for tag in ('os', 'od'):
            items.append({
                'kind': 'query', 'id': 's%s-%s' % (r['scenario_no'], tag), 'pid': r['patient_id'],
                'url': '../data/images/' + r[tag + '_image'],
                'labels': r[tag + '_labels'], 'age': r['age'], 'sex': r['sex'],
            })
    return items, skipped, scen


def extract_embeddings(items, chrome):
    json.dump(items, open(LIST_JSON, 'w', encoding='utf-8'), ensure_ascii=False)
    url = 'file://' + urllib.parse.quote(PAGE)
    cmd = [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
           '--allow-file-access-from-files', '--virtual-time-budget=1800000',
           '--dump-dom', url]
    print('임베딩 추출 %d개 — 몇 분 걸립니다 (약 0.35초/장)' % len(items))
    dom = subprocess.run(cmd, capture_output=True, text=True, errors='replace').stdout
    m = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
    if not m:
        sys.exit('브라우저 출력을 읽지 못했습니다. data/neighbors_embed.html 을 직접 열어 보세요.')
    body, ids, b64 = m.group(1), None, None
    for line in body.splitlines():
        if line.startswith('ERROR '):
            sys.exit('브라우저 오류: ' + line[6:])
        if line.startswith('IDS '):
            ids = line[4:].split(',')
        elif line.startswith('B64 '):
            b64 = line[4:]
    if not ids or not b64:
        sys.exit('임베딩이 비어 있습니다:\n' + body[:400])
    raw = base64.b64decode(b64)
    if len(raw) != len(ids) * DIM:
        sys.exit('임베딩 크기 불일치: %d != %d×%d' % (len(raw), len(ids), DIM))
    os.remove(LIST_JSON)
    return ids, raw


def knn(items, ids, raw, k):
    """순수 파이썬 코사인 kNN. numpy 를 요구하지 않는다 — 848×768 은 이 정도로 충분하다."""
    by_id = {it['id']: it for it in items}
    vecs = {}
    for n, i in enumerate(ids):
        v = [(x - 256 if x > 127 else x) for x in raw[n * DIM:(n + 1) * DIM]]
        s = sum(t * t for t in v) ** 0.5 or 1.0
        vecs[i] = [t / s for t in v]

    bank = [i for i in ids if by_id[i]['kind'] == 'bank']
    out = {}
    for qid in [i for i in ids if by_id[i]['kind'] == 'query']:
        qv, qpid = vecs[qid], by_id[qid]['pid']
        sims = []
        for bid in bank:
            if by_id[bid]['pid'] == qpid:      # 같은 환자는 이웃이 아니다
                continue
            bv = vecs[bid]
            sims.append((sum(qv[t] * bv[t] for t in range(DIM)), bid))
        sims.sort(reverse=True)
        out[qid] = [{'id': b, 'sim': round(s, 4),
                     'labels': by_id[b]['labels'], 'age': by_id[b]['age'],
                     'sex': by_id[b]['sex'], 'keywords': by_id[b].get('keywords', ''),
                     'file': by_id[b]['file']}
                    for s, b in sims[:k]]
    return out


def main():
    if not os.path.exists(EMBED_MODEL):
        sys.exit('임베딩 출력 모델이 없습니다: %s\n'
                 '받거나 만드세요 (둘 다 같은 파일이고 sha256 이 manifest 에 있습니다):\n'
                 '  hf download HEROJ137/WINTER-retina-models \\\n'
                 '      stage1_odir_convnextv2_tiny_int8_embed.onnx --local-dir data/model\n'
                 '  python3 data/make_embed_model.py' % EMBED_MODEL)
    if not os.path.exists(SRC_LABELS):
        sys.exit('원본 라벨이 없습니다: %s\n데이터셋을 가진 로컬에서 실행하세요.' % SRC_LABELS)

    b = load_build()
    items, skipped, scen = build_list(b)
    nbank = sum(1 for i in items if i['kind'] == 'bank')
    print('이웃 후보 %d눈 · 쿼리 %d눈' % (nbank, len(items) - nbank))
    print('  제외: 6클래스 밖 %d명 · 시연 환자 %d명 · 영상 없음 %d눈'
          % (skipped['out_of_set'], skipped['demo'], skipped['no_image']))

    ids, raw = extract_embeddings(items, find_chrome())
    nb = knn(items, ids, raw, K)

    # 화면에 뜨는 영상만 복사한다. 후보 828장 전부는 51MB 라 저장소에 넣지 않는다.
    if os.path.isdir(OUT_IMAGES):
        shutil.rmtree(OUT_IMAGES)
    os.makedirs(OUT_IMAGES)
    used = sorted({n['file'] for v in nb.values() for n in v})
    for f in used:
        shutil.copy2(os.path.join(SRC_IMAGES, f), os.path.join(OUT_IMAGES, f))

    truth = {}
    for r in scen:
        for t in ('os', 'od'):
            truth['s%s-%s' % (r['scenario_no'], t)] = r[t + '_labels']

    json.dump({
        'k': K, 'dim': DIM, 'bankSize': nbank,
        'model': os.path.basename(EMBED_MODEL),
        'note': ('유사도는 모델 임베딩 공간(분류기 직전 768차원)의 코사인 유사도이며 '
                 '동일 진단을 의미하지 않는다. 이웃의 눈별 라벨은 진단 키워드에서 '
                 '역추론한 파생값이다 — 눈별 gold 라벨이 아니다.'),
        'neighbors': nb,
    }, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print('\nneighbors.json — 쿼리 %d개 · 영상 %d장 (%.1f MB)'
          % (len(nb), len(used),
             sum(os.path.getsize(os.path.join(OUT_IMAGES, f)) for f in used) / 1e6))
    agree = 0
    for qid in sorted(nb, key=lambda s: (len(s), s)):
        gt = set(truth[qid].split('|'))
        hit = sum(1 for n in nb[qid] if set(n['labels'].split('|')) == gt)
        agree += hit
        print('  %-9s %-6s → %s  일치 %d/%d'
              % (qid, truth[qid],
                 '  '.join('%s %.3f %-5s' % (n['id'], n['sim'], n['labels']) for n in nb[qid]),
                 hit, K))
    print('\n이웃 라벨 완전일치 %d/%d — 케이스마다 갈리는 것이 정상이다.' % (agree, len(nb) * K))
    print('엇갈리는 이웃은 그 자체로 "애매한 케이스"라는 신호이며, 화면이 그렇게 표시한다.')


if __name__ == '__main__':
    main()
