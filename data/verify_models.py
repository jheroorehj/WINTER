#!/usr/bin/env python3
"""두 모델을 화면과 같은 방식으로 돌려 정답과 대조한다.

    python3 data/verify_models.py                  # 시연 20눈 + IDRiD 표본
    python3 data/verify_models.py --idrid 0         # 시연 20눈만
    python3 data/verify_models.py --idrid 5         # 등급 폴더마다 5장씩 추가

모델을 재학습해 올린 뒤 **반드시 이걸 돌려 숫자를 다시 뽑으세요.** README 의
"알려진 문제" 수치가 이 출력에서 나온 것이고, 모델이 바뀌면 그 수치는 거짓이 됩니다.

검증 대상은 "배포된 화면이 내는 값"이다. 그래서 파이썬으로 전처리를 다시 구현하지
않고 브라우저에서 돌린다 — 1단계는 화면과 같은 frontend/project/preproc.js 를,
2단계는 TM 공식 라이브러리를 쓴다. 재구현하면 화면과 다른 것을 재게 되고, 그 편차를
모델 탓으로 오해한다(실제로 그렇게 잘못된 수치를 문서에 적은 적이 있다).

두 모델이 서로 영향을 준다는 점을 기억할 것. 1단계 D 임계값이 곧 2단계 게이트이므로,
1단계 재보정은 2단계가 몇 눈에서 돌아가는지를 바꾼다. 한쪽만 고치고 측정하면 안 된다.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
PAGE = os.path.join(DATA, 'verify_models.html')
LIST = os.path.join(DATA, '_verify_list.json')
STATUS = os.path.join(DATA, 'model_status.json')
DR_DIR = os.path.join(ROOT, 'Diabetic retinopathy')

# 화면이 쓰는 것과 같은 주소여야 한다. 프로토타입의 MODEL_URL / DR_MODEL_BASE 와 맞춰 둔다.
STAGE1_URL = ('https://huggingface.co/HEROJ137/WINTER-retina-models'
              '/resolve/main/stage1_odir_convnextv2_tiny_int8.onnx')
STAGE2_BASE = 'https://teachablemachine.withgoogle.com/models/PmdZHe7ke/'

CHROME = ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          '/Applications/Chromium.app/Contents/MacOS/Chromium',
          'google-chrome', 'chromium')


def find_chrome():
    for c in CHROME:
        if os.path.exists(c):
            return c
        w = shutil.which(c)
        if w:
            return w
    sys.exit('Chrome/Chromium 을 찾을 수 없습니다.')


def build_items(n_idrid):
    items = []
    scen = os.path.join(DATA, 'scenarios.csv')
    if os.path.exists(scen):
        for r in csv.DictReader(open(scen, encoding='utf-8')):
            for tag in ('os', 'od'):
                g = r[tag + '_dr_grade']
                items.append({
                    'key': '%s|%s' % (r['scenario_no'], tag),
                    'id': 's%s-%s (%s)' % (r['scenario_no'], tag, r[tag + '_dataset']),
                    'url': '../data/images/' + r[tag + '_image'],
                    'labels': r[tag + '_labels'],
                    'grade': int(g) if g else None,
                })
    # IDRiD 등급 폴더는 2단계의 학습 도메인으로 추정된다. 여기서 틀리면 모델 문제다.
    for grade in (1, 2, 3, 4):
        d = os.path.join(DR_DIR, 'grading_%d' % grade)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, '*.jpg')))[:n_idrid]:
            items.append({
                'id': 'grading_%d/%s' % (grade, os.path.basename(p)),
                'url': '../' + os.path.relpath(p, ROOT).replace(os.sep, '/'),
                'labels': 'D', 'grade': grade,
            })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idrid', type=int, default=2,
                    help='등급 폴더마다 추가할 장수 (0 이면 시연 영상만)')
    ap.add_argument('--stage1', default=STAGE1_URL)
    ap.add_argument('--stage2', default=STAGE2_BASE)
    args = ap.parse_args()

    items = build_items(args.idrid)
    if not items:
        sys.exit('검증할 영상이 없습니다. data/scenarios.csv 를 먼저 만드세요.')
    json.dump({'items': items, 'stage1Url': args.stage1, 'stage2Base': args.stage2},
              open(LIST, 'w', encoding='utf-8'), ensure_ascii=False)

    print('%d장 검증 — 1단계 가중치 28.5MB 다운로드 포함, 몇 분 걸립니다' % len(items))
    out = subprocess.run(
        [find_chrome(), '--headless=new', '--disable-gpu', '--no-sandbox',
         '--allow-file-access-from-files', '--virtual-time-budget=1800000',
         '--dump-dom', 'file://' + urllib.parse.quote(PAGE)],
        capture_output=True, text=True, errors='replace').stdout
    os.remove(LIST)

    m = re.search(r'<pre id="out">(.*?)</pre>', out, re.S)
    if not m:
        sys.exit('브라우저 출력을 읽지 못했습니다. data/verify_models.html 을 직접 열어 보세요.')
    body = m.group(1)
    results = None
    shown = []
    for line in body.splitlines():
        if line.startswith('JSON '):
            results = json.loads(line[5:].replace('&quot;', '"').replace('&amp;', '&'))
        else:
            shown.append(line)
    print('\n'.join(shown).strip())
    if 'ERROR ' in body:
        sys.exit(1)
    if 'DONE' not in body:
        sys.exit('검증이 끝나지 않았습니다 (타임아웃 또는 로드 실패).')
    if results:
        write_status(results, args)


def badge(eyes):
    """시나리오 하나의 상태 배지. 정답을 지우지 않고, 현재 모델이 그걸 재현하는지만 적는다.

    배지는 손으로 쓰지 않는다 — 재학습하면 손으로 쓴 배지는 즉시 거짓이 된다.
    이 함수가 측정 결과에서 만들고, verify_models.py 를 다시 돌리면 갱신된다.
    """
    tags = []

    # 1단계: 정답 라벨셋과 검출 라벨셋 비교
    def norm(x):
        return '|'.join(sorted(t for t in (x or '').split('|') if t and t != 'N'))
    mism = [e for e in eyes if norm(e['truthLabels']) != norm(e['detected'])]
    if not mism:
        tags.append('1단계 일치')
    else:
        # D 오탐(정답에 D 가 없는데 검출됨)이 가장 흔하고 시연에서 눈에 띈다
        fp_d = [e for e in mism if 'D' in norm(e['detected']) and 'D' not in norm(e['truthLabels'])]
        miss = [e for e in mism if set(norm(e['truthLabels']).split('|')) - set(norm(e['detected']).split('|'))]
        if fp_d:
            tags.append('1단계 D 오탐')
        elif miss:
            tags.append('1단계 미검출')
        else:
            tags.append('1단계 불일치')

    # 2단계: 정답 등급이 있는 눈만 본다
    graded = [e for e in eyes if e['truthGrade']]
    if graded:
        ran = [e for e in graded if e['gated'] and e['topGrade'] is not None]
        if not ran:
            tags.append('2단계 미실행')
        else:
            hit = [e for e in ran if e['topGrade'] == e['truthGrade']]
            nodr = [e for e in ran if e['topGrade'] == 0]
            if len(hit) == len(graded):
                tags.append('2단계 일치')
            elif hit:
                tags.append('2단계 부분일치')
            elif nodr:
                tags.append('2단계 불일치')
            else:
                tags.append('2단계 등급 상이')
    return tags


def write_status(results, args):
    by_scen = {}
    for r in results:
        if not r.get('key'):
            continue
        no, tag = r['key'].split('|')
        by_scen.setdefault(no, {})[tag] = r
    out = {}
    for no, eyes in by_scen.items():
        vals = [eyes[t] for t in ('os', 'od') if t in eyes]
        tags = badge(vals)
        out[no] = {'tags': tags, 'badge': ' · '.join(tags),
                   'eyes': {t: eyes[t] for t in eyes}}
    json.dump({
        'measuredAt': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'stage1Url': args.stage1, 'stage2Base': args.stage2,
        'note': ('현재 모델이 시나리오 정답을 재현하는지의 측정 결과다. 시나리오 이름은 '
                 '정답을 그대로 두고 이 배지로 상태를 표시한다 — 정답은 모델이 틀려도 '
                 '사실이고 정확도 평가에 필요하다. 모델을 재학습하면 '
                 'python3 data/verify_models.py 를 다시 돌려 갱신할 것.'),
        'scenarios': out,
    }, open(STATUS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nmodel_status.json — 시나리오 %d건 배지 갱신' % len(out))
    for no in sorted(out, key=int):
        print('  s%-3s %s' % (no, out[no]['badge']))


if __name__ == '__main__':
    main()
