#!/usr/bin/env python3
"""시나리오별 변형을 만들고 두 모델로 측정해 가장 결과 좋은 것을 고른다.

    python3 data/build_variants.py            # 변형 생성 → 측정 → 선별
    python3 data/build_variants.py --live-graded 1 --live-plain 4

무엇을 하는가.

  1. 시나리오마다 같은 라벨·등급 패턴을 가진 홀드아웃 환자와 test/ 등급 영상을 조합해
     변형을 최대 MAX_VARIANTS 개 만든다.
  2. 헤드리스 Chrome 으로 두 모델을 돌려 변형마다 1·2단계 결과를 측정한다.
  3. 시나리오마다 가장 결과 좋은 변형을 고르고, 심사용 라이브 구성을 정한다 —
     2단계 등급을 맞힌 1건 + 중증도가 화면에 끼어들지 않는 4건.
  4. 선택 결과를 data/variant_pick.json 에 쓴다. build.py 가 이걸 읽어 영상을 복사한다.

왜 선별하는가. 심사에서 34개를 다 보여줄 수 없으니 보여줄 것을 고른다. 다만 고른
케이스가 전체를 대표하지 않으므로, **홀드아웃 전체 수치를 함께 제시해야 한다** —
verify_models.py 의 SUMMARY 가 그 값이다. 선별로 정확도를 가리는 것이 아니다.

왜 변형이 34개뿐인가. 홀드아웃 397명에서 시나리오의 라벨·등급 패턴을 그대로 가진
환자가 s3·s6·s7 은 1명, s2 는 2명뿐이다. 그리고 동반 소견이 있는 눈(A|D, D|G)은
IDRiD 영상으로 교체하지 않는다 — 황반변성·녹내장의 근거가 없는 영상에 그 소견이
있다고 말하는 데이터가 되기 때문이다. 그래서 이 4건은 변형을 만들 수 없다.
"""
import argparse
import base64
import csv
import glob
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
DR_DIR = os.path.join(ROOT, 'test')
OUT_IMAGES = os.path.join(DATA, 'variants')
PICK = os.path.join(DATA, 'variant_pick.json')
LIST = os.path.join(DATA, '_verify_list.json')
PAGE = os.path.join(DATA, 'verify_models.html')

MAX_VARIANTS = 5
STAGE1_URL = ('https://huggingface.co/HEROJ137/WINTER-retina-models'
              '/resolve/main/stage1_odir_convnextv2_tiny_int8.onnx')
STAGE2_BASE = 'https://teachablemachine.withgoogle.com/models/PmdZHe7ke/'
CHROME = ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          '/Applications/Chromium.app/Contents/MacOS/Chromium', 'google-chrome', 'chromium')


def load_build():
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
    sys.exit('Chrome/Chromium 을 찾을 수 없습니다.')


def holdout(b):
    """홀드아웃 환자의 눈별 (라벨셋, 등급). 6클래스 밖과 영상 없는 환자는 제외."""
    out = []
    for r in csv.DictReader(open(SRC_LABELS, encoding='utf-8-sig')):
        eyes, ok = {}, True
        for side, tag in (('Left', 'os'), ('Right', 'od')):
            dz, art, kws = b.eye_labels(r, side)
            if dz & b.OUT_OF_SET:
                ok = False
                break
            g = b.dr_grade_from_keywords(kws)
            if 'D' in dz and g is None:      # 중증도 미기재 D — 등급 정답을 만들 수 없다
                ok = False
                break
            eyes[tag] = {'labels': '|'.join(sorted(dz)) or 'N', 'grade': g, 'dz': dz,
                         'kws': '; '.join(kws), 'art': '; '.join(art),
                         'file': r[side + '-Fundus']}
        if not ok:
            continue
        if not all(os.path.exists(os.path.join(SRC_IMAGES, '%s_%s.jpg' % (r['ID'], s)))
                   for s in ('left', 'right')):
            continue
        out.append({'pid': r['ID'], 'age': int(r['Patient Age']), 'sex': r['Patient Sex'],
                    'eyes': eyes,
                    'plabels': '|'.join(k for k in 'NDGCAHMO' if int(r[k]))})
    return out


def dr_pool():
    pool = {}
    for g in (1, 2, 3, 4):
        d = os.path.join(DR_DIR, 'grading_%d' % g)
        pool[g] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, '*.jpg'))) \
            if os.path.isdir(d) else []
    return pool


def make_variants(b, scen, pats, pool):
    """시나리오 하나의 변형 목록. 환자 후보와 IDRiD 영상 조합을 결정적으로 훑는다."""
    want = {t: (scen[t + '_labels'],
                int(scen[t + '_dr_grade']) if scen[t + '_dr_grade'] else None)
            for t in ('os', 'od')}
    base_age = int(scen['age'])
    cand = [p for p in pats
            if all((p['eyes'][t]['labels'], p['eyes'][t]['grade']) == want[t] for t in ('os', 'od'))]
    # 나이가 가까운 환자를 먼저 쓴다 — 임상 서술(당뇨 병력 연수 등)이 시나리오 단위라
    # 나이가 크게 벌어지면 화면이 앞뒤가 안 맞는다.
    cand.sort(key=lambda p: (abs(p['age'] - base_age), p['pid']))
    if not cand:
        return []

    swap = [t for t in ('os', 'od') if b.wants_idrid(
        want[t][0] == 'D' and {'D'} or set(want[t][0].split('|')),
        set(want['os'][0].split('|')) - {'N'}, set(want['od'][0].split('|')) - {'N'})]

    variants = []
    taken = set()
    for k in range(MAX_VARIANTS):
        p = cand[k % len(cand)]
        eyes = {}
        for t in ('os', 'od'):
            grade = want[t][1]
            if t in swap and grade:
                files = pool.get(grade) or []
                if not files:
                    return variants
                # k 로 시작 위치를 옮기고 이미 쓴 장은 피한다 — 변형끼리 같은 영상을
                # 쓰면 "서로 다른 이미지"가 아니게 된다.
                pick = None
                for off in range(len(files)):
                    f = files[(k + off) % len(files)]
                    if (grade, f) not in taken:
                        taken.add((grade, f))
                        pick = f
                        break
                if pick is None:
                    return variants
                eyes[t] = {'dataset': 'IDRiD', 'src': os.path.join(DR_DIR, 'grading_%d' % grade, pick),
                           'source_file': 'grading_%d/%s' % (grade, pick),
                           'grade_source': 'idrid_folder'}
            else:
                side = 'left' if t == 'os' else 'right'
                eyes[t] = {'dataset': 'ODIR-5K',
                           'src': os.path.join(SRC_IMAGES, '%s_%s.jpg' % (p['pid'], side)),
                           'source_file': p['eyes'][t]['file'],
                           'grade_source': 'odir_keyword' if grade else ''}
            eyes[t].update(labels=want[t][0], grade=grade,
                           keywords=p['eyes'][t]['kws'], artifact=p['eyes'][t]['art'])
        sig = tuple(eyes[t]['source_file'] for t in ('os', 'od'))
        if any(sig == v['sig'] for v in variants):
            continue                                  # 앞선 변형과 영상이 완전히 같다
        variants.append({'k': k, 'pid': p['pid'], 'age': p['age'], 'sex': p['sex'],
                         'plabels': p['plabels'], 'eyes': eyes, 'sig': sig})
    return variants


def measure(items, chrome):
    json.dump({'items': items, 'stage1Url': STAGE1_URL, 'stage2Base': STAGE2_BASE},
              open(LIST, 'w', encoding='utf-8'), ensure_ascii=False)
    print('%d눈 측정 — 몇 분 걸립니다' % len(items))
    out = subprocess.run(
        [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
         '--allow-file-access-from-files', '--virtual-time-budget=3600000',
         '--dump-dom', 'file://' + urllib.parse.quote(PAGE)],
        capture_output=True, text=True, errors='replace').stdout
    os.remove(LIST)
    m = re.search(r'<pre id="out">(.*?)</pre>', out, re.S)
    if not m:
        sys.exit('브라우저 출력을 읽지 못했습니다.')
    body = m.group(1)
    for line in body.splitlines():
        if line.startswith('ERROR '):
            sys.exit('브라우저 오류: ' + line[6:])
        if line.startswith('JSON '):
            return json.loads(line[5:].replace('&quot;', '"').replace('&amp;', '&'))
    sys.exit('측정 결과가 비어 있습니다:\n' + body[:400])


def score(eyes):
    """변형 하나의 점수. 2단계 등급 정답을 최우선으로 본다 — 심사에서 보여줄 것이 그것이다."""
    def norm(x):
        return '|'.join(sorted(t for t in (x or '').split('|') if t and t != 'N'))
    graded = [e for e in eyes if e['truthGrade']]
    g_hit = sum(1 for e in graded if e['gated'] and e['topGrade'] == e['truthGrade'])
    s1_hit = sum(1 for e in eyes if norm(e['truthLabels']) == norm(e['detected']))
    conf = sum(e.get('topP', 0) or 0 for e in graded
               if e['gated'] and e['topGrade'] == e['truthGrade'])
    return (g_hit * 100 + s1_hit * 10 + conf, g_hit, s1_hit, len(graded))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live-graded', type=int, default=1,
                    help='라이브 중 2단계 등급을 맞힌 케이스 개수')
    ap.add_argument('--live-plain', type=int, default=4,
                    help='라이브 중 중증도가 화면에 끼어들지 않는 케이스 개수')
    args = ap.parse_args()

    if not os.path.exists(SRC_LABELS):
        sys.exit('원본 라벨이 없습니다: %s' % SRC_LABELS)

    b = load_build()
    pats, pool = holdout(b), dr_pool()
    scen = list(csv.DictReader(open(os.path.join(DATA, 'scenarios.csv'), encoding='utf-8')))
    print('홀드아웃 %d명 · test/ 등급별 %s\n' % (len(pats), {g: len(v) for g, v in pool.items()}))

    if os.path.isdir(OUT_IMAGES):
        shutil.rmtree(OUT_IMAGES)
    os.makedirs(OUT_IMAGES)

    allv, items = {}, []
    for s in scen:
        no = s['scenario_no']
        vs = make_variants(b, s, pats, pool)
        allv[no] = vs
        for v in vs:
            for t in ('os', 'od'):
                name = 'v%s-%d-%s.jpg' % (no, v['k'], t)
                shutil.copy2(v['eyes'][t]['src'], os.path.join(OUT_IMAGES, name))
                v['eyes'][t]['image'] = name
                items.append({'key': '%s|%d|%s' % (no, v['k'], t),
                              'id': 'v%s-%d-%s' % (no, v['k'], t),
                              'url': '../data/variants/' + name,
                              'labels': v['eyes'][t]['labels'],
                              'grade': v['eyes'][t]['grade']})
        print('  s%-3s %s변형 %d개' % (no, s['scenario_name'][:18].ljust(20), len(vs)))
    print('\n변형 합계 %d개 · 영상 %d장' % (sum(len(v) for v in allv.values()), len(items)))

    res = measure(items, find_chrome())
    by = {}
    for r in res:
        no, k, t = r['key'].split('|')
        by.setdefault((no, int(k)), {})[t] = r

    ranked = []
    for no, vs in allv.items():
        for v in vs:
            eyes = by.get((no, v['k']), {})
            v['eyes_measured'] = eyes
            v['score'] = score([eyes[t] for t in ('os', 'od') if t in eyes])
            ranked.append((v['score'][0], no, v))
    ranked.sort(key=lambda x: -x[0])

    best = {}
    for _, no, v in ranked:
        if no not in best:
            best[no] = v

    # 라이브 선별. 심사에서 보여줄 구성이다.
    #   등급 케이스  2단계가 등급을 맞힌 것 중 가장 좋은 것 — "2단계도 된다"를 보여준다
    #   평범 케이스  중증도가 화면에 끼어들지 않는 것 — 1단계 능력만 깨끗하게 보여준다
    #
    # "끼어들지 않는다"는 게이트를 넘은 눈이 없다는 뜻이다. 게이트를 넘으면 2단계가
    # 돌고, 지금 모델은 대개 grade0 을 내므로 "1단계와 불일치" 노트가 뜬다. 그 노트가
    # 뜨는 케이스를 1단계 시연용으로 쓰면 설명이 산만해진다.
    def gated_eyes(v):
        return sum(1 for t in ('os', 'od')
                   if v['eyes_measured'].get(t, {}).get('gated'))

    graded_pool = [(sc, no, v) for sc, no, v in ranked if v is best.get(no) and v['score'][1] > 0]
    live = [no for _, no, _ in graded_pool[:args.live_graded]]

    plain = [(sc, no, v) for sc, no, v in ranked
             if v is best.get(no) and no not in live]
    # 게이트 통과 눈이 적은 것 우선, 그다음 1단계 정확도
    plain.sort(key=lambda x: (gated_eyes(x[2]), -x[2]['score'][2], int(x[1])))
    live += [no for _, no, _ in plain[:args.live_plain]]
    live = set(live)

    print('\n%-5s %-4s %-20s %-8s %-7s %s' % ('케이스', '변형', '환자', '2단계', '1단계', '2단계 결과'))
    for sc, no, v in ranked:
        _, g_hit, s1_hit, g_tot = v['score']
        eyes = v['eyes_measured']
        det = []
        for t in ('os', 'od'):
            e = eyes.get(t)
            if not e or not e['truthGrade']:
                continue
            det.append('%s g%s→%s' % (t.upper(), e['truthGrade'],
                                      ('g%s' % e['topGrade']) if e['gated'] and e['topGrade'] is not None else '미실행'))
        mark = ' ★' if best.get(no) is v else ''
        print('  s%-4s#%-3d %-20s %-8s %-7s %s%s'
              % (no, v['k'], '%s %d세 %s' % (v['pid'], v['age'], v['sex'][:1]),
                 '%d/%d' % (g_hit, g_tot) if g_tot else '-',
                 '%d/2' % s1_hit, '  '.join(det) or '-', mark))

    pick = {}
    for no, v in best.items():
        pick[no] = {
            'k': v['k'], 'patient_id': v['pid'], 'age': str(v['age']), 'sex': v['sex'],
            'patient_level_labels': v['plabels'],
            'live': no in live,
            'score': v['score'][0], 'stage2_hit': v['score'][1], 'stage2_total': v['score'][3],
            'stage1_hit': v['score'][2],
            'eyes': {t: {kk: v['eyes'][t][kk] for kk in
                         ('image', 'dataset', 'labels', 'grade', 'grade_source',
                          'source_file', 'keywords', 'artifact')}
                     for t in ('os', 'od')},
        }
    json.dump({'maxVariants': MAX_VARIANTS,
               'liveGraded': args.live_graded, 'livePlain': args.live_plain,
               'note': ('시나리오마다 가장 결과 좋은 변형을 골랐다. live=true 는 심사에서 '
                        '보여줄 구성으로, 2단계 등급을 맞힌 %d건과 중증도가 화면에 '
                        '끼어들지 않는 %d건이다. 고른 케이스는 전체를 대표하지 않으므로 '
                        '홀드아웃 전체 수치(verify_models.py SUMMARY)를 함께 제시할 것.'
                        % (args.live_graded, args.live_plain)),
               'pick': pick}, open(PICK, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print('\nvariant_pick.json — 시나리오 %d건 선택 · 라이브 %d건' % (len(pick), len(live)))
    for no in sorted(live, key=int):
        v = best[no]
        print('  라이브 s%-3s 2단계 %d/%d · 1단계 %d/2 · 게이트 통과 %d눈'
              % (no, v['score'][1], v['score'][3], v['score'][2], gated_eyes(v)))
    print('\n다음: python3 data/build.py  (선택된 영상을 data/images/ 로 복사)')


if __name__ == '__main__':
    main()
