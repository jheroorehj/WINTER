#!/usr/bin/env python3
"""시연 시나리오 데이터 빌드: 홀드아웃 라벨 → scenarios.csv + scenarios.json + 영상 복사.

원본 데이터셋(test_dataset_seed42/, archive/)은 .gitignore 로 제외돼 있어
클린 클론에서는 실행되지 않는다. 데이터셋을 가진 로컬에서만 돌린다.

    python3 data/build.py            # 저장소 루트에서

시나리오를 추가하려면 아래 SCENARIOS 에 항목 하나만 넣으면 된다.
파생 라벨·영상 복사·CSV·JSON·검증이 모두 따라온다.
"""
import csv, json, os, shutil, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_LABELS = os.path.join(ROOT, 'test_dataset_seed42', 'test_labels.csv')
SRC_IMAGES = os.path.join(ROOT, 'archive', 'preprocessed_images')   # 512x512 전처리본
OUT_DIR = os.path.join(ROOT, 'data')
OUT_IMAGES = os.path.join(OUT_DIR, 'images')

# ── 키워드 → 클래스 매핑 ────────────────────────────────────────────────
# 임의로 정하지 않았다. 원본은 환자 단위 8클래스 라벨만 주고 눈별 정보는 자유 텍스트
# 키워드에만 있으므로, "단일라벨 + 양안 키워드 동일" 환자만 골라 역추론했다.
# 18종이 100% 일관되게 떨어졌고, 애매한 것은 라벨 동반 분포를 전수 확인해 결정했다:
#   refractive media opacity  9/9 가 O 동반  → 소견(O)
#   laser spot                7/7 가 O 동반  → 소견(O), 광응고 흔적
#   post retinal laser surgery 4/4 가 O 동반 → 소견(O)
#   lens dust                 N 단독 18건    → 인공물(라벨 없음)
#   low image quality         N/D 각 1건, O 없음 → 화질 플래그(라벨 없음)
#   anterior segment image    M 1건, O 없음  → 촬영 오류(라벨 없음)
# drusen 은 A 가 아니라 O 다 (10/10). 직관으로 분류하면 틀리는 부분.
CLS = {
    'N': {'normal fundus'},
    'D': {'moderate non proliferative retinopathy', 'mild nonproliferative retinopathy',
          'severe nonproliferative retinopathy', 'diabetic retinopathy',
          'proliferative diabetic retinopathy', 'severe proliferative diabetic retinopathy',
          'myopia retinopathy' and None} - {None},
    'G': {'glaucoma', 'suspected glaucoma'},
    'A': {'dry age-related macular degeneration', 'wet age-related macular degeneration'},
    'H': {'hypertensive retinopathy'},
    'C': {'cataract'},
    'M': {'pathological myopia', 'myopia retinopathy'},
}
# 소견이 아닌 촬영/처치 흔적. 클래스에 넣지 않고 별도 신호로 보존한다.
ARTIFACT = {'lens dust', 'low image quality', 'anterior segment image'}
SIX = ['N', 'D', 'G', 'A', 'H', 'O']          # 모델이 추론하는 라벨셋
OUT_OF_SET = {'C', 'M'}                        # 6클래스 밖 — 시연에서 제외

# ── 시연 시나리오 ───────────────────────────────────────────────────────
# 전부 6클래스 안에서 모델이 추론 가능한 케이스여야 한다.
# demo_primary=1 은 라이브 시연, 0 은 질문 대응용 대기.
SCENARIOS = [
    dict(no=1, key='both_normal', name='양안 정상', primary=1, pid='3079',
         proves='정상일 때 정상이라고 말한다 — 과경보가 없다',
         dname='한지영', bmi='58 kg · 22.5', bp='118 / 74', hba1c='5.4 %',
         hist='없음', vision='OS 0.9 / OD 1.0', med='없음', notes='',
         line='양쪽 눈 모두 이번 사진에서는 특별한 변화가 보이지 않습니다. '
              '정기 검진 주기를 지켜주시면 됩니다.'),
    dict(no=2, key='unilateral_single', name='단안 단일 소견', primary=1, pid='1417',
         proves='한쪽 눈에만 나타나는 질환. 무증상 진행이라 비전문의가 놓치는 전형',
         dname='최병호', bmi='63 kg · 23.1', bp='134 / 80', hba1c='5.8 %',
         hist='녹내장 (형)', vision='OS 0.8 / OD 0.5', med='없음',
         notes='hist=가족력 고위험|vision=우안 저하',
         line='오른쪽 눈 시신경 모양이 녹내장 초기와 비슷합니다. 녹내장은 증상 없이 '
              '진행되므로 안과에서 안압과 시야검사를 받아보시길 권합니다.'),
    dict(no=3, key='unilateral_multi', name='단안 복수 소견', primary=1, pid='4330',
         proves='한 눈에 두 소견이 동반. 1순위만 남기지 않는 설계의 핵심 근거',
         dname='윤미경', bmi='60 kg · 24.2', bp='138 / 86', hba1c='8.4 %',
         hist='T2DM 9년', vision='OS 0.9 / OD 0.4', med='Metformin',
         notes='hba1c=조절 미흡|vision=우안 왜곡 호소',
         line='오른쪽 눈에 두 가지 변화가 함께 보입니다. 당뇨로 인한 혈관 변화와 '
              '중심부(황반) 노화 변화입니다. 두 가지 모두 안과에서 확인이 필요합니다.'),
    dict(no=4, key='bilateral_same', name='양안 동일 소견', primary=0, pid='4318',
         proves='양안 대칭은 전신질환 신호 — 전신질환 연관 경고로 이어진다',
         dname='임재식', bmi='79 kg · 26.8', bp='142 / 88', hba1c='9.6 %',
         hist='T2DM 14년', vision='OS 0.6 / OD 0.7', med='Metformin, Glimepiride',
         notes='hba1c=장기 미조절|bp=고혈압 범위',
         line='양쪽 눈 혈관에 당뇨로 인한 변화가 같은 정도로 보입니다. 눈만의 문제가 '
              '아니라 전신 혈관 상태를 함께 봐야 합니다. 혈당 조절이 가장 중요한 치료입니다.'),
    dict(no=5, key='bilateral_diff', name='양안 서로 다른 소견', primary=1, pid='4215',
         proves='양안을 하나의 순위로 합치는 UI가 깨지는 케이스',
         dname='강동수', bmi='74 kg · 25.4', bp='156 / 94', hba1c='7.9 %',
         hist='T2DM 6년, HTN', vision='OS 0.5 / OD 0.7', med='Metformin, Amlodipine',
         notes='bp=고혈압 2기|hba1c=조절 미흡',
         line='두 눈에 서로 다른 변화가 있습니다. 왼쪽은 당뇨로 인한 변화, 오른쪽은 '
              '혈압으로 인한 변화입니다. 혈당과 혈압을 함께 관리해야 합니다.'),
    dict(no=6, key='bilateral_multi', name='양안 모두 복수 소견', primary=0, pid='931',
         proves='소견 4개를 우선순위로 누르지 않고 어떻게 보여주는가',
         dname='서정한', bmi='71 kg · 24.9', bp='146 / 90', hba1c='8.8 %',
         hist='T2DM 11년, 안압 상승 이력', vision='OS 0.4 / OD 0.5',
         med='Metformin, Latanoprost 점안',
         notes='hba1c=조절 미흡|hist=녹내장 추적 중|vision=양안 저하',
         line='양쪽 눈에 당뇨로 인한 혈관 변화와 시신경 변화가 함께 보입니다. '
              '두 가지가 겹치면 시력 손상이 빠를 수 있어 안과 진료가 필요합니다.'),
    dict(no=7, key='bilateral_overlap', name='양안 일부 겹침 (비대칭 진행)', primary=1, pid='323',
         proves='같은 소견이 양안에 있으나 한쪽에 소견이 더 있다 — 단안 추가 소견이 묻히지 않는가',
         dname='노경환', bmi='68 kg · 24.1', bp='152 / 92', hba1c='6.1 %',
         hist='HTN 8년', vision='OS 0.6 / OD 0.8', med='Losartan',
         notes='bp=고혈압 2기|vision=좌안 저하',
         line='양쪽 눈 혈관에 혈압으로 인한 변화가 있고, 왼쪽 눈에는 추가 변화가 '
              '한 가지 더 보입니다. 혈압 관리와 함께 왼쪽 눈을 우선 확인해야 합니다.'),
]


def keywords(row, side):
    out = []
    for a in row[side + '-Diagnostic Keywords'].split('，'):
        for b in a.split(','):
            b = b.strip()
            if b:
                out.append(b)
    return out


def classify(kw):
    """키워드 하나 → 클래스 코드. 인공물이면 None, 매핑에 없으면 O."""
    if kw in ARTIFACT:
        return None
    for code, words in CLS.items():
        if kw in words:
            return code
    return 'O'


def eye_labels(row, side):
    kws = keywords(row, side)
    codes = {c for c in (classify(k) for k in kws) if c}
    disease = codes - {'N'}
    artifacts = sorted(k for k in kws if k in ARTIFACT)
    return disease, artifacts, kws


def scenario_kind(os_dz, od_dz):
    if (os_dz | od_dz) & OUT_OF_SET:
        return 'out_of_labelset'
    if not os_dz and not od_dz:
        return 'both_normal'
    if bool(os_dz) != bool(od_dz):
        return 'unilateral_multi' if len(os_dz or od_dz) > 1 else 'unilateral_single'
    if os_dz == od_dz:
        return 'bilateral_multi' if len(os_dz) > 1 else 'bilateral_same'
    if os_dz & od_dz:
        return 'bilateral_overlap'
    return 'bilateral_diff'


PROTOTYPE = os.path.join(ROOT, 'frontend', 'project', '망막분석 EMR.dc.html')
BEGIN = '// ==== GENERATED'
END = '// ==== /GENERATED ===='

# 프로토타입은 단일 HTML 로 어디서든 열려야 하므로 데이터를 embed 한다.
# 두 곳에서 갈라지지 않도록 이 스크립트만 그 블록을 쓴다.
PROTO_FIELDS = ['scenario_no', 'scenario_key', 'scenario_name', 'demo_primary',
                'patient_id', 'age', 'sex', 'osImage', 'odImage', 'osLabels', 'odLabels',
                'osArtifact', 'odArtifact', 'osSourceFile', 'odSourceFile',
                'displayName', 'chartNo', 'visitDate', 'site', 'weightBmi',
                'bloodPressure', 'hba1c', 'history', 'correctedVision', 'medication',
                'vitalNotes', 'patientLine']
SNAKE = {'osImage': 'os_image', 'odImage': 'od_image', 'osLabels': 'os_labels',
         'odLabels': 'od_labels', 'osArtifact': 'os_artifact', 'odArtifact': 'od_artifact',
         'osSourceFile': 'os_source_file', 'odSourceFile': 'od_source_file',
         'displayName': 'display_name', 'chartNo': 'chart_no', 'visitDate': 'visit_date',
         'weightBmi': 'weight_bmi', 'bloodPressure': 'blood_pressure',
         'correctedVision': 'corrected_vision', 'vitalNotes': 'vital_notes',
         'patientLine': 'patient_line'}


def inject_prototype(records):
    if not os.path.exists(PROTOTYPE):
        print('  (프로토타입 없음 — 주입 생략)')
        return
    src = open(PROTOTYPE, encoding='utf-8').read()
    i, j = src.index(BEGIN), src.index(END)
    items = []
    for r in records:
        pairs = []
        for k in PROTO_FIELDS:
            v = r[SNAKE.get(k, k)]
            pairs.append('%s: %s' % (k, json.dumps(v, ensure_ascii=False)))
        items.append('  { ' + ', '.join(pairs) + ' }')
    block = (BEGIN + ' — 수정하지 마세요. `python3 data/build.py` 가 이 블록을 덮어씁니다. ====\n'
             '// 출처: data/scenarios.csv (%d건). 실제 데이터는 age/sex/영상/라벨뿐이며\n'
             '// 임상정보와 환자명은 생성값이다 — clinical_fields_synthetic 열 참고.\n'
             'const SCENARIOS = [\n%s\n];\n' % (len(records), ',\n'.join(items)))
    open(PROTOTYPE, 'w', encoding='utf-8').write(src[:i] + block + src[j:])
    print('  프로토타입 GENERATED 블록 주입: %d건' % len(records))


def main():
    if not os.path.exists(SRC_LABELS):
        sys.exit('원본 라벨이 없습니다: %s\n데이터셋을 가진 로컬에서 실행하세요.' % SRC_LABELS)
    rows = list(csv.DictReader(open(SRC_LABELS, encoding='utf-8-sig')))
    by_id = {r['ID']: r for r in rows}
    os.makedirs(OUT_IMAGES, exist_ok=True)

    records, problems = [], []
    for s in SCENARIOS:
        src = by_id.get(s['pid'])
        if not src:
            problems.append('시나리오 %d: 환자 ID %s 없음' % (s['no'], s['pid']))
            continue

        eyes = {}
        for side, tag in (('Left', 'os'), ('Right', 'od')):
            dz, art, kws = eye_labels(src, side)
            if dz & OUT_OF_SET:
                problems.append('시나리오 %d(%s) %s: 6클래스 밖 소견 %s'
                                % (s['no'], s['pid'], tag.upper(), sorted(dz & OUT_OF_SET)))
            img = 's%d-%s.jpg' % (s['no'], tag)
            srcp = os.path.join(SRC_IMAGES, '%s_%s.jpg' % (s['pid'], side.lower()))
            if not os.path.exists(srcp):
                problems.append('시나리오 %d: 전처리 영상 없음 %s' % (s['no'], srcp))
            else:
                shutil.copy2(srcp, os.path.join(OUT_IMAGES, img))
            eyes[tag] = dict(image=img, labels='|'.join(sorted(dz)) or 'N',
                             keywords='; '.join(kws), artifact='; '.join(art),
                             source=src[side + '-Fundus'])

        kind = scenario_kind(set(eyes['os']['labels'].split('|')) - {'N'},
                             set(eyes['od']['labels'].split('|')) - {'N'})
        if kind != s['key']:
            problems.append('시나리오 %d: 선언 key=%s 이지만 실제 라벨은 %s'
                            % (s['no'], s['key'], kind))

        records.append(dict(
            scenario_no=s['no'], scenario_key=s['key'], scenario_name=s['name'],
            demo_primary=s['primary'], proves=s['proves'],
            patient_id=src['ID'], age=src['Patient Age'], sex=src['Patient Sex'],
            patient_level_labels='|'.join(k for k in ['N','D','G','C','A','H','M','O']
                                         if int(src[k])),
            os_image=eyes['os']['image'], od_image=eyes['od']['image'],
            os_labels=eyes['os']['labels'], od_labels=eyes['od']['labels'],
            os_keywords=eyes['os']['keywords'], od_keywords=eyes['od']['keywords'],
            os_artifact=eyes['os']['artifact'], od_artifact=eyes['od']['artifact'],
            os_source_file=eyes['os']['source'], od_source_file=eyes['od']['source'],
            display_name=s['dname'], chart_no='DEMO-%s' % src['ID'],
            visit_date='2026-07-29', site='○○보건소 3차 순회 · 안저카메라 A',
            weight_bmi=s['bmi'], blood_pressure=s['bp'], hba1c=s['hba1c'],
            history=s['hist'], corrected_vision=s['vision'], medication=s['med'],
            vital_notes=s['notes'], patient_line=s['line'],
            clinical_fields_synthetic=('display_name;chart_no;visit_date;site;weight_bmi;'
                                       'blood_pressure;hba1c;history;corrected_vision;'
                                       'medication;vital_notes;patient_line'),
        ))

    if problems:
        print('검증 실패:')
        for p in problems:
            print('  -', p)
        sys.exit(1)

    cols = list(records[0].keys())
    with open(os.path.join(OUT_DIR, 'scenarios.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)

    with open(os.path.join(OUT_DIR, 'scenarios.json'), 'w', encoding='utf-8') as f:
        json.dump({'labelSet': SIX, 'scenarios': records}, f, ensure_ascii=False, indent=2)

    inject_prototype(records)

    print('scenarios.csv / scenarios.json — %d건 %d열' % (len(records), len(cols)))
    cov = collections.Counter()
    for r in records:
        for t in ('os', 'od'):
            for c in r[t + '_labels'].split('|'):
                cov[c] += 1
    print('클래스 커버리지:', dict(sorted(cov.items())))
    missing = [c for c in SIX if c not in cov]
    print('미등장 클래스:', missing or '없음')
    for r in records:
        print('  s%-2d %-20s %-5s %2s세 %-6s OS[%-5s] OD[%-5s] %s'
              % (r['scenario_no'], r['scenario_key'], r['patient_id'], r['age'],
                 r['sex'], r['os_labels'], r['od_labels'],
                 '라이브' if r['demo_primary'] else ''))


if __name__ == '__main__':
    main()
