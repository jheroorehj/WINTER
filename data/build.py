#!/usr/bin/env python3
"""시연 시나리오 데이터 빌드: 홀드아웃 라벨 → scenarios.csv + scenarios.json + 영상 복사.

파이프라인은 2단계다.
  1단계  양안 안저 영상 → 눈별 6클래스 다중라벨 (N,D,G,A,H,O)
  2단계  1단계가 D(당뇨망막병증)를 의심한 눈에만 → 4등급 중증도 (1~4, 상호배타)

그래서 시나리오도 눈별로 "1단계 라벨"과 "2단계 등급"을 함께 들고 있어야 한다.
등급 정답은 두 곳에서 나온다.
  ODIR 키워드   'mild/moderate/severe nonproliferative', 'proliferative' → 1/2/3/4
  IDRiD 폴더    test/grading_N/ 의 N (IDRiD 테스트 분할, 240x240 미전처리)

D 단독 눈은 영상을 IDRiD 쪽으로 교체한다(DR_IMAGE_SWAP 참고). 2단계 모델이
IDRiD 로 학습되므로 시연 영상도 같은 분포에 두고, 등급 정답을 폴더로 검증할 수 있다.
교체할 때도 등급은 ODIR 키워드와 IDRiD 폴더가 일치하는 것만 쓴다 — 둘이 어긋나면
빌드가 실패한다.

원본 데이터셋(test_dataset_seed42/, archive/)은 .gitignore 로 제외돼 있어
클린 클론에서는 실행되지 않는다. 데이터셋을 가진 로컬에서만 돌린다.
(test/ 은 240x240 미전처리 80장이라 저장소에 들어 있다. "미전처리"가 중요하다 —
1단계는 자체 전처리(FOV crop·histogram match·CLAHE)를 하고 2단계는 TM 라이브러리가
자체 전처리를 하므로, 이미 전처리된 영상을 넣으면 이중 처리가 된다. 실제로
val_stage2_ready/(224, 2단계용 전처리본)를 쓰면 2단계는 좋아지고 1단계가 나빠졌다.
val/, val_stage2_ready/, Diabetic retinopathy/ 도 저장소에 있지만 쓰지 않는다.)

    python3 data/build.py            # 저장소 루트에서

시나리오를 추가하려면 아래 SCENARIOS 에 항목 하나만 넣으면 된다.
파생 라벨·등급·영상 복사·CSV·JSON·검증이 모두 따라온다.
"""
import csv, json, os, shutil, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_LABELS = os.path.join(ROOT, 'test_dataset_seed42', 'test_labels.csv')
SRC_IMAGES = os.path.join(ROOT, 'archive', 'preprocessed_images')   # 512x512 전처리본 (모델 입력)
# 화면에 보여줄 원본. 임상의는 촬영된 사진을 그대로 봐야 한다 — 크롭·리사이즈된
# 파생본을 보여주면 "이 도구가 사진을 잘라 버렸다"로 읽힌다.
# 모델 입력은 SRC_IMAGES 를 계속 쓴다. 원본으로 바꿔도 1단계 검출이 12눈 중 10눈
# 동일했고 정확도는 4/12 vs 3/12 로 파생본이 오히려 조금 나았다 — 측정으로 확인했다.
# IDRiD 눈은 원본이 저장소에 없으므로 표시와 모델 입력이 같은 파일이다.
SRC_RAW = os.path.join(ROOT, 'test_dataset_seed42', 'images')
OUT_RAW = os.path.join(ROOT, 'data', 'images_raw')
DR_DIR = os.path.join(ROOT, 'test')                                 # 240x240 IDRiD 테스트 분할 (미전처리)
OUT_DIR = os.path.join(ROOT, 'data')
OUT_IMAGES = os.path.join(OUT_DIR, 'images')
# build_variants.py 가 쓴 선택 결과. 있으면 시나리오의 환자·영상을 이걸로 덮어쓴다.
# 없으면 SCENARIOS 의 pid 를 그대로 쓴다 — 변형을 만들지 않은 상태에서도 빌드된다.
VARIANT_PICK = os.path.join(OUT_DIR, 'variant_pick.json')

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
SIX = ['N', 'D', 'G', 'A', 'H', 'O']          # 1단계 모델이 추론하는 라벨셋
OUT_OF_SET = {'C', 'M'}                        # 6클래스 밖 — 시연에서 제외

# ── 2단계: 당뇨망막병증 중증도 ───────────────────────────────────────────
# 국제 임상 분류(ICDR)와 같은 4단계이며 test/grading_N/ 의 N 과 1:1.
#
# test/ 에는 grading_0(DR 없음) 15장도 있지만 여기서는 쓰지 않는다. 시나리오의 D 눈은
# 정의상 당뇨망막병증이 있는 눈이므로 0등급을 배정할 일이 없다. grading_0 은 2단계
# 모델이 "DR 아님"을 말할 수 있다는 뜻이고, 그건 화면 쪽 계약이다(1단계와 불일치할 때
# 병기를 붙이지 않는다). verify_models.py 는 grading_0 도 검증에 쓴다.
DR_GRADES = {
    1: 'Mild NPDR · 경증 비증식',
    2: 'Moderate NPDR · 중등도 비증식',
    3: 'Severe NPDR · 중증 비증식',
    4: 'Proliferative DR · 증식성',
}
# ODIR 자유 텍스트 키워드 → 등급. 'diabetic retinopathy'(중증도 미기재 12건)는
# 일부러 뺐다 — 등급을 추측하면 2단계 정답이 오염된다.
DR_SEV = {
    'mild nonproliferative retinopathy': 1,
    'moderate non proliferative retinopathy': 2,
    'severe nonproliferative retinopathy': 3,
    'proliferative diabetic retinopathy': 4,
    'severe proliferative diabetic retinopathy': 4,
}

# D 단독 눈의 영상을 IDRiD 로 교체하는 범위.
#   'pure_d_eye'      D 가 유일한 소견인 눈이면 교체 (기본)
#   'pure_d_patient'  양안이 모두 D 단독일 때만 교체 — 한 환자 안에서 두 데이터셋이
#                     섞이지 않아 화면상 카메라 특성이 일관된다
#   'off'             교체하지 않음. 등급 정답은 ODIR 키워드만 사용
# 동반 소견이 있는 눈(A|D, D|G 등)은 어떤 설정에서도 교체하지 않는다. IDRiD 영상에는
# 그 동반 소견의 근거가 없어서, 교체하면 없는 소견을 있다고 말하는 데이터가 된다.
DR_IMAGE_SWAP = 'pure_d_eye'

# ── 시연 시나리오 ───────────────────────────────────────────────────────
# 전부 6클래스 안에서 모델이 추론 가능한 케이스여야 한다.
# demo_primary=1 은 라이브 시연, 0 은 질문 대응용 대기.
#
# key      시나리오 식별자. 화면·문서에서 이 이름으로 부른다.
# pattern  1단계 라벨만으로 결정되는 유형. 생략하면 key 와 같다고 본다.
#          dr_* 시나리오는 1단계 라벨이 D/D 라서 pattern 이 key 와 다르다 —
#          1단계만으로는 s4 와 구분되지 않는다는 것이 이 시나리오들의 요점이다.
# dr       눈별 2단계 등급 선언. 실제 키워드에서 파생한 등급과 다르면 빌드 실패.
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
         dr=dict(od=1),
         proves='한 눈에 두 소견이 동반. 1순위만 남기지 않는 설계의 핵심 근거',
         dname='윤미경', bmi='60 kg · 24.2', bp='138 / 86', hba1c='8.4 %',
         hist='T2DM 9년', vision='OS 0.9 / OD 0.4', med='Metformin',
         notes='hba1c=조절 미흡|vision=우안 왜곡 호소',
         line='오른쪽 눈에 두 가지 변화가 함께 보입니다. 당뇨로 인한 혈관 변화와 '
              '중심부(황반) 노화 변화입니다. 두 가지 모두 안과에서 확인이 필요합니다.'),
    dict(no=4, key='bilateral_same', name='양안 동일 소견', primary=0, pid='4318',
         dr=dict(os=2, od=2),
         proves='양안 대칭은 전신질환 신호 — 전신질환 연관 경고로 이어진다',
         dname='임재식', bmi='79 kg · 26.8', bp='142 / 88', hba1c='9.6 %',
         hist='T2DM 14년', vision='OS 0.6 / OD 0.7', med='Metformin, Glimepiride',
         notes='hba1c=장기 미조절|bp=고혈압 범위',
         line='양쪽 눈 혈관에 당뇨로 인한 변화가 같은 정도로 보입니다. 눈만의 문제가 '
              '아니라 전신 혈관 상태를 함께 봐야 합니다. 혈당 조절이 가장 중요한 치료입니다.'),
    dict(no=5, key='bilateral_diff', name='양안 서로 다른 소견', primary=1, pid='4215',
         dr=dict(os=1),
         proves='양안을 하나의 순위로 합치는 UI가 깨지는 케이스',
         dname='강동수', bmi='74 kg · 25.4', bp='156 / 94', hba1c='7.9 %',
         hist='T2DM 6년, HTN', vision='OS 0.5 / OD 0.7', med='Metformin, Amlodipine',
         notes='bp=고혈압 2기|hba1c=조절 미흡',
         line='두 눈에 서로 다른 변화가 있습니다. 왼쪽은 당뇨로 인한 변화, 오른쪽은 '
              '혈압으로 인한 변화입니다. 혈당과 혈압을 함께 관리해야 합니다.'),
    dict(no=6, key='bilateral_multi', name='양안 모두 복수 소견', primary=0, pid='931',
         dr=dict(os=2, od=2),
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

    # ── 2단계(중증도) 시연용 ─────────────────────────────────────────────
    # 8~10 은 1단계 라벨이 각각 D / D·D / D·D 로 앞의 시나리오와 겹친다.
    # 겹치는 것이 요점이다 — 1단계 출력만 보면 구분되지 않고, 등급이 붙어야
    # 시급도가 갈린다. 그래서 세 건 모두 D 단독이고 영상이 IDRiD 로 교체된다.
    dict(no=8, key='dr_mild_unilateral', name='DR 경증 · 단안', primary=1, pid='4452',
         pattern='unilateral_single', dr=dict(os=1),
         proves='2단계가 경증(1등급)이라고 말할 때 시급도가 내려간다 — '
                '당뇨 소견이 있다는 이유만으로 즉시 의뢰로 튀지 않는다',
         dname='정수미', bmi='62 kg · 23.4', bp='124 / 78', hba1c='7.2 %',
         hist='T2DM 4년', vision='OS 0.9 / OD 1.0', med='Metformin',
         notes='hba1c=조절 양호',
         line='왼쪽 눈에 당뇨로 인한 초기 변화가 조금 보입니다. 지금 단계에서는 '
              '시력에 영향을 주는 정도는 아니어서, 혈당 관리를 유지하면서 정해진 '
              '주기로 안과 검진을 받으시면 됩니다.'),
    dict(no=9, key='dr_grade_adjacent', name='DR 인접 등급 (2 vs 3)', primary=0, pid='4593',
         pattern='bilateral_same', dr=dict(os=3, od=2),
         proves='2단계 등급이 인접 등급과 갈리는 지점. 한 단계 차이는 사진만으로 '
                '구분이 어려워 — 등급을 단정하지 않고 분포를 함께 보여줘야 하는 근거',
         dname='하성길', bmi='77 kg · 26.2', bp='148 / 90', hba1c='9.1 %',
         hist='T2DM 16년', vision='OS 0.4 / OD 0.6',
         # 헤더 셀은 2줄까지만 보인다(폭 약 146px). 약제명은 성분명 단독으로 적는다.
         med='Metformin, Glargine',
         notes='hba1c=장기 미조절|vision=좌안 저하',
         line='양쪽 눈 모두 당뇨로 인한 변화가 진행돼 있고, 왼쪽이 오른쪽보다 한 단계 '
              '더 심해 보입니다. 다만 인접한 단계는 사진만으로 구분이 어려우므로 '
              '안과에서 정밀검사로 확인해야 합니다.'),
    dict(no=10, key='dr_proliferative', name='증식성 DR (4등급)', primary=1, pid='412',
         pattern='bilateral_same', dr=dict(os=3, od=4),
         proves='최고 등급 경로 — 실명 위험 경고와 최고 시급도가 등급에서 파생되는지. '
                '눈별 등급이 다르므로 더 심한 쪽 기준으로 결정돼야 한다',
         dname='배옥분', bmi='54 kg · 21.8', bp='152 / 94', hba1c='10.4 %',
         hist='T2DM 22년, 당뇨병성 신병증', vision='OS 0.3 / OD 0.15',
         med='Lispro, Glargine, Losartan',
         notes='hba1c=장기 미조절|hist=신병증 추적 중|vision=우안 급격 저하',
         line='오른쪽 눈은 당뇨망막병증이 가장 진행된 단계로 보입니다. 이 단계는 '
              '갑작스러운 시력 상실로 이어질 수 있어 오늘 바로 안과 진료를 받으셔야 '
              '합니다. 왼쪽 눈도 함께 확인이 필요합니다.'),
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


def dr_grade_from_keywords(kws):
    """눈별 키워드 → 2단계 등급. 등급을 특정할 수 없으면 None.

    중증도가 다른 DR 키워드가 한 눈에 둘 이상 붙은 경우는 더 심한 쪽을 쓴다
    (임상적으로도 병기는 가장 진행된 소견으로 정한다).
    """
    grades = {DR_SEV[k] for k in kws if k in DR_SEV}
    return max(grades) if grades else None


def dr_pool():
    """grading_N/ → 정렬된 파일명 목록. 정렬해 두므로 배정이 실행마다 같다."""
    pool = {}
    for g in sorted(DR_GRADES):
        d = os.path.join(DR_DIR, 'grading_%d' % g)
        pool[g] = sorted(f for f in os.listdir(d)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))) if os.path.isdir(d) else []
    return pool


def pick_idrid(pool, grade, no, side_idx, taken):
    """등급 폴더에서 IDRiD 영상 한 장 배정. 같은 장이 두 눈에 재사용되지 않는다.

    시작 위치를 scenario_no 로만 정하므로 시나리오를 중간에 끼워 넣어도 기존
    시나리오의 영상이 바뀌지 않는다(순번 배정이면 전부 밀린다).
    """
    files = pool.get(grade) or []
    if not files:
        return None
    start = (no * 2 + side_idx) % len(files)
    for k in range(len(files)):
        f = files[(start + k) % len(files)]
        if f not in taken:
            taken.add(f)
            return f
    return None                                  # 등급 폴더의 장수가 모자란다


def wants_idrid(dz, os_dz, od_dz):
    """이 눈의 영상을 IDRiD 로 교체할지. 동반 소견이 있으면 항상 교체하지 않는다."""
    if DR_IMAGE_SWAP == 'off' or dz != {'D'}:
        return False
    if DR_IMAGE_SWAP == 'pure_d_patient':
        return os_dz == {'D'} and od_dz == {'D'}
    return True


def dr_pattern(os_g, od_g):
    """양안 등급 관계. 2단계 결과를 화면에서 어떻게 합칠지가 여기서 갈린다."""
    if not os_g and not od_g:
        return 'none'
    if bool(os_g) != bool(od_g):
        return 'unilateral'
    return 'bilateral_same_grade' if os_g == od_g else 'bilateral_diff_grade'


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
                'patient_id', 'age', 'sex', 'osImage', 'odImage',
                'osDisplay', 'odDisplay', 'osLabels', 'odLabels',
                'osDrGrade', 'odDrGrade', 'drPattern',
                'osDataset', 'odDataset', 'osDrGradeSource', 'odDrGradeSource',
                'osArtifact', 'odArtifact', 'osSourceFile', 'odSourceFile',
                'displayName', 'chartNo', 'visitDate', 'site', 'weightBmi',
                'bloodPressure', 'hba1c', 'history', 'correctedVision', 'medication',
                'vitalNotes', 'patientLine']
SNAKE = {'osImage': 'os_image', 'odImage': 'od_image',
         'osDisplay': 'os_display', 'odDisplay': 'od_display', 'osLabels': 'os_labels',
         'odLabels': 'od_labels', 'osArtifact': 'os_artifact', 'odArtifact': 'od_artifact',
         'osSourceFile': 'os_source_file', 'odSourceFile': 'od_source_file',
         'osDataset': 'os_dataset', 'odDataset': 'od_dataset',
         'osDrGrade': 'os_dr_grade', 'odDrGrade': 'od_dr_grade',
         'osDrGradeSource': 'os_dr_grade_source', 'odDrGradeSource': 'od_dr_grade_source',
         'drPattern': 'dr_pattern',
         'displayName': 'display_name', 'chartNo': 'chart_no', 'visitDate': 'visit_date',
         'weightBmi': 'weight_bmi', 'bloodPressure': 'blood_pressure',
         'correctedVision': 'corrected_vision', 'vitalNotes': 'vital_notes',
         'patientLine': 'patient_line'}


def inject_prototype(records, total=None):
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
             '// 출처: data/scenarios_demo.csv — 심사 발표용 %d건%s.\n'
             '// 어느 케이스를 보여줄지는 build_variants.py 의 측정 결과가 정한다\n'
             '// (data/variant_pick.json). 고른 케이스는 전체를 대표하지 않으므로\n'
             '// 발표에서 홀드아웃 전체 수치를 함께 제시할 것.\n'
             '// 실제 데이터는 age/sex/영상/라벨뿐이며 임상정보와 환자명은 생성값이다\n'
             '// — clinical_fields_synthetic 열 참고.\n'
             'const SCENARIOS = [\n%s\n];\n'
             % (len(records),
                (' (전체 %d건 중)' % total) if total else '',
                ',\n'.join(items)))
    open(PROTOTYPE, 'w', encoding='utf-8').write(src[:i] + block + src[j:])
    print('  프로토타입 GENERATED 블록 주입: %d건' % len(records))


def main():
    if not os.path.exists(SRC_LABELS):
        sys.exit('원본 라벨이 없습니다: %s\n데이터셋을 가진 로컬에서 실행하세요.' % SRC_LABELS)
    rows = list(csv.DictReader(open(SRC_LABELS, encoding='utf-8-sig')))
    by_id = {r['ID']: r for r in rows}
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_RAW, exist_ok=True)

    pool = dr_pool()
    if not any(pool.values()):
        sys.exit('IDRiD 등급 영상이 없습니다: %s/grading_N/' % DR_DIR)
    taken = set()                       # 같은 IDRiD 장을 두 번 쓰지 않도록

    pick = {}
    if os.path.exists(VARIANT_PICK):
        pick = json.load(open(VARIANT_PICK, encoding='utf-8')).get('pick', {})
        print('variant_pick.json 적용 — 시나리오 %d건이 측정으로 고른 변형을 쓴다' % len(pick))

    records, problems = [], []
    for s in SCENARIOS:
        chosen = pick.get(str(s['no']))
        src = by_id.get(chosen['patient_id'] if chosen else s['pid'])
        if not src:
            problems.append('시나리오 %d: 환자 ID %s 없음'
                            % (s['no'], (chosen or s)['patient_id' if chosen else 'pid']))
            continue

        # 1단계 라벨을 양안 먼저 확정한다. 영상 교체 여부가 반대쪽 눈에도 달려 있다.
        truth = {}
        for side, tag in (('Left', 'os'), ('Right', 'od')):
            dz, art, kws = eye_labels(src, side)
            truth[tag] = dict(side=side, dz=dz, art=art, kws=kws)
            if dz & OUT_OF_SET:
                problems.append('시나리오 %d(%s) %s: 6클래스 밖 소견 %s'
                                % (s['no'], s['pid'], tag.upper(), sorted(dz & OUT_OF_SET)))

        declared = s.get('dr') or {}
        eyes = {}
        for si, tag in enumerate(('os', 'od')):
            t = truth[tag]
            dz, side = t['dz'], t['side']
            img = 's%d-%s.jpg' % (s['no'], tag)

            # 2단계 등급: 키워드에서 파생하고 선언값과 맞는지 본다.
            grade = dr_grade_from_keywords(t['kws'])
            if 'D' in dz and grade is None:
                problems.append('시나리오 %d %s: D 인데 중증도 키워드가 없어 2단계 등급을 '
                                '정할 수 없다 (%s)' % (s['no'], tag.upper(), '; '.join(t['kws'])))
            if 'D' not in dz and grade is not None:
                problems.append('시나리오 %d %s: D 가 아닌데 DR 중증도 키워드가 있다'
                                % (s['no'], tag.upper()))
                grade = None
            if tag in declared and declared[tag] != grade:
                problems.append('시나리오 %d %s: 선언 등급 %s 이지만 키워드는 %s'
                                % (s['no'], tag.upper(), declared[tag], grade))
            elif grade is not None and tag not in declared:
                problems.append('시나리오 %d %s: 등급 %d 인데 dr= 선언이 없다'
                                % (s['no'], tag.upper(), grade))

            # 선택된 변형이 있으면 그 영상을 쓴다. build_variants.py 가 이미 규칙에 따라
            # 고르고 측정까지 한 결과이므로 여기서 다시 배정하지 않는다.
            ce = (chosen or {}).get('eyes', {}).get(tag)
            if ce:
                srcp = os.path.join(OUT_DIR, 'variants', ce['image'])
                dataset, source, gsrc = ce['dataset'], ce['source_file'], ce['grade_source']
            # 영상 선택. D 단독 눈은 IDRiD, 그 외는 ODIR 전처리본.
            elif wants_idrid(dz, truth['os']['dz'], truth['od']['dz']) and grade:
                f = pick_idrid(pool, grade, s['no'], si, taken)
                if not f:
                    problems.append('시나리오 %d %s: grading_%d 에 배정할 영상이 없다'
                                    % (s['no'], tag.upper(), grade))
                    srcp, dataset, source, gsrc = None, '', '', ''
                else:
                    srcp = os.path.join(DR_DIR, 'grading_%d' % grade, f)
                    dataset = 'IDRiD'
                    source = 'grading_%d/%s' % (grade, f)
                    gsrc = 'idrid_folder'          # 폴더가 등급 정답 — 검증 가능
            else:
                srcp = os.path.join(SRC_IMAGES, '%s_%s.jpg' % (s['pid'], side.lower()))
                dataset = 'ODIR-5K'
                source = src[side + '-Fundus']
                gsrc = 'odir_keyword' if grade else ''

            if srcp and not os.path.exists(srcp):
                problems.append('시나리오 %d: 영상 없음 %s' % (s['no'], srcp))
            elif srcp:
                shutil.copy2(srcp, os.path.join(OUT_IMAGES, img))

            # 화면 표시용 원본. ODIR 눈만 원본이 있고, 없으면 모델 입력과 같은 파일을 본다.
            disp = ''
            if dataset == 'ODIR-5K':
                rawp = os.path.join(SRC_RAW, '%s_%s.jpg' % (src['ID'], side.lower()))
                if os.path.exists(rawp):
                    shutil.copy2(rawp, os.path.join(OUT_RAW, img))
                    disp = img

            eyes[tag] = dict(image=img, display=disp, labels='|'.join(sorted(dz)) or 'N',
                             keywords='; '.join(t['kws']), artifact='; '.join(t['art']),
                             source=source, dataset=dataset,
                             grade=grade or '', grade_source=gsrc)

        kind = scenario_kind(set(eyes['os']['labels'].split('|')) - {'N'},
                             set(eyes['od']['labels'].split('|')) - {'N'})
        if kind != s.get('pattern', s['key']):
            problems.append('시나리오 %d: 선언 pattern=%s 이지만 실제 라벨은 %s'
                            % (s['no'], s.get('pattern', s['key']), kind))

        records.append(dict(
            scenario_no=s['no'], scenario_key=s['key'], scenario_name=s['name'],
            # 라이브 여부는 측정 결과로 정한다. 선택 파일이 없으면 선언값을 쓴다.
            demo_primary=(1 if chosen['live'] else 0) if chosen else s['primary'],
            proves=s['proves'],
            patient_id=src['ID'], age=src['Patient Age'], sex=src['Patient Sex'],
            patient_level_labels='|'.join(k for k in ['N','D','G','C','A','H','M','O']
                                         if int(src[k])),
            os_image=eyes['os']['image'], od_image=eyes['od']['image'],
            os_display=eyes['os']['display'], od_display=eyes['od']['display'],
            os_labels=eyes['os']['labels'], od_labels=eyes['od']['labels'],
            os_dr_grade=eyes['os']['grade'], od_dr_grade=eyes['od']['grade'],
            dr_pattern=dr_pattern(eyes['os']['grade'], eyes['od']['grade']),
            os_dataset=eyes['os']['dataset'], od_dataset=eyes['od']['dataset'],
            os_dr_grade_source=eyes['os']['grade_source'],
            od_dr_grade_source=eyes['od']['grade_source'],
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
        json.dump({
            'stage1': {'labelSet': SIX, 'multilabel': True,
                       'note': '눈별 독립 sigmoid. 확률 합은 1이 아니다.'},
            'stage2': {'gate': 'D', 'gradeSet': sorted(DR_GRADES), 'multilabel': False,
                       'grades': {str(k): v for k, v in sorted(DR_GRADES.items())},
                       'note': '1단계가 D 를 의심한 눈에만 조건부로 돌린다. '
                               'grading_0 이 없어 2단계는 "DR 아님"을 말할 수 없다.'},
            'labelSet': SIX,                     # 하위 호환 — 기존 소비자용
            'scenarios': records,
        }, f, ensure_ascii=False, indent=2)

    # ── 발표용 서브셋 ──────────────────────────────────────────────────
    # 심사에서 보여줄 것만 담는다. 전체 10건은 scenarios.csv 에 그대로 남는다 —
    # 정확도 평가와 재측정에 필요하고, 보여줄 것을 골랐다는 사실 자체를 지우지 않는다.
    # 어느 5건인지는 build_variants.py 의 측정 결과(variant_pick.json)가 정한다.
    demo = [r for r in records if r['demo_primary']]
    with open(os.path.join(OUT_DIR, 'scenarios_demo.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(demo)
    with open(os.path.join(OUT_DIR, 'scenarios_demo.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'stage1': {'labelSet': SIX, 'multilabel': True},
            'stage2': {'gate': 'D', 'gradeSet': sorted(DR_GRADES), 'multilabel': False,
                       'grades': {str(k): v for k, v in sorted(DR_GRADES.items())}},
            'note': ('심사 발표용 서브셋 %d건. 전체 %d건은 scenarios.csv 에 있다. '
                     '고른 케이스는 전체를 대표하지 않으므로 홀드아웃 전체 수치'
                     '(verify_models.py SUMMARY)를 함께 제시할 것.'
                     % (len(demo), len(records))),
            'scenarios': demo,
        }, f, ensure_ascii=False, indent=2)

    # 배포되는 화면은 발표용 서브셋을 쓴다. Vercel 이 이 프로토타입을 서브한다.
    inject_prototype(demo, len(records))

    print('scenarios.csv / scenarios.json — %d건 %d열' % (len(records), len(cols)))
    print('scenarios_demo.csv / .json — 발표용 %d건 (s%s)'
          % (len(demo), ', s'.join(str(r['scenario_no']) for r in demo)))
    cov = collections.Counter()
    gcov = collections.Counter()
    gsrc = collections.Counter()
    for r in records:
        for t in ('os', 'od'):
            for c in r[t + '_labels'].split('|'):
                cov[c] += 1
            if r[t + '_dr_grade']:
                gcov[r[t + '_dr_grade']] += 1
                gsrc[r[t + '_dr_grade_source']] += 1
    print('1단계 클래스 커버리지:', dict(sorted(cov.items())))
    print('  미등장 클래스:', [c for c in SIX if c not in cov] or '없음')
    print('2단계 등급 커버리지:', dict(sorted(gcov.items())), '· 등급 근거', dict(gsrc))
    print('  미등장 등급:', [g for g in sorted(DR_GRADES) if g not in gcov] or '없음')
    print('  IDRiD 사용 %d장 / 재사용 0장' % len(taken))
    for r in records:
        g = lambda t: ('g%s' % r[t + '_dr_grade']) if r[t + '_dr_grade'] else '  '
        print('  s%-2d %-19s %-5s %2s세 %-6s OS[%-5s %s] OD[%-5s %s] %-9s %s'
              % (r['scenario_no'], r['scenario_key'], r['patient_id'], r['age'],
                 r['sex'], r['os_labels'], g('os'), r['od_labels'], g('od'),
                 r['dr_pattern'], '라이브' if r['demo_primary'] else ''))


if __name__ == '__main__':
    main()
