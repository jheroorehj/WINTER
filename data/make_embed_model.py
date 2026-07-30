#!/usr/bin/env python3
"""1단계 모델에 임베딩 출력을 추가한다 (비파괴).

    python3 data/make_embed_model.py

원본 산출물은 `logits` 만 출력하므로 유사 사례 kNN 을 할 수 없다. 그런데 분류기 바로 앞에
pooled + LayerNorm 특징 텐서가 있고 **양자화 이전이라 float32** 다. 그래프 출력 목록에
이 텐서를 추가하면 재학습·재양자화 없이 임베딩이 나온다.

노드·가중치·초기값은 하나도 건드리지 않는다. 그래서 `logits` 는 비트 단위로 같아야 하고,
이 스크립트가 그것을 검증한다 — 같지 않으면 실패한다. 원본 파일은 그대로 남는다.

왜 로짓을 임베딩 대신 쓰지 않는가. 6차원 로짓 거리로 이웃을 뽑으면 이웃 라벨이 예측과
일치하는 것이 거의 구조적으로 보장돼, "이웃 라벨 일치 n/3" 이라는 불확실성 신호가
순환논리가 되어 죽는다. 768차원 특징 공간이어야 준독립적인 신호가 된다.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'data', 'model')
STEM = 'stage1_odir_convnextv2_tiny'
SRC = os.path.join(MODEL_DIR, '%s_int8.onnx' % STEM)
DST = os.path.join(MODEL_DIR, '%s_int8_embed.onnx' % STEM)

# 분류기(head.fc) 직전 텐서. Flatten 출력이며 DynamicQuantizeLinear 앞이라 float32 다.
TAP = '/head/flatten/Flatten_output_0'
DIM = 768


def main():
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError as e:
        sys.exit('의존성이 없습니다 (%s).\n  pip install -r data/requirements.txt onnx' % e)

    if not os.path.exists(SRC):
        sys.exit('가중치가 없습니다: %s\nREADME 팀원 안내대로 먼저 받으세요.' % SRC)

    model = onnx.load(SRC)
    names = [o.name for o in model.graph.output]
    if names != ['logits']:
        sys.exit('예상과 다른 출력입니다: %s (logits 하나여야 합니다)' % names)
    if not any(TAP in n.output for n in model.graph.node):
        sys.exit('탭 지점을 찾을 수 없습니다: %s\n모델 구조가 바뀌었다면 '
                 '분류기 직전 텐서 이름을 다시 확인하세요.' % TAP)

    model.graph.output.append(helper.make_tensor_value_info(TAP, TensorProto.FLOAT, ['batch', DIM]))
    onnx.save(model, DST)

    # logits 동일성 검증. 다르면 산출물을 지우고 실패한다 — 조용히 다른 모델을 배포하는
    # 것보다 멈추는 것이 낫다.
    a = ort.InferenceSession(SRC, providers=['CPUExecutionProvider'])
    c = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
    rs = np.random.RandomState(0)
    worst = 0.0
    for x in (np.zeros((1, 3, 224, 224), np.float32),
              np.ones((1, 3, 224, 224), np.float32),
              rs.randn(1, 3, 224, 224).astype(np.float32)):
        la = a.run(None, {'input': x})[0]
        out = c.run(None, {'input': x})
        worst = max(worst, float(np.abs(la - out[0]).max()))
        if out[1].shape[1] != DIM:
            os.remove(DST)
            sys.exit('임베딩 차원이 %d 가 아닙니다: %s' % (DIM, out[1].shape))
    if worst != 0.0:
        os.remove(DST)
        sys.exit('logits 가 달라졌습니다 (max|diff|=%.3e). 출력만 추가했는데 값이 바뀌면 '
                 '그래프 가정이 틀린 것이므로 배포하지 않습니다.' % worst)

    sha = hashlib.sha256(open(DST, 'rb').read()).hexdigest()
    print('%s (%.3f MB)' % (os.path.basename(DST), os.path.getsize(DST) / 1e6))
    print('  출력: %s' % [o.name for o in c.get_outputs()])
    print('  logits 최대 편차 %.3e → 비트 동일' % worst)
    print('  sha256 %s' % sha)

    # manifest 에 등재한다. 외부에서 받은 가중치의 무결성 기준이 되어야 한다.
    mp = os.path.join(MODEL_DIR, 'manifest.json')
    man = json.load(open(mp, encoding='utf-8'))
    entry = {'file': os.path.basename(DST),
             'size_mb': round(os.path.getsize(DST) / 1e6, 3), 'sha256': sha}
    man = [e for e in man if e['file'] != entry['file']] + [entry]
    man.sort(key=lambda e: e['file'])
    json.dump(man, open(mp, 'w', encoding='utf-8'), indent=2)
    print('  manifest.json 등재 — 항목 %d개' % len(man))
    print('\n다음: hf upload 로 올리고(README 팀원 안내), python3 data/build_neighbors.py')


if __name__ == '__main__':
    main()
