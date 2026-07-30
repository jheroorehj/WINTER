// WINTER 1단계 전처리 — 프로토타입과 임베딩 뱅크 빌더가 함께 쓴다.
//
// 이 파일이 하나뿐이어야 하는 이유. 이웃 검색은 코사인 유사도 kNN 이므로 뱅크 임베딩과
// 화면의 쿼리 임베딩이 "같은 전처리"에서 나와야 한다. 전처리를 두 곳에 복사하면 한쪽이
// 조용히 낡고, kNN 이 서로 다른 공간을 비교하게 된다. 그래서 복사하지 않고 공유한다.
//
// 학습 파이프라인과의 정합성은 아직 미검증이다 — README "알려진 문제" 참고.

// ── 전처리 ──────────────────────────────────────────────────────────────
// 학습 때와 하나라도 다르면 추론이 조용히 틀린다(오류가 나지 않는다). 파이썬 참조
// 구현은 data/reference_infer.py 에 있고, 두 구현이 같은 텐서를 만드는지 수치로
// 비교하는 작업이 남아 있다 — README "알려진 문제" 참고.
//
// 체인: FOV crop → square pad → resize 224 → histogram match → CLAHE(green) →
//       ImageNet 정규화 → NCHW
// 이 "순서"는 preprocessing.json 에 명시돼 있지 않아 가정한 것이다.

function loadImage(url) {
  return new Promise((ok, no) => {
    const im = new Image();
    im.crossOrigin = 'anonymous';
    im.onload = () => ok(im);
    im.onerror = () => no(new Error('영상을 읽을 수 없습니다: ' + url));
    im.src = url;
  });
}

function canvasOf(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

// 안저 원판의 바운딩 박스. 검은 배경을 남기면 정규화 통계가 오염된다.
function fovBox(data, w, h, thr) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      // cv2.COLOR_BGR2GRAY 와 같은 계수
      const g = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      if (g > thr) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? { x: 0, y: 0, w: w, h: h } : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

// 채널별 CDF 매칭. 기준 CDF 는 학습셋 512장에서 만든 값이다. 안저 영상은 카메라·
// 조명에 따라 색조가 크게 흔들리므로 학습셋 색분포로 끌어당겨야 추론 분포가 맞는다.
// 마스크 밖(검은 배경)은 통계에서 뺀다 — 넣으면 0 이 대량 유입돼 매핑이 무너진다.
function matchHistogram(px, n, ref, blend, thr) {
  const refCdf = ref.cdf, refVal = ref.values;
  const mask = new Uint8Array(n * n);
  for (let i = 0, j = 0; i < n * n; i++, j += 4) {
    const g = 0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2];
    mask[i] = g > thr ? 1 : 0;
  }
  for (let ch = 0; ch < 3; ch++) {          // 0=R 1=G 2=B, 기준 CDF 도 RGB 순서
    const hist = new Float64Array(256);
    let cnt = 0;
    for (let i = 0, j = ch; i < n * n; i++, j += 4) {
      if (mask[i]) { hist[px[j]]++; cnt++; }
    }
    if (!cnt) continue;
    // 소스 CDF → 기준 CDF 역보간으로 LUT 를 만든다
    const lut = new Float32Array(256);
    let acc = 0, k = 0;
    const rc = refCdf[ch], rv = refVal[ch];
    for (let v = 0; v < 256; v++) {
      acc += hist[v] / cnt;
      while (k < 255 && rc[k] < acc) k++;
      // np.interp 와 같은 선형 보간
      if (k === 0) lut[v] = rv[0];
      else {
        const c0 = rc[k - 1], c1 = rc[k];
        const t = c1 > c0 ? (acc - c0) / (c1 - c0) : 0;
        lut[v] = rv[k - 1] + t * (rv[k] - rv[k - 1]);
      }
    }
    // 매핑은 마스크 안쪽에만 쓴다. 통계는 전경에서 내면서 매핑을 배경까지 덮으면
    // 검은 배경이 밝은 값으로 칠해진다 — 정사각 패딩 후 배경이 화면의 약 21% 라
    // 무시할 수 있는 영역이 아니다. 학습 저장소에서 실측한 이탈량 2.50 gray level.
    for (let i = 0, j = ch; i < n * n; i++, j += 4) {
      if (!mask[i]) continue;
      const v = px[j];
      px[j] = Math.max(0, Math.min(255, Math.round((1 - blend) * v + blend * lut[v])));
    }
  }
}

// CLAHE. cv2.createCLAHE 와 같은 방식이다 — 타일별 히스토그램을 clipLimit 로 자르고
// 초과분을 균등 재분배한 뒤 CDF LUT 을 만들고, 타일 경계는 이중선형 보간한다.
// cv2 의 clipLimit 은 정규화된 값이라 실제 자르는 높이는 clip * 타일면적 / 256 이다.
function clahe(plane, w, h, clipLimit, gx, gy) {
  const tw = Math.ceil(w / gx), th = Math.ceil(h / gy);
  const area = tw * th;
  let clip = Math.max(1, Math.round(clipLimit * area / 256));
  const luts = new Uint8Array(gx * gy * 256);

  for (let ty = 0; ty < gy; ty++) {
    for (let tx = 0; tx < gx; tx++) {
      const hist = new Int32Array(256);
      const xs = tx * tw, ys = ty * th;
      const xe = Math.min(xs + tw, w), ye = Math.min(ys + th, h);
      for (let y = ys; y < ye; y++) {
        for (let x = xs; x < xe; x++) hist[plane[y * w + x]]++;
      }
      let excess = 0;
      for (let v = 0; v < 256; v++) if (hist[v] > clip) { excess += hist[v] - clip; hist[v] = clip; }
      const inc = Math.floor(excess / 256);
      let rest = excess - inc * 256;
      for (let v = 0; v < 256; v++) hist[v] += inc;
      // 남은 픽셀은 cv2 와 같이 일정 간격으로 흩뿌린다
      if (rest > 0) {
        const step = Math.max(1, Math.floor(256 / rest));
        for (let v = 0; v < 256 && rest > 0; v += step) { hist[v]++; rest--; }
        for (let v = 0; v < 256 && rest > 0; v++) { hist[v]++; rest--; }
      }
      const n = (xe - xs) * (ye - ys);
      const scale = 255 / n;
      let sum = 0, base = (ty * gx + tx) * 256;
      for (let v = 0; v < 256; v++) {
        sum += hist[v];
        luts[base + v] = Math.max(0, Math.min(255, Math.round(sum * scale)));
      }
    }
  }

  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    // 타일 중심 기준 좌표
    let fy = y / th - 0.5, ty0 = Math.floor(fy);
    let wy = fy - ty0;
    if (ty0 < 0) { ty0 = 0; wy = 0; }
    let ty1 = Math.min(ty0 + 1, gy - 1);
    if (ty0 > gy - 1) { ty0 = ty1 = gy - 1; wy = 0; }
    for (let x = 0; x < w; x++) {
      let fx = x / tw - 0.5, tx0 = Math.floor(fx);
      let wx = fx - tx0;
      if (tx0 < 0) { tx0 = 0; wx = 0; }
      let tx1 = Math.min(tx0 + 1, gx - 1);
      if (tx0 > gx - 1) { tx0 = tx1 = gx - 1; wx = 0; }
      const v = plane[y * w + x];
      const a = luts[(ty0 * gx + tx0) * 256 + v], b = luts[(ty0 * gx + tx1) * 256 + v];
      const c = luts[(ty1 * gx + tx0) * 256 + v], d = luts[(ty1 * gx + tx1) * 256 + v];
      out[y * w + x] = Math.round((a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy);
    }
  }
  return out;
}

// 영상 → NCHW Float32Array. meta 는 preprocessing.json + color_reference.json.
function preprocess(img, meta) {
  const pre = meta.pre, ref = meta.ref;
  const thr = (ref && ref.mask_threshold !== undefined) ? ref.mask_threshold : 12;
  const n = pre.img_size;

  // 1) 원본을 캔버스로
  let w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  let cv = canvasOf(w, h), cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, 0, 0);
  let box = { x: 0, y: 0, w: w, h: h };

  // 2) FOV crop
  if (pre.fov_crop) box = fovBox(cx.getImageData(0, 0, w, h).data, w, h, thr);

  // 3) square pad + 4) resize 를 한 번의 drawImage 로 합친다. 정사각 캔버스의
  //    가운데에 크롭 영역을 비율 유지해 그리면 패딩과 리사이즈가 동시에 끝난다.
  const side = Math.max(box.w, box.h);
  const dst = canvasOf(n, n), dc = dst.getContext('2d', { willReadFrequently: true });
  // 학습은 PIL bilinear 로 축소하며 antialias 가 켜져 있다. drawImage 의 기본
  // 품질은 구현에 맡겨져 있어 antialias 없이 줄어들 수 있다(실측 3.17 gray level).
  dc.imageSmoothingEnabled = true;
  dc.imageSmoothingQuality = 'high';
  dc.fillStyle = '#000';
  dc.fillRect(0, 0, n, n);
  const k = n / side;
  dc.drawImage(img, box.x, box.y, box.w, box.h,
               (n - box.w * k) / 2, (n - box.h * k) / 2, box.w * k, box.h * k);

  const id = dc.getImageData(0, 0, n, n);
  const px = id.data;

  // 5) histogram match
  if (pre.histogram_match && ref && ref.cdf) {
    matchHistogram(px, n, ref, pre.color_match_blend === undefined ? 1 : pre.color_match_blend, thr);
  }

  // 6) CLAHE — green 채널만이 기본이다. 안저에서 병변 대비가 가장 큰 채널이다.
  if (pre.clahe) {
    const grid = pre.clahe_grid || [8, 8];
    const chans = pre.clahe_green_only ? [1] : [0, 1, 2];
    for (let ci = 0; ci < chans.length; ci++) {
      const ch = chans[ci];
      const plane = new Uint8Array(n * n);
      for (let i = 0, j = ch; i < n * n; i++, j += 4) plane[i] = px[j];
      const eq = clahe(plane, n, n, pre.clahe_clip_limit, grid[0], grid[1]);
      for (let i = 0, j = ch; i < n * n; i++, j += 4) px[j] = eq[i];
    }
  }

  // 7) ImageNet 정규화 → NCHW
  const mean = pre.imagenet_mean || [0, 0, 0], std = pre.imagenet_std || [1, 1, 1];
  const out = new Float32Array(3 * n * n);
  const plane = n * n;
  for (let i = 0, j = 0; i < plane; i++, j += 4) {
    out[i] = (px[j] / 255 - mean[0]) / std[0];
    out[plane + i] = (px[j + 1] / 255 - mean[1]) / std[1];
    out[2 * plane + i] = (px[j + 2] / 255 - mean[2]) / std[2];
  }
  return out;
}
