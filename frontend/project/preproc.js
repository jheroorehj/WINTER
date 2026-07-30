// WINTER 1단계 전처리 — 프로토타입과 임베딩 뱅크 빌더가 함께 쓴다.
//
// 이 파일은 학습 파이프라인(build_stage1_preprocess)의 JS 이식이다. 학습 코드를 받아
// 한 줄씩 대조해 옮겼으므로 추측으로 채운 부분이 없다. 순서는 학습 코드가 정하고,
// HistogramMatch 의 docstring 이 "반드시 아래 순서로 사용한다"고 못 박고 있다.
//
//   1. FOV crop        (grayscale > 12 의 바운딩 박스, 32px 미만이면 크롭하지 않음)
//   2. square padding  (검은색, 종횡비 유지)
//   3. resize 224      (bilinear)
//   4. histogram match (ODIR train 기준 CDF, 마스크는 max(R,G,B) > 12)
//   5. CLAHE           (green 채널만, clip 2.0, grid 8x8)
//   6. ToTensor        (uint8 → float32 [0,1])
//   7. ImageNet 정규화 → NCHW
//
// 이 파일이 하나뿐이어야 하는 이유. 이웃 검색은 코사인 유사도 kNN 이므로 뱅크 임베딩과
// 화면의 쿼리 임베딩이 "같은 전처리"에서 나와야 한다. 전처리를 두 곳에 복사하면 한쪽이
// 조용히 낡고, kNN 이 서로 다른 공간을 비교하게 된다. 그래서 복사하지 않고 공유한다.
//
// 2단계(Teachable Machine)는 이 전처리를 쓰지 않는다. TM 은 자체 전처리(중앙 정사각
// 크롭 + [-1,1])로 학습됐으므로 이 파이프라인을 먹이면 이중 처리가 된다.
//
// 남은 비트 단위 차이는 리샘플링 하나다. 학습은 PIL bilinear(축소 시 antialias)이고
// 브라우저는 drawImage + imageSmoothingQuality='high' 다 — 동일함이 증명되지 않았다.

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

// PIL 의 img.convert("L") 과 같은 ITU-R 601-2 휘도. 반올림까지 맞춘다.
function lumaAt(data, j) {
  return Math.round(0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2]);
}

// crop_fov — grayscale > tol 인 영역의 바운딩 박스.
// 학습 코드에 32px 가드가 있다: 비정상적으로 작은 영역이 검출되면 크롭하지 않는다.
// 이 가드가 없으면 거의 검은 영상에서 몇 픽셀만 남기고 잘라 버린다.
function fovBox(data, w, h, tol) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (lumaAt(data, (y * w + x) * 4) > tol) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return { x: 0, y: 0, w: w, h: h };          // mask.any() == False
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  if (bh < 32 || bw < 32) return { x: 0, y: 0, w: w, h: h };
  return { x: x0, y: y0, w: bw, h: bh };
}

// HistogramMatch — 채널별 CDF 를 ODIR train 기준 CDF 에 맞춘다.
//
// 마스크가 휘도가 아니라 max(R,G,B) > threshold 다. 학습 코드가 그렇게 쓴다
// (array.max(axis=2) > mask_threshold). FOV crop 은 휘도를 쓰고 이쪽은 max 를 쓰는
// 비대칭이 실제로 존재하므로 맞춰야 한다 — 휘도로 두면 어두운 적색 화소가 배경으로
// 빠져 통계가 달라진다.
//
// uint8 변환은 numpy .astype(np.uint8) 과 같이 절단(floor)이다. 반올림하면 값의
// 절반가량이 1 gray level 씩 어긋난다.
function matchHistogram(px, n, ref, blend, thr) {
  const N = n * n;
  const fg = new Uint8Array(N);
  let cnt = 0;
  for (let i = 0, j = 0; i < N; i++, j += 4) {
    if (Math.max(px[j], px[j + 1], px[j + 2]) > thr) { fg[i] = 1; cnt++; }
  }
  if (!cnt) return;                                       // foreground.any() == False

  for (let ch = 0; ch < 3; ch++) {
    const hist = new Float64Array(256);
    for (let i = 0, j = ch; i < N; i++, j += 4) if (fg[i]) hist[px[j]]++;

    const rc = ref.cdf[ch], rv = ref.values[ch], M = rc.length;
    const lut = new Float64Array(256);
    let acc = 0, k = 0;
    for (let v = 0; v < 256; v++) {
      acc += hist[v];
      if (hist[v] === 0) { lut[v] = v; continue; }         // 등장하지 않는 값은 쓰이지 않는다
      const c = acc / cnt;                                 // source_cdf
      while (k < M && rc[k] < c) k++;                      // acc 가 단조라 포인터 하나로 충분
      let m;
      if (k === 0) m = rv[0];                              // np.interp: 왼쪽 밖은 rv[0]
      else if (k >= M) m = rv[M - 1];                      // 오른쪽 밖은 rv[-1]
      else {
        const c0 = rc[k - 1], c1 = rc[k];
        const t = c1 > c0 ? (c - c0) / (c1 - c0) : 0;
        m = rv[k - 1] + t * (rv[k] - rv[k - 1]);
      }
      lut[v] = blend < 1 ? blend * m + (1 - blend) * v : m;
    }
    for (let i = 0, j = ch; i < N; i++, j += 4) {
      if (fg[i]) px[j] = Math.min(255, Math.max(0, Math.floor(lut[px[j]])));
    }
  }
}

// clahe_channel — 학습 코드의 numpy 구현을 그대로 옮긴다. cv2.createCLAHE 와 다르다.
//
//   limit  = max(1.0, clip * tile.size / 256)      실수. 반올림하지 않는다.
//   재분배 = min(hist, limit) + excess/256          단일 패스. 나머지를 흩뿌리지 않는다.
//   LUT    = cumsum / max(cumsum[-1], 1e-9) * 255  float32 유지. 중간 반올림 없음.
//
// cv2 는 초과분을 정수로 나눠 담고 나머지를 다시 흩뿌리는 2패스라서 값이 갈린다.
function claheChannel(chan, w, h, clipLimit, gridY, gridX) {
  const tileY = Math.ceil(h / gridY), tileX = Math.ceil(w / gridX);
  const luts = new Float32Array(gridY * gridX * 256);

  for (let iy = 0; iy < gridY; iy++) {
    for (let ix = 0; ix < gridX; ix++) {
      const base = (iy * gridX + ix) * 256;
      const ys = iy * tileY, ye = Math.min((iy + 1) * tileY, h);
      const xs = ix * tileX, xe = Math.min((ix + 1) * tileX, w);
      const size = Math.max(0, ye - ys) * Math.max(0, xe - xs);
      if (size === 0) {                                    // tile.size == 0 → 항등 LUT
        for (let v = 0; v < 256; v++) luts[base + v] = v;
        continue;
      }
      const hist = new Float64Array(256);
      for (let y = ys; y < ye; y++) {
        for (let x = xs; x < xe; x++) hist[chan[y * w + x]]++;
      }
      const limit = Math.max(1.0, clipLimit * size / 256.0);
      let excess = 0;
      for (let v = 0; v < 256; v++) if (hist[v] > limit) excess += hist[v] - limit;
      const add = excess / 256.0;
      let cum = 0;
      for (let v = 0; v < 256; v++) {
        cum += Math.min(hist[v], limit) + add;
        luts[base + v] = cum;
      }
      const last = Math.max(luts[base + 255], 1e-9);
      for (let v = 0; v < 256; v++) luts[base + v] = luts[base + v] / last * 255.0;
    }
  }

  // 인접 네 타일 mapping 의 이중선형 보간. np.clip 후 절단하는 것까지 같다.
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(Math.max(y / tileY - 0.5, 0), gridY - 1);
    const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, gridY - 1), wy = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(Math.max(x / tileX - 0.5, 0), gridX - 1);
      const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, gridX - 1), wx = fx - x0;
      const v = chan[y * w + x];
      const m00 = luts[(y0 * gridX + x0) * 256 + v], m01 = luts[(y0 * gridX + x1) * 256 + v];
      const m10 = luts[(y1 * gridX + x0) * 256 + v], m11 = luts[(y1 * gridX + x1) * 256 + v];
      const o = (1 - wy) * ((1 - wx) * m00 + wx * m01) + wy * ((1 - wx) * m10 + wx * m11);
      out[y * w + x] = Math.min(255, Math.max(0, Math.floor(o)));
    }
  }
  return out;
}

// 영상 → NCHW Float32Array. meta 는 preprocessing.json + color_reference.json.
function preprocess(img, meta) {
  const pre = meta.pre, ref = meta.ref;
  const tol = (ref && ref.mask_threshold !== undefined) ? ref.mask_threshold : 12;
  const n = pre.img_size;

  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const src = canvasOf(w, h), sx = src.getContext('2d', { willReadFrequently: true });
  sx.drawImage(img, 0, 0);

  // 1. FOV crop
  let box = { x: 0, y: 0, w: w, h: h };
  if (pre.fov_crop) box = fovBox(sx.getImageData(0, 0, w, h).data, w, h, tol);

  // 2. square padding — resize 와 합치지 않는다. 학습은 패딩된 정사각 영상을 리샘플하고
  //    그때 검은 패딩이 경계 픽셀의 필터에 참여한다. 한 번의 drawImage 로 합치면 그
  //    기여가 사라져 테두리가 달라진다.
  const dst = canvasOf(n, n), dc = dst.getContext('2d', { willReadFrequently: true });
  dc.imageSmoothingEnabled = true;
  dc.imageSmoothingQuality = 'high';                       // 학습은 PIL bilinear(antialias)

  if (pre.square_pad) {
    const side = Math.max(box.w, box.h);
    const sq = canvasOf(side, side), qc = sq.getContext('2d', { willReadFrequently: true });
    qc.fillStyle = '#000';
    qc.fillRect(0, 0, side, side);
    qc.drawImage(src, box.x, box.y, box.w, box.h,
                 Math.floor((side - box.w) / 2), Math.floor((side - box.h) / 2), box.w, box.h);
    dc.drawImage(sq, 0, 0, side, side, 0, 0, n, n);         // 3. resize 224
  } else {
    dc.drawImage(src, box.x, box.y, box.w, box.h, 0, 0, n, n);
  }

  const px = dc.getImageData(0, 0, n, n).data;

  // 4. histogram match
  if (pre.histogram_match && ref && ref.cdf) {
    matchHistogram(px, n, ref, pre.color_match_blend === undefined ? 1 : pre.color_match_blend, tol);
  }

  // 5. CLAHE — green 채널만이 기본이다. 안저에서 병변 대비가 가장 큰 채널이다.
  if (pre.clahe) {
    const grid = pre.clahe_grid || [8, 8];
    const chans = pre.clahe_green_only ? [1] : [0, 1, 2];
    for (let ci = 0; ci < chans.length; ci++) {
      const ch = chans[ci];
      const plane = new Uint8Array(n * n);
      for (let i = 0, j = ch; i < n * n; i++, j += 4) plane[i] = px[j];
      const eq = claheChannel(plane, n, n, pre.clahe_clip_limit, grid[0], grid[1]);
      for (let i = 0, j = ch; i < n * n; i++, j += 4) px[j] = eq[i];
    }
  }

  // 6-7. ToTensor + ImageNet 정규화 → NCHW
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
