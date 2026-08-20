"""
Synthetic training data for Aadhaar card corner detection.

Why synthetic at all, when MIDV-500/2020 exist: those datasets are real and
valuable, but they contain no Aadhaar cards and none of the specific
conditions this shop actually hits. Every failure this project has had came
from a condition absent from the test set - card filling the frame, glare
sweeping the surface, a multi-colour background, a rough speckled table, and
most recently a *light cream patterned cloth* that the classical detector
cannot separate from a white card at all. A generator lets us produce those
conditions deliberately and in quantity, with exact ground truth.

The plan is to train on synthetic + MIDV together (see midv.py), because
synthetic alone tends to leave a model brittle to real camera artefacts, and
MIDV alone never shows the conditions we care about.

Nothing here uses or requires a real cardholder's data.
"""

import json
import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

CARD_W, CARD_H = 856, 540          # ISO ID-1 proportions, 1.586:1
BG_KINDS = [
    "dark_leather", "wood", "light_cloth", "patterned_cloth",
    "multicolor", "white_table", "concrete", "plain_dark",
]


# --------------------------------------------------------------------------
# card faces
# --------------------------------------------------------------------------
def _text_block(d, x, y, w, h, rng, colour=(35, 35, 35)):
    """A run of dark bars standing in for a line of print."""
    d.rectangle([x, y, x + int(w * rng.uniform(0.6, 1.0)), y + h], fill=colour)


def make_card(front, rng):
    """One card face. Deliberately varied - real cards differ by issue year,
    wear, lamination yellowing and print alignment, and a model that only ever
    saw one exact layout would lean on that layout instead of the geometry."""
    base = rng.randint(232, 250)
    warm = rng.randint(0, 12)                      # ageing / yellowing
    img = Image.new("RGB", (CARD_W, CARD_H), (base, base - warm // 2, base - warm))
    d = ImageDraw.Draw(img)

    # tricolour header bands
    band_y = int(CARD_H * rng.uniform(0.03, 0.07))
    band_h = int(CARD_H * rng.uniform(0.045, 0.065))
    ox = int(CARD_W * rng.uniform(0.26, 0.34))
    ow = int(CARD_W * rng.uniform(0.34, 0.44))
    d.rectangle([ox, band_y, ox + ow, band_y + band_h],
                fill=(rng.randint(215, 240), rng.randint(110, 140), rng.randint(30, 60)))
    d.rectangle([ox - int(CARD_W * 0.02), band_y + int(band_h * 1.35),
                 ox + ow + int(CARD_W * 0.02), band_y + int(band_h * 2.35)],
                fill=(rng.randint(45, 75), rng.randint(140, 165), rng.randint(75, 100)))

    # emblem block, and the Aadhaar roundel on the opposite side
    d.rectangle([int(CARD_W * 0.05), int(CARD_H * 0.05),
                 int(CARD_W * 0.09), int(CARD_H * 0.21)], fill=(85, 82, 76))
    rx = int(CARD_W * rng.uniform(0.86, 0.92))
    d.ellipse([rx - 34, int(CARD_H * 0.06), rx + 34, int(CARD_H * 0.06) + 68],
              fill=(rng.randint(190, 225), rng.randint(60, 95), rng.randint(110, 140)))

    if front:
        # photograph block - a real one is a mid-grey portrait, not flat
        px, py = int(CARD_W * 0.11), int(CARD_H * 0.28)
        pw, ph = int(CARD_W * 0.17), int(CARD_H * 0.46)
        # int16 throughout: adding noise to a uint8 array wraps around instead
        # of clipping, which speckles the portrait with black and white dots
        photo = np.full((ph, pw, 3), rng.randint(120, 165), np.int16)
        photo += (np.random.randn(ph, pw, 3) * 16).astype(np.int16)
        img.paste(Image.fromarray(photo.clip(0, 255).astype(np.uint8)), (px, py))
        d.ellipse([px + pw // 4, py + ph // 5, px + 3 * pw // 4, py + 3 * ph // 5],
                  fill=(rng.randint(85, 120),) * 3)
        for i in range(rng.randint(3, 5)):
            _text_block(d, int(CARD_W * 0.33), int(CARD_H * (0.30 + i * 0.085)),
                        int(CARD_W * 0.36), int(CARD_H * 0.042), rng)
    else:
        for i in range(rng.randint(4, 7)):
            _text_block(d, int(CARD_W * 0.06), int(CARD_H * (0.25 + i * 0.072)),
                        int(CARD_W * 0.34), int(CARD_H * 0.036), rng)
        # QR block - the single most persistent false positive for the
        # classical detector, so the model must see it constantly and learn
        # that it is *inside* the card rather than the card itself
        qs = int(min(CARD_W * rng.uniform(0.24, 0.30), CARD_H * 0.52))
        qx, qy = int(CARD_W * rng.uniform(0.56, 0.66)), int(CARD_H * 0.22)
        d.rectangle([qx - 5, qy - 5, qx + qs + 5, qy + qs + 5], fill=(255, 255, 255))
        cells = rng.randint(30, 45)
        step = qs / cells
        for a in range(cells):
            for b in range(cells):
                if rng.random() > 0.47:
                    d.rectangle([qx + a * step, qy + b * step,
                                 qx + (a + 1) * step, qy + (b + 1) * step], fill=(15, 15, 15))

    # number strip and the red rule beneath it
    d.rectangle([int(CARD_W * 0.30), int(CARD_H * 0.66),
                 int(CARD_W * 0.72), int(CARD_H * 0.735)], fill=(20, 20, 20))
    ry = int(CARD_H * rng.uniform(0.78, 0.82))
    d.rectangle([int(CARD_W * 0.05), ry, int(CARD_W * 0.95), ry + 4], fill=(200, 45, 45))
    d.rectangle([int(CARD_W * 0.28), ry + int(CARD_H * 0.04),
                 int(CARD_W * 0.72), ry + int(CARD_H * 0.09)], fill=(40, 40, 40))

    card = np.array(img)

    # wear: scuffs, a crease, dirt - several of the real samples were damaged
    if rng.random() < 0.35:
        for _ in range(rng.randint(1, 4)):
            x0, y0 = rng.randint(0, CARD_W), rng.randint(0, CARD_H)
            cv2.line(card, (x0, y0),
                     (x0 + rng.randint(-160, 160), y0 + rng.randint(-60, 60)),
                     (rng.randint(200, 245),) * 3, rng.randint(1, 4))
    if rng.random() < 0.15:
        x0 = rng.randint(0, CARD_W - 120)
        y0 = rng.randint(0, CARD_H - 90)
        card[y0:y0 + rng.randint(30, 90), x0:x0 + rng.randint(40, 120)] = rng.randint(235, 252)
    # PIL draws in RGB, everything downstream (compositing, imwrite) is
    # OpenCV BGR. Without this the saffron band trains as blue.
    return cv2.cvtColor(card, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# backgrounds
# --------------------------------------------------------------------------
def make_background(kind, W, H, rng):
    bg = np.zeros((H, W, 3), np.uint8)
    if kind == "dark_leather":
        bg[:] = (rng.randint(28, 52),) * 3
        noise = (np.random.randn(H, W, 1) * rng.uniform(9, 20)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif kind == "plain_dark":
        bg[:] = (rng.randint(18, 40),) * 3
        bg = np.clip(bg.astype(np.int16) +
                     (np.random.randn(H, W, 1) * 5).astype(np.int16), 0, 255).astype(np.uint8)
    elif kind == "wood":
        c1 = np.array([rng.randint(70, 110), rng.randint(50, 85), rng.randint(30, 60)])
        c2 = np.array([rng.randint(140, 190), rng.randint(105, 150), rng.randint(65, 105)])
        ramp = np.linspace(0, 1, W)[None, :, None]
        bg = (c1 + (c2 - c1) * ramp).astype(np.uint8).repeat(H, 0)
        for y in range(0, H, rng.randint(5, 11)):
            cv2.line(bg, (0, y + rng.randint(-2, 2)), (W, y + rng.randint(-2, 2)),
                     (max(0, int(bg[y, 0, 0]) - 30),) * 3, 1)
    elif kind in ("light_cloth", "patterned_cloth", "white_table"):
        # The case that defeats the classical detector entirely: a pale
        # surface a white card cannot be thresholded away from.
        tone = rng.randint(205, 244)
        bg[:] = (tone, tone - rng.randint(0, 10), tone - rng.randint(4, 22))
        bg = np.clip(bg.astype(np.int16) +
                     (np.random.randn(H, W, 1) * rng.uniform(4, 12)).astype(np.int16),
                     0, 255).astype(np.uint8)
        if kind == "patterned_cloth":
            step = int(W * rng.uniform(0.06, 0.11))
            sq = max(3, int(step * rng.uniform(0.25, 0.42)))
            col = (rng.randint(140, 200), rng.randint(85, 140), rng.randint(55, 110))  # BGR blue
            for gy in range(0, H, step):
                for gx in range(0, W, step):
                    if rng.random() > 0.12:
                        bg[gy:gy + sq, gx:gx + sq] = col
            for _ in range(rng.randint(2, 5)):        # woven bands
                y0 = rng.randint(0, H - 20)
                bg[y0:y0 + rng.randint(8, 30)] = (rng.randint(150, 200),
                                                  rng.randint(45, 90), rng.randint(45, 80))
            for _ in range(rng.randint(3, 9)):        # floral motifs
                cv2.circle(bg, (rng.randint(0, W), rng.randint(0, H)),
                           rng.randint(int(W * 0.04), int(W * 0.11)),
                           (rng.randint(40, 220), rng.randint(40, 200), rng.randint(30, 120)), -1)
    elif kind == "multicolor":
        for _ in range(rng.randint(18, 40)):
            cv2.circle(bg, (rng.randint(0, W), rng.randint(0, H)),
                       rng.randint(int(W * 0.08), int(W * 0.35)),
                       (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)), -1)
        bg = cv2.GaussianBlur(bg, (0, 0), W * 0.02)
    elif kind == "concrete":
        bg[:] = (rng.randint(95, 150),) * 3
        bg = np.clip(bg.astype(np.int16) +
                     (np.random.randn(H, W, 1) * 22).astype(np.int16), 0, 255).astype(np.uint8)
        for _ in range(rng.randint(200, 900)):
            cv2.circle(bg, (rng.randint(0, W), rng.randint(0, H)), rng.randint(1, 4),
                       (rng.randint(60, 200),) * 3, -1)
    return bg


def add_clutter(bg, rng):
    H, W = bg.shape[:2]
    for _ in range(rng.randint(0, 3)):
        w, h = rng.randint(int(W * 0.08), int(W * 0.3)), rng.randint(int(H * 0.03), int(H * 0.12))
        x, y = rng.randint(0, max(1, W - w)), rng.randint(0, max(1, int(H * 0.25)))
        shade = rng.choice([rng.randint(10, 45), rng.randint(215, 250)])
        cv2.rectangle(bg, (x, y), (x + w, y + h), (shade,) * 3, -1)
    return bg


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------
def _perspective_quad(W, H, rng):
    """Place the card as a genuine quadrilateral. Scale reaches down to 5% of
    the frame because operators photograph small cards from a distance, and
    up to 78% because they also fill the frame - both have been real failures."""
    frac = rng.uniform(0.05, 0.78)
    cw = math.sqrt(frac * W * H * 1.586)
    ch = cw / 1.586
    cw, ch = min(cw, W * 0.95), min(ch, H * 0.95)
    cx = rng.uniform(cw / 2, W - cw / 2)
    cy = rng.uniform(ch / 2, H - ch / 2)
    ang = math.radians(rng.uniform(-180, 180) if rng.random() < 0.25
                       else rng.uniform(-25, 25))
    pts = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2],
                    [cw / 2, ch / 2], [-cw / 2, ch / 2]], np.float32)
    jitter = min(cw, ch) * rng.uniform(0.0, 0.20)     # perspective foreshortening
    pts += np.random.uniform(-jitter, jitter, pts.shape).astype(np.float32)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], np.float32)
    return (pts @ R.T) + np.array([cx, cy], np.float32)


def compose(rng, W=None, H=None):
    if W is None:
        W, H = rng.choice([(480, 640), (640, 480), (512, 512), (448, 640), (640, 448)])
    kind = rng.choice(BG_KINDS)
    bg = add_clutter(make_background(kind, W, H, rng), rng)

    front = rng.random() < 0.5
    card = make_card(front, rng)
    quad = _perspective_quad(W, H, rng)

    src = np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))

    # rounded corners, so the model never learns to expect a sharp 90-degree tip
    card_a = np.dstack([card, np.full(card.shape[:2], 255, np.uint8)])
    r = int(CARD_H * rng.uniform(0.03, 0.09))
    mask_r = np.zeros(card.shape[:2], np.uint8)
    cv2.rectangle(mask_r, (r, 0), (CARD_W - r, CARD_H), 255, -1)
    cv2.rectangle(mask_r, (0, r), (CARD_W, CARD_H - r), 255, -1)
    for cxr, cyr in [(r, r), (CARD_W - r, r), (r, CARD_H - r), (CARD_W - r, CARD_H - r)]:
        cv2.circle(mask_r, (cxr, cyr), r, 255, -1)
    card_a[..., 3] = mask_r

    warped = cv2.warpPerspective(card_a, M, (W, H), flags=cv2.INTER_LINEAR)
    alpha = (warped[..., 3:4].astype(np.float32) / 255.0)

    # drop shadow, offset - it out-contrasts the card edge and has repeatedly
    # pulled the classical detector outward
    if rng.random() < 0.8:
        sh = cv2.warpPerspective(mask_r, M, (W, H))
        k = int(max(3, min(W, H) * rng.uniform(0.01, 0.035))) | 1
        sh = cv2.GaussianBlur(sh, (k, k), 0).astype(np.float32) / 255.0
        off = int(min(W, H) * rng.uniform(0.004, 0.018))
        sh = np.roll(np.roll(sh, off, 0), off, 1)[..., None]
        bg = (bg.astype(np.float32) * (1 - sh * rng.uniform(0.25, 0.6))).astype(np.uint8)

    img = (bg.astype(np.float32) * (1 - alpha) + warped[..., :3].astype(np.float32) * alpha)

    # laminate sleeve: a second boundary just outside the card. The label
    # stays the CARD, so the model is taught to prefer the card over the pouch.
    if rng.random() < 0.3:
        pad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
        centre = quad.mean(0)
        pq = centre + (quad - centre) * rng.uniform(1.05, 1.18) + pad * 2
        pm = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(pm, pq.astype(np.int32), 255)
        pmf = (pm[..., None].astype(np.float32) / 255.0) * rng.uniform(0.10, 0.30)
        img = img * (1 - pmf) + 245 * pmf
        # draw on uint8 and convert back: OpenCV's drawing calls want an 8-bit
        # contiguous array, and img is a float blend at this point
        tmp = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
        cv2.polylines(tmp, [pq.astype(np.int32)], True, (250, 250, 250), rng.randint(1, 3))
        img = tmp.astype(np.float32)

    img = _lighting(img, rng)
    img = _camera(img, rng)
    return img.astype(np.uint8), quad


def _lighting(img, rng):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if rng.random() < 0.7:                      # directional falloff
        ang = rng.uniform(0, 2 * math.pi)
        ramp = (xx * math.cos(ang) + yy * math.sin(ang))
        ramp = (ramp - ramp.min()) / (ramp.ptp() + 1e-6)
        img = img * (1 - ramp[..., None] * rng.uniform(0.15, 0.6))
    if rng.random() < 0.55:                     # vignette
        d = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        d = d / (d.max() + 1e-6)
        img = img * (1 - (d ** 2)[..., None] * rng.uniform(0.2, 0.65))
    if rng.random() < 0.4:                      # glare / specular blob
        gx, gy = rng.uniform(0, W), rng.uniform(0, H)
        rad = min(W, H) * rng.uniform(0.15, 0.55)
        d = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
        g = np.clip(1 - d / rad, 0, 1) ** 2 * rng.uniform(0.35, 0.95)
        img = img + (255 - img) * g[..., None]
    if rng.random() < 0.25:                     # low light
        img = img * rng.uniform(0.28, 0.55)
    return np.clip(img, 0, 255)


def _camera(img, rng):
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:                      # motion blur
        k = rng.randint(3, 9)
        kern = np.zeros((k, k), np.float32)
        kern[k // 2] = 1.0 / k
        if rng.random() < 0.5:
            kern = kern.T
        img = cv2.filter2D(img, -1, kern)
    img = img + np.random.randn(*img.shape) * rng.uniform(2, 12)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.8:                      # JPEG artefacts, as from a phone
        q = rng.randint(45, 92)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img.astype(np.float32)


def generate(n, out_dir, seed=0):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rng = random.Random(seed)
    np.random.seed(seed)
    labels = []
    for i in range(n):
        img, quad = compose(rng)
        name = f"{i:06d}.jpg"
        cv2.imwrite(os.path.join(out_dir, "images", name), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        H, W = img.shape[:2]
        labels.append({"file": name, "w": W, "h": H,
                       "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in quad]})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    print(f"wrote {n} samples to {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--out", default="data/synth")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    generate(a.n, a.out, a.seed)
