"""
Trains a small card-segmentation model and exports it to ONNX for the browser.

Design choices worth knowing:

* It predicts a MASK ("is this pixel card?"), not four corner coordinates
  directly. Regression to 8 numbers is simpler but gives no way to say "no
  card here" and no natural confidence, and confidence is the thing this
  project most needs - a wrong crop labelled "very good" is worse than a
  wrong crop labelled "please check". A mask also lets us reuse the
  quad-fitting already written in JS, and degrades gracefully: a partly wrong
  mask still yields roughly the right quadrilateral.

* It is deliberately tiny (~0.4M params, ~1.6MB fp32). This runs on a cheap
  Android phone in a print shop, not a workstation, and it ships as a file the
  browser downloads. Accuracy past a point is worth less here than staying
  small and fast.

* Input is letterboxed, never stretched. Stretching would change the card's
  aspect ratio, which is one of the few strong priors available.

Run:
    python synth.py --n 30000 --out data/synth
    python train.py --data data/synth --epochs 30 --out card_seg.onnx
"""

import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IN_SIZE = 256          # network input, letterboxed
OUT_SIZE = 64          # mask resolution; quad fitting doesn't need more


# --------------------------------------------------------------------------
# shared letterbox convention - the browser MUST match this exactly
# --------------------------------------------------------------------------
def letterbox(img, size=IN_SIZE):
    h, w = img.shape[:2]
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), img.dtype)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out, s, ox, oy


class CardData(Dataset):
    def __init__(self, root, train=True):
        with open(os.path.join(root, "labels.json")) as f:
            self.items = json.load(f)
        cut = int(len(self.items) * 0.97)
        self.items = self.items[:cut] if train else self.items[cut:]
        self.root = root
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(os.path.join(self.root, "images", it["file"]), cv2.IMREAD_COLOR)
        quad = np.array(it["quad"], np.float32)

        if self.train and np.random.rand() < 0.5:          # horizontal flip
            img = img[:, ::-1].copy()
            quad[:, 0] = img.shape[1] - quad[:, 0]
            quad = quad[[1, 0, 3, 2]]

        img, s, ox, oy = letterbox(img)
        quad = quad * s + np.array([ox, oy], np.float32)

        if self.train:                                     # mild photometric jitter
            img = img.astype(np.float32)
            img = img * np.random.uniform(0.8, 1.2) + np.random.uniform(-18, 18)
            img = np.clip(img, 0, 255)

        mask = np.zeros((OUT_SIZE, OUT_SIZE), np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad * (OUT_SIZE / IN_SIZE)).astype(np.int32), 1)

        x = torch.from_numpy(np.ascontiguousarray(img.astype(np.float32).transpose(2, 0, 1)) / 255.0)
        y = torch.from_numpy(mask.astype(np.float32))[None]
        return x, y


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CardSeg(nn.Module):
    def __init__(self, c=(16, 32, 64, 96)):
        super().__init__()
        c1, c2, c3, c4 = c
        self.e1 = nn.Sequential(block(3, c1, 2), block(c1, c1))        # 128
        self.e2 = nn.Sequential(block(c1, c2, 2), block(c2, c2))       # 64
        self.e3 = nn.Sequential(block(c2, c3, 2), block(c3, c3))       # 32
        self.e4 = nn.Sequential(block(c3, c4, 2), block(c4, c4))       # 16
        self.d3 = block(c4 + c3, c3)
        self.d2 = block(c3 + c2, c2)
        self.head = nn.Conv2d(c2, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        u3 = F.interpolate(e4, scale_factor=2, mode="nearest")
        d3 = self.d3(torch.cat([u3, e3], 1))
        u2 = F.interpolate(d3, scale_factor=2, mode="nearest")
        d2 = self.d2(torch.cat([u2, e2], 1))
        return self.head(d2)                                            # B,1,64,64 logits


def dice_bce(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + 1
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1
    return bce + (1 - num / den).mean()


# --------------------------------------------------------------------------
# quad fitting + IoU, so training reports the number we actually care about
# --------------------------------------------------------------------------
def mask_to_quad(prob, thr=0.5):
    m = (prob > thr).astype(np.uint8)
    if m.sum() < 12:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def quad_iou(a, b, size=OUT_SIZE):
    ma = np.zeros((size, size), np.uint8)
    mb = np.zeros((size, size), np.uint8)
    cv2.fillConvexPoly(ma, np.round(a).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, np.round(b).astype(np.int32), 1)
    u = (ma | mb).sum()
    return float((ma & mb).sum()) / u if u else 0.0


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    ious, miss = [], 0
    for x, y in loader:
        prob = torch.sigmoid(model(x.to(dev))).cpu().numpy()[:, 0]
        tgt = y.numpy()[:, 0]
        for p, t in zip(prob, tgt):
            gq = mask_to_quad(t, 0.5)
            pq = mask_to_quad(p, 0.5)
            if gq is None:
                continue
            if pq is None:
                miss += 1
                ious.append(0.0)
            else:
                ious.append(quad_iou(pq, gq))
    return float(np.mean(ious)) if ious else 0.0, miss, len(ious)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="card_seg.onnx")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    tr = DataLoader(CardData(a.data, True), batch_size=a.bs, shuffle=True,
                    num_workers=2, drop_last=True, pin_memory=True)
    va = DataLoader(CardData(a.data, False), batch_size=a.bs, num_workers=2)

    model = CardSeg().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_par/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(tr))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    best = -1.0

    for ep in range(a.epochs):
        model.train()
        run = 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                loss = dice_bce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        miou, miss, n = evaluate(model, va, dev)
        print(f"epoch {ep+1:3d}/{a.epochs}  loss {run/len(tr):.4f}  val quad-IoU {miou:.4f}"
              f"  no-detection {miss}/{n}")
        if miou > best:
            best = miou
            torch.save(model.state_dict(), "card_seg_best.pt")

    print(f"best val quad-IoU: {best:.4f}")
    model.load_state_dict(torch.load("card_seg_best.pt", map_location=dev))
    model.eval().cpu()

    dummy = torch.zeros(1, 3, IN_SIZE, IN_SIZE)
    torch.onnx.export(
        model, dummy, a.out,
        input_names=["image"], output_names=["mask"],
        opset_version=12,                 # widely supported by onnxruntime-web
        do_constant_folding=True,
    )
    print("exported", a.out, f"({os.path.getsize(a.out)/1e6:.2f} MB)")

    # int8 is roughly a quarter the size; check its IoU before shipping it,
    # a smaller file is not worth a worse crop
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = a.out.replace(".onnx", "_int8.onnx")
        quantize_dynamic(a.out, q, weight_type=QuantType.QUInt8)
        print("exported", q, f"({os.path.getsize(q)/1e6:.2f} MB)")
    except Exception as e:
        print("int8 quantization skipped:", e)


if __name__ == "__main__":
    main()
