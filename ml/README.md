# Card detection model

Training pipeline for the Aadhaar card corner detector that replaces (and
falls back to) the hand-written detector in `aadhaar-print-assistant.html`.

## Why this exists

The classical detector plateaued at ~0.91 mean IoU on a 20-image bench, and
its failures are structural rather than tuning problems:

- It separates card from background by **brightness/colour**. A white card on
  a cream cloth cannot be separated that way at all — measured at **0.213
  IoU**, selecting the cloth (73% of the frame) instead of the card. No
  threshold fixes this, because the signal is not there.
- It fits a **rotated rectangle**, so a card photographed at an angle — a real
  trapezoid — can never be fitted exactly.
- Its scoring judges **shape only**, so glare patches, QR blocks and colour
  bands score like cards. Five separate attempts to arbitrate between
  candidates by hand-designed features were measured and all made results
  worse or barely moved them (Hough lines 0.738, edge refinement 0.830–0.903,
  adaptive threshold 0.473, threshold sweep 0.393, higher resolution 0.884).

A learned model addresses all three: it uses shape *and* content, outputs a
region rather than a rectangle, and gives a usable confidence.

## What it produces

`card_seg.onnx` (~1.6 MB fp32, ~0.4 MB int8) — input `1×3×256×256` RGB in
`[0,1]`, output `1×1×64×64` logits, sigmoid → probability that a pixel is card.
The browser thresholds it, takes the largest region and fits a quadrilateral,
reusing code already in the app.

Roughly 0.4M parameters, chosen to stay fast on the cheap Android phones this
is actually used on and small enough to ship as a downloaded asset.

## Privacy

**No customer Aadhaar images are used, at any stage.** Training data is
synthetic (`synth.py`) plus, optionally, the public MIDV datasets, which are
mock and public-domain identity documents. Collecting real cardholder images
as a training set would be legally fraught in India and is unnecessary.

Inference runs **on the device** — no photo is uploaded, and there is no
per-scan cost, which is what makes a ₹99/year price viable. A cloud vision API
would break both properties.

## Run it (Google Colab, free tier is enough)

Runtime → Change runtime type → **T4 GPU**, then:

```python
# 1. dependencies
!pip -q install onnx onnxruntime opencv-python-headless

# 2. get the scripts
!git clone https://github.com/7249999294/Aadhar-cards-Print-Assistance.git repo
%cd repo/ml

# 3. generate training data (~25 min for 30k on Colab CPU)
!python synth.py --n 30000 --out data/synth --seed 0

# 4. train (~35-50 min for 30 epochs on a T4)
!python train.py --data data/synth --epochs 30 --bs 64 --out card_seg.onnx

# 5. download the model
from google.colab import files
files.download('card_seg.onnx')
files.download('card_seg_int8.onnx')
```

Sanity checks while it runs:

- Parameter count should print ~0.4M.
- `val quad-IoU` should pass 0.90 within a few epochs and settle **0.95+**. If
  it stalls below 0.85, the generator and the real photos have diverged —
  fix the generator, not the model.
- `no-detection` should fall to near zero. A high count means the model is
  producing empty masks, which matters more than a mediocre IoU because it is
  the case that must trigger manual mode.

**These scripts have not been executed** — there is no Python in the
development environment they were written in. Expect to fix a small error or
two on the first Colab run.

## Then what

1. Drop `card_seg.onnx` into `public/`.
2. Add ONNX Runtime Web to the app, run the model, fit the quad from the mask.
3. Keep the classical detector as the fallback when model confidence is low —
   the same hybrid arrangement CamScanner uses (their patent describes a CNN
   and a classical path proposing candidates that are then rated together).
4. Re-run the existing 20-image benchmark **and** the light-cloth case that
   currently scores 0.213. Ship only if it beats 0.91 mean and, more
   importantly, lifts the worst case.

## Honest expectations

Synthetic-only training usually leaves a model weak on real camera artefacts
the generator does not imitate. The realistic sequence is:

1. Train on synthetic, confirm it beats 0.91 on the existing bench.
2. Add MIDV-500/2020 for real-camera realism (phase 2).
3. Fine-tune on a few dozen photos of a **dummy or personally-owned card**
   shot in the actual shop, on the actual surfaces — never customer cards.

Step 3 is what typically closes the last gap, and it needs no customer data.

## Files

| file | purpose |
|---|---|
| `synth.py` | generates training images + corner labels |
| `train.py` | dataset, model, training loop, ONNX export |

The generator deliberately covers every condition that has broken the
classical detector: pale/patterned backgrounds, dark textured surfaces, glare,
vignetting, low light, drop shadows, motion blur, JPEG artefacts, laminate
sleeves (labelled as the card, not the sleeve), rounded corners, damaged
cards, clutter, cards from 5% to 78% of the frame, and full rotation.
