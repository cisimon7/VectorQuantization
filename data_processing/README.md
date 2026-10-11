# Oxford Flowers 102 preparation

Processing code lives here. Images, annotations, and prepared arrays live on the
external drive at `/Volumes/PortableSSD/datasets/oxford_flowers102`.
The ignored `dataset.local.json` records this machine's path, resolution, and seed.
On another machine, supply `--root` explicitly.

## Prepare or verify

From the project root, with the external drive mounted:

```sh
source .venv/bin/activate
pip install -e '.[data,dev]'
python -m data_processing.flowers102 download
python -m data_processing.flowers102 prepare
python -m data_processing.flowers102 verify
```

An explicit path works without local configuration:

```sh
python -m data_processing.flowers102 prepare \
  --root /Volumes/PortableSSD/datasets/oxford_flowers102 --size 128 --seed 0
```

Downloads come from the [official Oxford dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
The image archive, class labels, and official split file are checked against the
SHA-256 hashes and byte sizes published by
[TensorFlow Datasets](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/oxford_flowers102/checksums.tsv).
TensorFlow is not required. Interrupted downloads can resume; completed artifacts
are verified and reused. Preparation uses a staging directory and publishes it
only after all checks pass, without overwriting an existing prepared dataset.

## Output

```text
oxford_flowers102/
  raw/
    102flowers.tgz
    imagelabels.mat
    setid.mat
    sources.json
  processed/128_seed0/
    dataset.json
    train.npy, validation.npy, test.npy
    train_labels.npy, validation_labels.npy, test_labels.npy
    train.json, validation.json, test.json
```

Each image array is uint8 NCHW at 128×128. All 8,189 image pixels together occupy
about 384 MiB, rather than approximately 1.5 GiB as float32. The source archive
retains every original JPEG. Preparation decodes directly from the
archive without extracting 8,189 small files, avoiding wasted disk allocation
on the external drive. `vqgan.data.ImageDataset` memory-maps the arrays and converts only
the current batch to float32 `[-1, 1]`.

Preprocessing corrects EXIF orientation, converts to RGB, resizes the shorter side
to 128 using Lanczos while retaining aspect ratio, and center-crops 128×128. This
provides a deterministic baseline. Training applies random horizontal flips;
validation/testing do not augment. Random crop augmentation is not implemented.

Splits use seeded, class-stratified 80/10/10 allocation across the full dataset,
rounding by largest remainder within each class. Labels are zero-based (0–101),
and original Oxford image IDs and official split membership are preserved in the
JSON manifests. This is a **custom reconstruction split**, not Oxford's official
classification benchmark split. Every image occurs once, and all categories
appear in each split. `verify` checks hashes, shapes, manifest/label alignment,
class coverage, and split disjointness.

The prepared seed-0 dataset contains **6,553 training**, **834 validation**, and
**802 test** images. The ratios are approximate because allocation is rounded
within each of the 102 classes. Download checksums and full preparation checks
passed. A short training/evaluation smoke check verifies integration; a useful
model still needs a full training run.

## Train, validate, and test

Training reads only `train.npy`; periodic checks read `validation.npy`. The test
split is consumed only by the separate evaluation command.

```sh
python examples/train.py --model vqgan \
  --dataset-dir /Volumes/PortableSSD/datasets/oxford_flowers102/processed/128_seed0 \
  --steps 30000 --batch-size 2 --disc-start 5000 --val-every 1000 --bf16 \
  --perceptual-weight 1.0 --perceptual-network vgg16 \
  --checkpoint /Volumes/PortableSSD/datasets/oxford_flowers102/checkpoints/train.eqx
```

For prepared data, the example uses a smaller model with base channels 64,
four channel levels, 8× spatial compression, 128-dimensional embeddings, and
1,024 codebook vectors. Bottleneck attention runs at 16×16 for 128×128 inputs.
Use `--base-channels 32` and a smaller batch for a lighter run; use `--bf16` where
supported. Synthetic smoke-test defaults remain small. Resume with `--resume`;
the checkpoint supplies the model architecture, while optimizer and loss settings
must match the previous run.

Perceptual training requires `pip install -e '.[perceptual]'`. The default
`--perceptual-network vgg16` uses frozen pretrained jaxlpips weights; the first
perceptual run downloads and caches those weights. A batch size of 2 is a safe
starting point on an 8 GB GPU because VGG16 adds activation memory.

A pretrained-free structural VQGAN can use `--feature-matching-weight 0.5`,
`--normalize-feature-matching`, `--ms-ssim-weight 0.2`, and
`--laplacian-weight 0.1`. MS-SSIM and Laplacian supervision start immediately;
feature matching follows the configured discriminator warm-up.

To train the VQVAE baseline, use `--model vqvae` and a separate checkpoint path,
for example `checkpoints/flowers-vqvae.eqx`. This creates only the autoencoder and
its optimizer. Keep the seed, model size, batch size, precision, augmentation,
and step count the same across VQVAE/VQGAN runs. Both modes share the prepared
splits, autoencoder architecture, reconstruction/VQ losses, validation code, and
test evaluator. See the [comparison commands](../README.md#train-vqvae-alone-or-compare-with-vqgan)
for paired runs.

The evaluator identifies the model type from its checkpoint automatically; use
the same split and batch size when comparing the two saved models.

Periodic validation evaluates four batches by default; use `--val-batches` to
change this. For full deterministic validation or testing:

```sh
python examples/evaluate.py \
  --dataset-dir /Volumes/PortableSSD/datasets/oxford_flowers102/processed/128_seed0 \
  --checkpoint /Volumes/PortableSSD/datasets/oxford_flowers102/checkpoints/train.eqx \
  --split validation --batch-size 8

python examples/evaluate.py \
  --dataset-dir /Volumes/PortableSSD/datasets/oxford_flowers102/processed/128_seed0 \
  --checkpoint /Volumes/PortableSSD/datasets/oxford_flowers102/checkpoints/train.eqx \
  --split test --batch-size 8 --output test_metrics.json
```

Evaluation reports mean per-image L1, MSE, and PSNR (dynamic range 2), using
unclipped reconstructions. The final batch is padded to preserve the compiled
shape, and padding is excluded from every metric. Evaluation does not update
model parameters or optimizer state. Use validation to choose settings, then
evaluate the held-out test split after choosing the final model.

After the models are trained, generate a multi-sample PDF with aligned
`Test | VQVAE | VQGAN | VQGAN + LPIPS | VQGAN + Feature Matching | VQGAN +
Structural Losses` columns. Long headings wrap within their columns:

```sh
python examples/compare_reconstructions.py \
  --dataset-dir /media/cisimon7/USB/datasets/oxford_flowers102/processed/128_seed0 \
  --vqvae-checkpoint /media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqvae.eqx \
  --vqgan-checkpoint /media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqgan.eqx \
  --vqgan-lpips-checkpoint /media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqgan-lpips-30k.eqx \
  --vqgan-feature-matching-checkpoint /media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqgan-feature-matching-30k.eqx \
  --vqgan-structural-checkpoint /media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqgan-structural-30k.eqx \
  --num-samples 12 --output comparison.pdf
```

Generate a full test-split Markdown scorecard and machine-readable JSON report:

```sh
python examples/evaluate.py \
  --dataset-dir /media/cisimon7/USB/datasets/oxford_flowers102/processed/128_seed0 \
  --checkpoint /media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqvae.eqx \
  --compare-checkpoint /media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqgan.eqx \
  --split test --batch-size 4 \
  --readme data_processing/previews/EVALUATION.md \
  --output data_processing/previews/evaluation.json
```

## COCO 2017 Captions for CLIP experiments

`data_processing.coco_clip` downloads the official COCO 2017 training and
validation images plus caption annotations. It preserves the original JPEGs and
creates deterministic JSONL manifests with one image and all its captions per
row. Random resized crops and CLIP normalization are intentionally deferred to
the future training loader.

```sh
python -m data_processing.coco_clip download \
  --root /media/cisimon7/USB/datasets/coco_2017_clip
python -m data_processing.coco_clip prepare \
  --root /media/cisimon7/USB/datasets/coco_2017_clip
python -m data_processing.coco_clip verify \
  --root /media/cisimon7/USB/datasets/coco_2017_clip
```

The output contains 118,287 training images with 591,753 captions and 5,000
validation images with 25,014 captions. Raw ZIP files remain available for
reproducibility, so allow roughly 40 GB for archives plus extracted files.
