# CIFAR-10 DMMD feature-learning experiment

An **independent, paper-inspired prototype** for our first research question:
can a noise-conditioned comparison network provide useful MMD gradients for
CIFAR-10 generation?

This trains the feature network and tests its particle flow. It does not
implement a MeanFlow student, use pretrained MAE/VAE features, or claim the
authors' reported FID. No public author repository was verified when preparing it.

## What is implemented

1. Learn a time-conditioned U-Net feature map from real and corrupted images.
2. Minimize negative MMD squared plus feature L2 and witness-gradient penalties.
3. Freeze that map and move pixel particles using the MMD witness gradient.
4. Optionally substitute cached corrupted-real means for the current-particle mean
   to test the paper's approximate independent-particle sampler.

The pilot uses 32×32 RGB pixels, 128 features, and a linear base kernel.
**Larger feature-training MMD alone is not evidence of good generation.**
Inspect particle previews, gradient magnitudes, and eventually sample quality/FID.

## Files

    configs/pilot.yaml          default 20k-step small-U-Net experiment
    configs/paper_inspired.yaml selected reported settings; much more expensive
    configs/cifar10_4gpu.yaml    full-budget four-GPU configuration, global batch 64
    configs/smoke.yaml          two-step offline synthetic wiring test
    dmmd/model.py              time-conditioned multi-scale feature U-Net
    dmmd/noise.py              linear-beta DDPM corruption schedule
    dmmd/objective.py          MMD estimator and both penalties
    dmmd/sampling.py           interacting and independent particle updates
    train.py                  feature training, resume, logs and previews
    prepare_data.py            download CIFAR-10 separately
    cache_stats.py             checkpoint-specific clean/noisy feature statistics
    sample.py                  generated images and sampling metadata
    evaluate_fid.py            optional FID/IS against CIFAR-10 training images
    scripts/train_sge.sh       SGE job supporting one, two or four GPUs
    scripts/train_sge_4gpu.sh  four-GPU full-budget job
    tests/test_core.py         mathematical and gradient checks

## 1. Environment and data

Run commands from this folder. Use Python 3.10+ and a matching PyTorch/torchvision
installation. On the cluster you can start from your existing CUDA environment:

    conda activate mmd_flow
    python -m pip install -r requirements.txt
    python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"

If PyTorch is not installed, install the appropriate CUDA build using the
[official installer](https://pytorch.org/get-started/locally/), then install
remaining requirements. This project uses float32: its gradient penalty requires
second derivatives. AMP and torch.compile are not enabled.

Download data once on a machine with network access:

    python prepare_data.py --data-root /path/to/cifar10_data

The compute job can then run offline. That directory should contain
cifar-10-batches-py/. Relative data/workdir/output paths resolve from this
experiment folder; absolute paths remain absolute.

## 2. Offline wiring test

    python -m unittest discover -s tests -v
    python train.py --config configs/smoke.yaml --workdir runs/smoke

The smoke config uses synthetic tensors without downloading data. Its images are
not evidence of quality. Use a new workdir for a fresh test.

## 3. Train features on one GPU

    python -u train.py \
      --config configs/pilot.yaml \
      --data-root /path/to/cifar10_data \
      --workdir runs/cifar10_pilot

For an initial short run, append --steps 2000. Increase the total budget later
with resume. Neither 2k nor 20k steps is guaranteed sufficient for good generation;
they are pilot budgets rather than the paper's full training duration.

Outputs include:

- config.yaml: resolved configuration.
- train.jsonl: loss, MMD squared, penalties, gradient norms and throughput.
- validation.jsonl: held-out feature diagnostics and preview traces.
- checkpoints/latest.pt: resumable model/optimizer checkpoint.
- checkpoints/step_XXXXXXXX.pt: last three periodic checkpoints by default.
- previews/step_XXXXXXXX/generated.png: generation from fixed noise.
- previews/step_XXXXXXXX/reference.png: fixed real reference images.

Previews use 16 interacting particles and a small reference set. They are cheap
diagnostics, not the final 200-particle sampling protocol. Preview randomness is
isolated from the training RNG.

Resume after interruption or a cluster time limit:

    python -u train.py \
      --config configs/pilot.yaml \
      --data-root /path/to/cifar10_data \
      --workdir runs/cifar10_pilot --resume --steps 20000

The steps argument is the total target, not extra steps. Resume rejects changes
to the model, noise schedule, loss, learning rate and training-batch recipe.
Start a separate workdir to compare another recipe.

Model/optimizer step and per-rank RNG states are saved. DataLoader worker
augmentation and prefetched batches are not saved, so resume is not guaranteed
bitwise identical to uninterrupted training.

## 4. One, two or four GPUs on SGE

**For the available four-GPU allocation, use the full-budget configuration:**

    export DATA_ROOT=/path/to/cifar10_data
    export DMMD_CONDA_ENV=mmd_flow
    qsub scripts/train_sge_4gpu.sh

This selects configs/cifar10_4gpu.yaml: 250,000 updates, 16 sampled noise levels
per update, U-Net base width 128 and all 50K training images. Each rank loads
16 images, giving a global batch of 64.

The four-GPU configuration pools differentiable feature sums and squared norms
across ranks before computing MMD, feature L2 and the witness reference means.
Its gradient penalty averages local query penalties against those global means.
The resulting update targets one 64-image comparison, rather than four independent
16-image comparisons. Levels are processed sequentially with gradient accumulation
before one optimizer step, which avoids storing 16 separate computation graphs.

The configuration requires four processes and rejects an accidental one-process
training launch. Statistics caching and sampling can still use a single GPU.
The U-Net is still our independent architecture, so matching these hyperparameters
does not make this an exact paper reproduction.

For an interactive four-GPU allocation:

    python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
      --config configs/cifar10_4gpu.yaml --data-root /path/to/cifar10_data \
      --workdir runs/cifar10_full_4gpu

The same selection works with qsub -pe gpu 4 scripts/train_sge.sh. If CONFIG is
already exported in your shell, unset it or set it explicitly to
configs/cifar10_4gpu.yaml so it does not override the intended configuration.

### Earlier one/two-GPU pilot options

Submit from this folder. The example uses the GPU/account directives from your
existing cluster scripts; adjust them if your cluster configuration differs.

    export DATA_ROOT=/path/to/cifar10_data
    export DMMD_CONDA_ENV=mmd_flow
    qsub scripts/train_sge.sh

The default run name contains the job ID. To select a stable path:

    export WORKDIR=/path/to/runs/cifar10_pilot
    qsub scripts/train_sge.sh

To resume that same path in a later job:

    export RESUME=auto
    qsub scripts/train_sge.sh

For two GPUs:

    qsub -pe gpu 2 scripts/train_sge.sh

Or, inside an allocation:

    python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
      --config configs/pilot.yaml --data-root /path/to/cifar10_data \
      --workdir runs/cifar10_two_gpu

**Two-GPU semantics:** batch size is per rank: the default is 64 images/rank and
128 images/update globally on two GPUs. Each rank computes an MMD and gradient
penalty from its own batch; parameter gradients are averaged after backward.
This is not an all-gathered global MMD comparison. To use 64 images globally,
append --set train.batch_size=32 to the direct torchrun command. Its two local
32-image comparisons still differ from one 64-image comparison.
Independent one-GPU runs are a simple alternative for controlled pilots.
This paragraph describes the local-MMD pilot; the new four-GPU recipe explicitly
uses global MMD statistics.

Rendezvous uses a free port rather than a hard-coded shared port.
SIGTERM/SIGINT/SGE notifications request a checkpoint after the current update;
abrupt SIGKILL requires resuming from the last periodic checkpoint.
Parameter gradients are explicitly all-reduced after backward to handle the
multiple forwards and second-order gradient penalty.

The one-process CPU path was smoke-tested locally. The two-process launcher
could not finish local rendezvous on the development Mac, so CUDA/NCCL and the
two-GPU path still need their first check on the cluster. An OS file lock prevents
two jobs from writing to the same workdir and releases automatically on exit.

## 5. Cache statistics for a selected checkpoint

    python cache_stats.py \
      --checkpoint runs/cifar10_pilot/checkpoints/step_00002000.pt \
      --data-root /path/to/cifar10_data \
      --output runs/cifar10_pilot/stats_step2000.pt

This caches clean and corrupted feature means at the sampler's noise levels.
The pilot uses 1,024 unaugmented training-split references. Use --reference-size 0
for all training-split references. The pilot reserves 5,000 training images for
feature validation, leaving 45,000 for training and these reference statistics.

Statistics are tied to the checkpoint's SHA-256 hash, architecture and noise
schedule. Regenerate them for another checkpoint, including when latest.pt
changes. To change the grid, pass --noise-levels, --t-min and --t-max here.
Sampling uses the grid saved in the statistics.

Noisy means use one corruption per reference image per level. More references
can reduce Monte Carlo error at extra precomputation cost.

## 6. Generate images

Interacting flow:

    python sample.py \
      --checkpoint runs/cifar10_pilot/checkpoints/step_00002000.pt \
      --stats runs/cifar10_pilot/stats_step2000.pt \
      --output runs/cifar10_pilot/samples_step2000 \
      --mode interacting --num-samples 1024

Independent approximate flow:

    python sample.py \
      --checkpoint runs/cifar10_pilot/checkpoints/step_00002000.pt \
      --stats runs/cifar10_pilot/stats_step2000.pt \
      --output runs/cifar10_pilot/samples_step2000_independent \
      --mode independent --num-samples 1024

Both use frozen features and update pixels. The independent variant substitutes
the noisy-real mean for the current generated mean, changing the movement field.
It does not guarantee that four particles reproduce a large interacting flow.
One independent particle can still need many pixel updates.

Default sampling: 10 noise levels, 10 updates/level, step size 0.1. Adjust
--step-size, --steps-per-level or --particles in a separate output folder.
The paper separately tunes the independent sampler; sharing settings initially
is a controlled starting comparison, not a claim of optimality.

Outputs: individual uint8 PNGs in images/, grid.png, and metadata.json with the
seed, checkpoint, sampling grid and pixel/gradient diagnostics. The optional
--save-tensors flag saves float particle batches too.
Pixels are clamped only for PNG export. The flow itself is not clipped by default.
Interacting groups always have the configured particle count; the last group is
trimmed only when saving.

## 7. Optional FID/IS

    python -m pip install -r requirements-fid.txt
    python evaluate_fid.py \
      --samples runs/cifar10_pilot/samples_step2000 \
      --data-root /path/to/cifar10_data

Metrics use torch-fidelity's default Inception implementation and all 50,000
unaugmented CIFAR-10 training images as reference. The first use may download
Inception weights: prepare its cache on a network-enabled machine if necessary.
A 1,024-sample FID is noisy screening evidence. Label the sample count and use a
larger fixed count for final comparisons. To generate 50K images, pass
--num-samples 50000 to sample.py.

Keep sample count, reference, PNG handling and feature extractor fixed across
comparisons. The pilot split and simplified U-Net also differ from the paper.

## Scientific details and deliberate deviations

- The sign follows main-text Eq. (11): minimize **negative MMD squared** when
  learning features. Appendix expansions labelled negative MMD have inconsistent
  displayed signs; this implementation follows the stated maximization objective.
- The estimator omits within-set diagonals and can be negative; it is not clamped.
  It is unbiased for independent two-sample draws. The paper-style pairing of a
  clean batch with its own corruptions induces finite-batch dependence in cross
  terms, so that pairing is not an exactly unbiased population estimate.
- Feature L2 sums the two mean squared vector norms, without dividing by feature
  dimension. The witness gradient penalty targets input-gradient norm 1.
  Feature means retain parameter gradients during penalty training; they are
  detached only for pixel sampling with frozen network weights.
- One scalar noise level is shared inside each comparison. Multiple levels per
  update are averaged before the optimizer step.
- Class labels are ignored. There is no classification or reconstruction head.
- The default U-Net uses GroupNorm, sinusoidal time embeddings and four globally
  pooled heads. The spatial-head ablation below replaces global pooling with a
  4x4 grid before each head's projection. Simplified blocks and lack of attention are implementation choices,
  not an exact reconstruction of unavailable author code.
- Pilot: width 64, 20K steps, 1 level/update, 45K training images.
  Paper-inspired config: width 128, 250K steps, 16 levels/update, all 50K images.
  The latter matches selected reported settings, not the exact architecture.
  The four-GPU configuration distributes this full budget with global batch 64
  and global feature statistics.
- Corruption interpolates a standard 1000-step linear-beta DDPM alpha-bar table.
  Exact author time indexing is unverified. Time 0 is exactly clean; time 1 is
  nearly Gaussian.
- No final high-step-size denoising extension, EMA, feature pretraining or
  MeanFlow student is included. Those are separate experiments.

## Spatial feature-head ablation (one GPU, 10K updates)

This is a controlled alternative to the existing global-head 10K run, not a
claimed reproduction or a guaranteed remedy for its samples. It tests whether
retaining coarse spatial positions helps distribution matching. Only the heads
change scientifically: normalization and activation are followed by adaptive
pooling to 4x4, flattening, then projection to 32 features per head (128 total).
The backbone, noise schedule, data, seed, batch size 64, 16 noise levels per
update, learning rate 1e-4 and regularization remain as in the one-GPU baseline.
The model has 13,984,256 parameters versus 13,615,616 for global pooling; the
368,640 extra parameters mean this test changes head capacity as well as layout
sensitivity. The config retains the 4K and 10K checkpoints and requires one rank.

Use a new workdir and start from scratch. First check the full batch on the GPU:

    python -u train.py --config configs/cifar10_spatial4_1gpu_10k.yaml \
      --workdir runs/spatial4_memory_check --device cuda --steps 2 \
      --set train.log_every=1 --set train.eval_every=0

Then launch the actual 10K run:

    nohup python -u train.py --config configs/cifar10_spatial4_1gpu_10k.yaml \
      --workdir runs/cifar10_spatial4_10k --device cuda \
      > cifar10_spatial4_10k.log 2>&1 &

`head_spatial_size` defaults to 1. Existing global-head checkpoints remain
loadable for sampling, caching, diagnostics and a recipe-preserving resume.
Resuming a global-head checkpoint as a spatial-head model is rejected.

At 4K and 10K, cache reference means from all 50K training images and use the same
standalone sampler as for the baseline: interacting particles 200, 10 time
levels from 0.9 to 0.1, 10 steps/level, step size 0.1, seed 100042. Example at 10K:

    python -u cache_stats.py \
      --checkpoint runs/cifar10_spatial4_10k/checkpoints/step_00010000.pt \
      --output runs/eval_spatial4_10000/stats.pt --device cuda --reference-size 0
    python -u sample.py \
      --checkpoint runs/cifar10_spatial4_10k/checkpoints/step_00010000.pt \
      --stats runs/eval_spatial4_10000/stats.pt \
      --output runs/eval_spatial4_10000/samples_1024 --device cuda \
      --mode interacting --num-samples 1024 --particles 200 \
      --steps-per-level 10 --step-size 0.1 --seed 100042

For screening FID, generate the same 1,024 samples from the baseline using its
own checkpoint and cached means in a separate output folder. Existing 200-image
grids are useful visual references but are not a matched FID sample count.
Install `requirements-fid.txt`, then run `evaluate_fid.py --samples OUTPUT_DIR`
for both directories with the same reference and metric settings. A 1,024-image
FID is a noisy screening metric, not a replacement for final 50K evaluation.

For the existing 4K comparison, `compare_heads.py` validates both runs and reuses
the full-reference feature caches already computed. Keep GPU training paused.
Its defaults find the preserved global checkpoint at
`runs/eval_4000/checkpoint.pt` (or its original training checkpoint if available),
the spatial checkpoint under `runs/cifar10_spatial4_10k/checkpoints/`, and caches
under `runs/eval_4000/` and `runs/eval_spatial4_4000/`.

    python -m pip install -r requirements-fid.txt
    python -u compare_heads.py --step 4000 --validate-only
    nohup python -u compare_heads.py --step 4000 > compare_heads_4000.log 2>&1 &
    tail -f compare_heads_4000.log

The helper generates 1,024 images per model with identical sampling settings and
seed, evaluates each against all 50,000 unaugmented CIFAR-10 training images with
the existing FID evaluator, prints a two-row table, and saves
`runs/compare_heads_4000_1024/comparison.json`. The first metric run may download
Inception weights. This is a small-sample screening comparison, not final FID.
Override paths with `--baseline-checkpoint`, `--spatial-checkpoint`,
`--baseline-stats`, `--spatial-stats` or `--data-root` if files were moved.
Complete protocol-checked sample directories are reusable. Partial or mismatched
outputs are rejected; choose a fresh `--output-root` instead of mixing runs.
The helper never trains models or modifies checkpoints.

The changes are CPU-tested for exact legacy-head arithmetic, legacy checkpoint
loading/resume, sensitivity to spatial rearrangement, full second-order gradient
penalty, and frozen-network pixel sampling. CUDA memory and image quality for the
spatial variant still need to be checked on the experiment machine.

## Sources

- [DMMD paper](https://arxiv.org/abs/2405.06780): Sections 4.1, 4.2 and 5;
  Appendices B.2–B.4 and E.
- [PyTorch input gradients](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).
- [Torchvision CIFAR-10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
- [torch-fidelity API](https://torch-fidelity.readthedocs.io/en/latest/usage_api.html).
