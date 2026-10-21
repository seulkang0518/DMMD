# Local validation

Validation performed on CPU with PyTorch 2.12.0 and torchvision 0.27.0.
This is implementation validation, not evidence of CIFAR-10 generation quality.

Checks:

- The O(ND) linear MMD estimator agrees with an explicit Gram-matrix computation,
  including input gradients.
- A gradient update with the feature objective increases discrepancy when its
  regularizers are disabled.
- The witness gradient penalty has the expected full parameter derivative,
  including the contribution from reference feature means.
- Frozen-network pixel updates have the correct sign and no accidental division
  by the number of particles.
- Interacting references update between particle steps.
- Independent-particle updates do not depend on companion particles.
- The U-Net supports the input gradients and second-order parameter derivatives
  needed by the gradient penalty.
- Two-step synthetic feature training, checkpointing and image previews completed.
- A float32 update with the default 3,572,160-parameter, four-scale pilot U-Net
  also completed on CPU using a two-image synthetic batch.
- Training resumed from step 2 and completed step 3.
- Statistics caching and both interacting/independent PNG generation completed.
- Exported smoke images were checked to be 32×32 RGB.
- Config inheritance, recipe-preserving resume, Python syntax, and the SGE shell
  syntax were checked.
- Global feature-statistic pooling matches a concatenated batch's MMD and L2
  values and gradients. A simulated four-rank SUM checks gradient scaling through
  both the statistic reduction and witness-gradient penalty.
- The four-GPU config resolves to 16 images/rank, global MMD, 16 noise levels and
  250,000 optimizer updates.
- A single-process synthetic update accumulated all 16 noise-level losses with
  the global-statistics code path enabled.

Not verified here:

- Training on downloaded CIFAR-10, CUDA memory use, speed, or resulting FID.
- CUDA/NCCL multi-GPU execution. Local two-process CPU rendezvous failed because
  of the development Mac's networking/name-resolution behavior.
- The optional torch-fidelity/Inception-weight download and FID computation.

Use the synthetic smoke config first in the cluster environment, then a short
CIFAR-10 pilot. The full paper-inspired budget is not necessary for a wiring test.

## Spatial-head ablation checks (2026-10-21)

- All 25 unit checks passed, including the 18 existing checks and seven new head,
  configuration, compatibility and gradient-penalty checks.
- The actual pre-change global-head source and the updated default initialize
  with identical state dictionaries and give bitwise-equal CPU outputs.
- Four-by-four head outputs distinguish spatial rearrangements that global
  pooling removes. The backbone initialization and output dimension are retained.
- The width-128 spatial model completed one synthetic CPU optimizer update with
  all 16 noise levels and finite parameter gradients (batch two for this check).
- The spatial model has 13,984,256 parameters, up from 13,615,616; the feature-head
  test changes capacity as well as spatial sensitivity.
- Old checkpoint configs without head_spatial_size validate as size one. Loading
  their weights and recipe-preserving resume remain supported; size-one to
  size-four resume is rejected.
- Frozen statistics caching and PNG sampling completed on CPU for both the
  trained full-width spatial model and a legacy-config global model.
- This is still a prototype ablation. Batch-64 CUDA memory, throughput, CIFAR-10
  learning and FID for the spatial model have not been verified locally.
