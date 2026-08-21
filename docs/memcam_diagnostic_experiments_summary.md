# MemCam Diagnostic Experiments: Consolidated Results

Reviewer-facing risks, unsupported claims, and the experiments required to
resolve them are tracked in
[`iclr_reviewer_risk_register.md`](iclr_reviewer_risk_register.md). That file
is the decision record for what may enter the manuscript.

## Bottom line

The strongest current evidence is that unbounded MemCam increasingly selects
generated memory frames whose image content is corrupted relative to the
dataset ground truth. RI and especially SLAM select substantially cleaner
frame indices, even when every policy is evaluated using images from the same
unbounded rollout. This explains an important part of why bounded policies can
beat unbounded memory, but the causal propagation test is not complete.

The analysis does **not** establish that there is one ground-truth useful
memory for each target. The retention/retrieval decomposition uses a hindsight
DINO best-match as a diagnostic proxy. Earlier analyses that treated the
unbounded selector as an oracle should not be used as evidence of correctness.

## Definitions used below

- **View mismatch:** DINO distance between the dataset ground-truth image at
  the selected memory index and the dataset ground-truth target image. This
  measures whether the selected index depicts a similar view.
- **Memory corruption:** DINO distance between the generated memory image and
  the dataset ground-truth image at the same index. This measures how damaged
  the stored generated image is.
- **Effective mismatch:** DINO distance between the generated memory image and
  the dataset ground-truth target image. It combines view mismatch and memory
  corruption.
- **Hindsight-best candidate:** the available historical frame with the lowest
  effective mismatch. It is a useful diagnostic reference, not a true oracle.
- **Retention gap:** how much worse the best frame remaining in a bounded bank
  is than the hindsight-best frame in the complete history.
- **Retrieval gap:** how much worse the actually selected frame is than the
  hindsight-best frame still present in that policy's bank.

## 1. Retention versus retrieval decomposition

Scope: 180-second runs.

| Policy | Retention gap | Retrieval gap |
| --- | ---: | ---: |
| Unbounded | approximately 0 by construction | 0.2267 |
| FIFO | 0.2039 | unavailable in the saved summary |
| RI | 0.0474 | 0.1582 |
| SLAM | 0.0639 | 0.1338 |

This shows that RI and SLAM retain banks that remain closer to the hindsight
reference than FIFO. SLAM has the lowest measured retrieval gap. It does not
show that the hindsight-best DINO frame is the uniquely correct conditioning
frame.

### Per-trajectory validation of RI versus SLAM

Across 15 trajectories:

- RI had lower retention gap than SLAM on 10 trajectories and higher gap on 5;
  mean RI-minus-SLAM difference was -0.01649, with sign-test p=0.302.
- SLAM had lower retrieval gap than RI on 12 trajectories and higher gap on 3;
  mean SLAM-minus-RI difference was -0.02444, with p=0.0352.
- RI beat FIFO on retention gap on 15/15 trajectories; mean difference
  -0.15648, p=0.000061.
- SLAM beat unbounded on retrieval gap on 13/15 trajectories; mean difference
  -0.09291, p=0.00739.

The practical reading is that RI mainly helps retention, while SLAM more
consistently helps the frozen selector choose among retained candidates.

## 2. Pool growth and worsening selected memories

Scope: 4,200 queries, 1,050 sections, 15 trajectories from the 180-second
unbounded run.

- Spearman correlation between candidate count and retrieval gap: 0.2029.
- Per-trajectory trend was positive for 13 trajectories and negative for 2;
  sign-test p=0.00739.
- Late-minus-early retrieval gap: +0.07234, trajectory-bootstrap 95% CI
  [0.03960, 0.10882].
- Late-minus-early selected effective mismatch: +0.07316, 95% CI approximately
  [0.016, 0.130].
- Late-minus-early hindsight-best mismatch: +0.00082, 95% CI
  [-0.04961, 0.05457].

The selected memory becomes worse while the best available historical option
stays approximately flat. This localizes the deterioration to selection, under
the DINO-based diagnostic. Candidate count and elapsed generation time grow
together, so this analysis cannot claim that pool size alone causes the trend.

### View mismatch versus memory corruption

| Diagnostic | Spearman with candidate count | Late minus early | 95% CI |
| --- | ---: | ---: | --- |
| Selected view mismatch | -0.1053 | -0.03351 | [-0.05106, -0.01707] |
| Selected memory corruption | 0.1907 | +0.08735 | [0.02356, 0.15715] |
| Selected effective mismatch | 0.1661 | +0.07316 | [0.01647, 0.12956] |

The selected index becomes slightly better aligned with the target view, but
the generated image stored at that index becomes much more corrupted. The net
conditioning match therefore gets worse. This is the clearest evidence for a
memory-mediated autoregressive snowball mechanism.

## 3. Retrieved image quality on each policy's own rollout

Scope: 15 videos and 1,050 sections per policy at 180 seconds. Higher PSNR and
SSIM are better.

| Policy | Selected PSNR | Selected SSIM | Late selected PSNR | Late selected SSIM | Late next-chunk PSNR | Late next-chunk SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unbounded | 11.703 | 0.3089 | 11.433 | 0.3146 | 11.308 | 0.3082 |
| FIFO-32 | 10.619 | 0.2617 | 10.135 | 0.2546 | 10.113 | 0.2543 |
| RI-32 | 13.396 | 0.3830 | 12.896 | 0.3723 | 11.323 | 0.3203 |
| SLAM-32 | 16.522 | 0.4759 | 15.119 | 0.4386 | 11.569 | 0.3253 |

Paired trajectory-bootstrap differences versus unbounded:

| Policy | Selected PSNR delta | Selected SSIM delta | Next-chunk PSNR delta | Next-chunk SSIM delta |
| --- | --- | --- | --- | --- |
| FIFO-32 | -1.084 [-1.639, -0.604] | -0.0472 [-0.0673, -0.0276] | -0.886 [-1.363, -0.469] | -0.0380 [-0.0558, -0.0202] |
| RI-32 | +1.693 [0.845, 2.696] | +0.0741 [0.0495, 0.1027] | +0.015 [-0.216, 0.271] | +0.0133 [0.0028, 0.0279] |
| SLAM-32 | +4.819 [3.565, 6.199] | +0.1670 [0.1217, 0.2137] | +0.164 [-0.111, 0.519] | +0.0169 [0.0012, 0.0392] |

RI and SLAM retrieve cleaner generated images than unbounded; SLAM's advantage
is large. Following-chunk SSIM also improves, but this comparison is
observational because each policy has already produced a different history.

## 4. Common-source selection test

This test removes the different-history confound from the selected-memory
measurement. Every policy contributes only selected frame indices; all image
content is read from the same unbounded videos before comparison with dataset
ground truth.

| Selection policy | Selected PSNR | Selected SSIM | Late PSNR | Late SSIM |
| --- | ---: | ---: | ---: | ---: |
| Unbounded | 11.703 | 0.3089 | 11.433 | 0.3146 |
| FIFO-32 | 11.487 | 0.2980 | 11.320 | 0.3070 |
| RI-32 | 13.478 | 0.3761 | 13.024 | 0.3667 |
| SLAM-32 | 16.332 | 0.4601 | 14.846 | 0.4174 |

Paired differences versus unbounded:

- FIFO: PSNR -0.216, 95% CI [-0.299, -0.141]; SSIM -0.0109,
  CI [-0.0150, -0.0072]. It lost on all 15 trajectories.
- RI: PSNR +1.775, CI [1.005, 2.683]; SSIM +0.0672,
  CI [0.0456, 0.0926]. It won PSNR on 14/15 and SSIM on 15/15.
- SLAM: PSNR +4.629, CI [3.521, 5.874]; SSIM +0.1512,
  CI [0.1100, 0.1929]. It won both metrics on 15/15.

This is strong evidence that RI and SLAM's **selection rules** prefer cleaner
indices, rather than their apparent advantage arising only because their own
rollouts had already become cleaner.

## 5. Diagnostic-to-quality correlations

Scope: 24 matched policy/trajectory pairs. This is a small observational
sample; FVD is excluded because it is distribution-level rather than
per-video.

Expected-direction associations included:

- Retention gap versus LPIPS: Spearman rho=+0.507, bootstrap CI [0.372, 0.643].
- Total hindsight gap versus LPIPS: rho=+0.450, CI [0.084, 0.759].
- Selected memory corruption versus LPIPS: rho=+0.411,
  CI [0.199, 0.650].
- Retrieval gap versus temporal-delta MAE: rho=+0.461,
  CI [0.170, 0.760].

The retrieval-gap result was not consistently aligned with quality:
retrieval gap versus LPIPS was rho=-0.399, CI [-0.678, -0.189], the opposite
of the simple hypothesis. Candidate count versus LPIPS was also negative
(rho=-0.718). These mixed directions mean the DINO gap should not be promoted
as a validated surrogate for video quality.

## 6. Mechanisms tested and not supported

### Occlusion poisoning

Scope: 1,380 queries, 15 videos at 60 seconds.

- Mean appearance penalty of geometric argmax: 0.0455 DINO distance.
- 75.4% of queries had some geometry/appearance conflict.
- Small-pool mean penalty: 0.0364; large/capped-pool penalty: 0.0464.

The effect is real but modest and was not large enough to explain the main
degradation alone.

### Within-section context collapse

Scope: 345 sections at 60 seconds.

- Mean largest-frame share: 0.172.
- Mean effective distinct memories: 25.65 out of 76 target selections.
- One frame served over half the targets in 28/345 sections (8.1%).
- Candidate count versus largest-frame share: rho=-0.314; corrected rho=-0.253.

Concentration exists, but decreases as the pool grows, the wrong direction
for explaining unbounded degradation.

### Simple aggregate memory-drift check

- Mean corruption in the first third: 0.5817.
- Mean corruption in the last third: 0.5641.
- Spearman correlation with section index: 0.176.

This coarse aggregate did not show worsening corruption. The later,
query-conditioned analysis in Section 2 is more targeted and found that the
**retrieved subset** becomes more corrupted. The two tests answer different
questions and should not be conflated.

### Zero-overlap fallback

- Baseline and FIFO hit rates were 100% at overlap thresholds 0.01 and 0.1.
- Baseline overlap median was 0.9335; exact zero rate was 0%; only 0.68% were
  below 0.3 and 4.0% below 0.5.

MemCam almost always has geometrically overlapping candidates. WorldMem's
zero-overlap fallback mechanism does not explain MemCam.

### Monte Carlo IoU noise

Scope: 1,380 queries at 60 seconds.

- Re-estimating overlap at 10x samples changed the winner in 39.06% of queries.
- Mean low-precision top-two gap: 0.0182.
- Flip rate was 57.5% for small uncapped pools and 37.3% for large capped
  pools.
- Mean appearance cost of a winner flip: only 0.0029.

The overlap estimator is unstable among near-ties, but instability falls with
pool size and the visual cost is negligible. It does not explain the observed
duration trend.

## 7. Earlier proxy analyses that should not carry the story

An earlier memory-mechanism report treated frames selected by unbounded memory
as “needed” frames. Under that reference, FIFO preserved far more unbounded
choices than RI or SLAM, and RI/SLAM appeared to have higher eviction regret:

| Policy | Exact availability of unbounded selection | Near availability | Exact eviction-regret rate | Near regret rate |
| --- | ---: | ---: | ---: | ---: |
| FIFO-32 | 0.321 | 0.332 | 0.307 | 0.546 |
| RI-32 | 0.031 | 0.096 | 0.382 | 0.649 |
| SLAM-32 | 0.029 | 0.061 | 0.385 | 0.652 |

These results are useful only for describing how differently the policies
behave. Unbounded selection is not a ground-truth useful-frame label, so the
numbers cannot establish that RI or SLAM evicted useful memories.

The same report did establish the policies' temporal profiles:

| Policy | Median retained age | 90th-percentile age | Fraction older than 304 frames |
| --- | ---: | ---: | ---: |
| Unbounded | 1591.0 | 3716.0 | 0.891 |
| FIFO-32 | 15.5 | 28.0 | 0.000 |
| RI-32 | 1377.5 | 3999.0 | 0.800 |
| SLAM-32 | 2089.0 | 4516.0 | 0.883 |

## 8. Attention intervention pilot

Scope: 10 videos, 270 interventions, 90 section-step probe groups.

| Predictor | Global Spearman | Mean within-probe Spearman |
| --- | ---: | ---: |
| Total attention | 0.6787 | 0.9056 |
| Attention per slot | -0.0321 | 0.2222 |
| Slot count | 0.7275 | 0.9072 |
| Retrieval overlap | -0.1319 | 0.4167 |
| Memory age | -0.3794 | -0.4833 |

High-attention memories caused the larger immediate ablation effect in 98.89%
of paired probes, with mean high-minus-low relative effect 0.0248. However,
slot count predicted the effect at least as strongly as total attention, while
attention per slot was weak. The pilot shows denoiser sensitivity among
already retrieved memories; it does not validate attention as future utility
or as an eviction score.

## 9. Causal replay status

### Context-identity swap

One valid late-section replay changed only the selected contexts from
unbounded choices to SLAM choices while preserving the preceding history:

- Row 13, section 49; 73 target selections changed.
- Control LPIPS 0.84149; swap LPIPS 0.83972; delta -0.00177.
- Control DINO distance 0.84623; swap 0.82932; delta -0.01691.

The changed contexts slightly improved that section, but n=1 is far too small
for a general causal claim.

### Ground-truth memory cleaning

Four high-corruption late sections were selected. Their mean selected-memory
corruption values were 0.9839, 0.8981, 0.8609, and 0.8249. At the last recorded
status, only one clean-GT branch had completed and the complete control/clean
pairs were unavailable. No causal result can yet be reported.

This is the decisive remaining test: keep the selected frame identities and
all preceding history fixed, replace only the selected generated memory images
with their dataset-ground-truth counterparts for one section, and measure the
change in the generated output.

## 10. Policy probes informed by the diagnostics

These are not mechanism proofs, but they constrain policy design.

- Density-balanced view coverage beat SLAM only at 10 seconds. At 60 seconds,
  LPIPS was 0.59413 versus 0.58495 and FVD was 750.30 versus 690.53, both worse.
- RI with three rarity neighbors improved FVD over k=1 at 10 seconds
  (706.71 versus 713.02) and 20 seconds (684.59 versus 721.48), but was worse
  at 60 seconds (705.91 versus 660.12).
- Surprise forcing versus matched unbounded was mixed: at 30 seconds it had
  better FVD (755.46 versus 880.58) but worse LPIPS (0.53083 versus 0.51563).
- A 50/50 SLAM-RI score blend did not beat both constituent policies. This
  argues against relying on an uncalibrated weighted sum as the final method.

## 11. Failed generic image-quality gate

We tested whether a score computed from the generated RGB frame alone could
identify the bottom 20% of frames by within-trajectory PSNR/SSIM rank. The
calibration used 2,160 frames from unbounded and SLAM runs, with ten complete
trajectories for threshold fitting and five held-out trajectories.

The strongest estimator was `unclipped_fraction`, but it was not deployable:

- held-out AUC: 0.630;
- balanced accuracy: 0.550;
- bad-frame recall at the conservative threshold: 0.183;
- clean-frame false-rejection rate: 0.125;
- mean within-trajectory Spearman correlation: 0.339, CI [0.138, 0.554].

At 20% bad-frame prevalence, these rates imply that only about 26.8% of
rejected frames are bad: for every 1,000 frames, the gate catches roughly 37
bad frames while incorrectly removing 100 clean frames. Generic learned IQA
(MUSIQ, CLIP-IQA+, and TOPIQ-NR) was near chance. Sharpness, contrast,
gradient energy, and Laplacian variance were often inversely correlated with
exact-index fidelity.

**Decision:** generic no-reference IQA must not be injected into the memory
policy. The diagnosed error is reference-dependent scene fidelity, not simply
blur, clipping, or aesthetic quality.

## 12. Rejected circular consistency shortcut

Comparing a new frame only with an arbitrary previous generated frame is not
an absolute quality test. If the previous frame is already corrupted, high
agreement can certify consistent propagation of the same error. The earlier
cross-view `reliable_slam_ri` prototype therefore does not constitute a
validated quality gate and must not be reported as the final method.

The next hypothesis is narrower and remains unproven: compare a new frame with
the frame that actually conditioned its generation, then subtract the expected
DINO similarity for that camera displacement. The expected similarity is fit
from ground-truth frame pairs on training trajectories only. Validation must
report held-out AUC, bad-frame precision and recall, clean-frame rejection,
performance when the conditioning frame is itself corrupted, and a control
anchored to the clean input frame. The score will be integrated into generation
only if it passes pre-declared deployment criteria.

## Current scientific conclusion

The defensible claim today is:

> In MemCam, retaining every generated frame does not guarantee better
> conditioning. As the rollout grows, the native selector increasingly exposes
> the generator to corrupted historical images even though similarly aligned,
> cleaner historical candidates remain available. Bounded RI and SLAM curation,
> especially SLAM, strongly biases retrieval toward cleaner frame indices.

The unsupported extension is:

> The growing archive itself causally poisons retrieval, and corrupted selected
> memories causally produce the later quality loss.

The first part is confounded with elapsed autoregressive time. The second part
requires completed matched GT-cleaning replays. Until those replays finish, the
result is strong observational mechanism evidence rather than a closed causal
explanation.
