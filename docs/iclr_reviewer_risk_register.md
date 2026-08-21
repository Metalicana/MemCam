# ICLR Reviewer Risk Register

Last updated: 2026-08-21

## Purpose

This document records the strongest reviewer attacks against the project and
turns them into evidence requirements. It is not paper prose. A criticism is
not accepted merely because it sounds severe, and a proposed mechanism is not
promoted into the paper until the architecture and experiments support it.

The governing rule is simple: every causal claim needs an identified path
through the implemented model and a matched intervention. Every final policy
component must beat its simpler ablation or be removed.

## Current decision

- The diagnosis is stronger than the final method.
- Generic no-reference image-quality gating has failed and is closed.
- The fixed `0.75 Geometric Coverage + 0.25 RI` blend is an unvalidated
  hypothesis, not the final method.
- Pose-calibrated conditioning consistency is being evaluated as a possible
  admission gate. It enters the policy only if the predeclared held-out test
  returns `INJECT`.
- Direct archive-induced softmax dilution is not a valid MemCam mechanism:
  archive growth changes the candidate search pool, while the retriever still
  supplies a fixed-size conditioning set to the generator.
- Cross-representation claims remain pending. Results on MemCam alone support
  a MemCam claim, not a universal memory-management claim.

## Claim ledger

| Claim | Status | Evidence or blocker |
| --- | --- | --- |
| Unbounded storage and exhaustive retrieval grow with duration | Supported | Archive size grows linearly; cumulative exhaustive search grows superlinearly under the current implementation. |
| Bounded curation can outperform unbounded MemCam | Supported on the current benchmark | Completed LPIPS/FVD comparisons show RI and Geometric Coverage can beat unbounded. |
| Unbounded selects increasingly corrupted historical images | Supported observationally | Pool-growth split and common-source selection analysis. |
| Candidate-pool growth alone causes the degradation | Not established | Pool size and elapsed autoregressive time co-vary. |
| Corrupted selected memories cause the next chunk to degrade | Pending causal test | Requires complete matched GT-content cleaning replays. |
| Archive growth directly dilutes MemCam denoiser attention | Rejected as stated | The archive is reduced to a fixed-size retrieved context before denoising. |
| Generic IQA can gate corrupted memories | Rejected | Held-out calibration was not deployable. |
| Pose-calibrated conditioning consistency can gate corrupted memories | Pending | Validation code exists; results are not yet available. |
| A fixed 75/25 Geometric Coverage-RI blend is optimal or transferable | Not established | The 50/50 blend lost; 75/25 and sensitivity evidence are incomplete. |
| The method is representation agnostic | Not established | Equivalent, successful tests on another memory representation are required. |

## R1: Venue and contribution risk

**Reviewer attack:** the work may look like a system-specific cache policy
rather than a learning or representation contribution.

**Assessment:** valid positioning risk, but venue fit cannot be repaired by
inventing a neural mechanism that MemCam does not contain. The defensible core
is a mechanism study of online generative memory plus a policy that follows
from that diagnosis.

**Required evidence:**

1. A clean decomposition of selection failure, stored-content corruption, and
   downstream generation damage.
2. A matched intervention showing that changing memory selection or memory
   content changes the next generated section.
3. A final bounded policy that consistently beats Geometric Coverage, or an
   honest decision to make Geometric Coverage the strongest method studied.
4. Quality, camera consistency, efficiency, and memory scaling results under
   one locked protocol.

**Kill criterion:** if no proposed addition beats Geometric Coverage, do not
manufacture a complicated final method. Reframe the work around the diagnostic
finding and the strongest reproducible curation result, then reassess venue.

## R2: Softmax dilution and attention entropy

**Reviewer demand:** make softmax dilution from the growing archive the central
mechanism and prove it with attention entropy.

**Assessment for MemCam:** rejected as a direct explanation. The archive may
contain thousands of candidates, but MemCam performs retrieval before
generation and passes a fixed number of selected context items into the
denoiser. The archive size therefore does not directly enlarge the denoiser's
attention softmax denominator. Archive growth can still hurt through candidate
competition, noisy selection, or exposure to corrupted memories.

**Permissible attention experiment:** measure whether the *selected fixed-size
context* receives diffuse or ineffective attention when its contents are poor.
That would be an outcome of bad selection, not proof of archive-token dilution.

**WorldMem status:** unknown until its exact integration path is audited. An
attention-dilution claim is allowed only if the number of memory tokens seen by
the attention layer actually grows with duration.

## R3: The 75/25 weighting

**Reviewer attack:** a fixed coefficient can look tuned to one dataset and one
backbone.

**Assessment:** valid. The current evidence does not justify calling 75/25
principled or final. A 50/50 blend already failed to beat both components.

**Required evidence:**

1. Select the coefficient using calibration trajectories only.
2. Report a sensitivity curve including pure RI, interior mixtures, and pure
   Geometric Coverage.
3. Lock the coefficient before final benchmark evaluation.
4. If claiming a universal fixed coefficient, transfer it unchanged to the
   second system. If per-system calibration is part of the method, state that
   explicitly instead of claiming zero-shot transfer.

**Kill criterion:** if gains exist only at one narrow coefficient, or the blend
does not beat Geometric Coverage, remove the blend from the final method.

## R4: Diagnosis-to-solution gap

**Reviewer attack:** the diagnosis may be strong while the proposed remedy is
only a weighted sum of existing scores.

**Assessment:** valid. Complexity is not the cure. A gate is valuable only if
it detects the diagnosed failure online and improves rollout quality.

The current candidate is pose-calibrated conditioning consistency: compare a
new generated frame with the context that actually conditioned it, then remove
the expected similarity change caused by camera displacement. This is a
hypothesis, not yet a method result, and it is not a full forward cycle.

**Admission rule:** inject the gate only if the held-out validator satisfies
all predeclared criteria, including precision, recall, clean-frame rejection,
pose-calibration gain, and performance when the conditioning frame is itself
corrupted. Otherwise the gate is rejected.

## R5: Cross-system and representation claims

**Reviewer attack:** MemCam hard retrieval and WorldMem soft memory integration
may be too different for one diagnostic and policy to transfer unchanged.

**Assessment:** valid. "Representation agnostic" describes an interface, not
an empirical result.

**Required evidence:**

1. Document what constitutes one item, its cost, the retrieval operation, and
   the generator-facing context for each system.
2. Define equivalent retention and retrieval diagnostics without forcing
   MemCam's top-1 semantics onto soft-attention systems.
3. Run at least one matched intervention and one final-policy comparison on the
   second system.

**Decision:** if WorldMem supports only a superficial comparison, narrow the
claim rather than using it as weak evidence of universality.

## R6: Oracle terminology

"Zero-gen oracle" is not sufficiently defined and must not appear in the
paper without an exact intervention. The project currently has three distinct
references:

- **Hindsight-best historical candidate:** chooses the available history item
  with minimum diagnostic mismatch. It is an offline proxy, not true utility.
- **Common-source selection control:** applies each policy's selected indices
  to images from one shared rollout. It isolates index selection quality.
- **GT-content cleaning replay:** preserves selected indices and history, but
  replaces selected generated memory content with dataset ground truth for one
  section. This is the causal test of memory-content corruption.

None of these is a deployable oracle. Their names and conclusions must remain
separate.

## Closed failures

### Generic no-reference IQA

`unclipped_fraction` was statistically above chance but operationally poor. At
the tested threshold it caught too few corrupted frames while rejecting too
many clean frames. MUSIQ, CLIP-IQA+, TOPIQ-NR, and handcrafted sharpness or
contrast scores were near chance or pointed in the wrong direction. This path
is closed unless a materially different estimator and protocol are proposed.

### Arbitrary predecessor consistency

Agreement with an arbitrary previous generated frame can reward propagation of
an already corrupted state. It is circular and is not an accepted quality
gate.

### Acceptance guarantees

No experiment can guarantee ICLR acceptance. The goal of this register is to
remove unsupported claims and expose decisive tests, not to promise an outcome.

## Ordered decision queue

1. Complete and evaluate the matched GT-content cleaning replay.
2. Run the pose-calibrated conditioning-consistency validator and obey its
   `INJECT` or `DO_NOT_INJECT` result.
3. Finish the 75/25 blend evaluation and compare it directly with pure
   Geometric Coverage and pure RI.
4. Run a coefficient sensitivity sweep only if the blend remains competitive.
5. Audit WorldMem's memory-to-attention path before designing an attention
   entropy experiment.
6. Scope the final claim to the systems on which the locked method actually
   works.

