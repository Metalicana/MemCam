"""Offline MCE hyperparameter sweep -- no video generation required.

Extracts DINO/RGB features once from an already-generated video (e.g. an
existing baseline run), then replays the exact per-section retention loop
``wan_video_memcam.py`` uses for ``memory_policy="mce"`` under many
``(alpha, gamma, lambda)`` combinations, entirely in-process. Camera poses
come straight from the manifest's pose JSON (deterministic, no generation
needed either).

Caveat, read this before trusting the numbers: this replays the *write*
path (what ``compute_marginal_coverage_eviction_scores`` decides to keep)
but not the *read* path (which retained candidate the native retriever
actually selects per query). The real access-trace reuse/entropy stats
(reuse_gini, section_entropy_norm, ...) come from the read path and need a
real generation run to observe. What this script measures directly and
cheaply is the "anchor vs. churn" axis on the write side alone: retained-set
age distribution and per-frame survival duration (how many sections a frame
stays alive before eviction) -- the same axis that explained SLAM/RI's edge
over FIFO/baseline earlier this project. Use it to narrow down candidate
hyperparameters fast, then confirm the top 1-2 with a real pilot run.

Usage:
    python utils/mce_offline_selection_sweep.py \
        --video ~/memcam_results/context_memory_60s/baseline/seed0_AncientTempleEnv_5_1368_60s_custom.mp4 \
        --pose_json /path/to/AncientTempleEnv.json \
        --start_frame 1368 \
        --num_frames 1825 \
        --budget 32 \
        --alpha 0.5,0.65,0.8 \
        --gamma 0.1,0.25,0.5
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MEMORY_POLICIES_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"
_spec = importlib.util.spec_from_file_location("memory_policies", MEMORY_POLICIES_PATH)
memory_policies = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(memory_policies)

FrameMemoryBuffer = memory_policies.FrameMemoryBuffer
VisualMemoryFeatureExtractor = memory_policies.VisualMemoryFeatureExtractor
compute_marginal_coverage_eviction_scores = (
    memory_policies.compute_marginal_coverage_eviction_scores
)
_historical_query_medoids = memory_policies._historical_query_medoids

FRAMES_PER_SECTION = 77
PREDICT_FRAMES = 76


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_frames_as_pil(video_path, num_frames=None):
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    frames = []
    for index, frame in enumerate(reader):
        if num_frames is not None and index >= num_frames:
            break
        frames.append(Image.fromarray(frame))
    reader.close()
    return frames


def section_boundaries(total_frames):
    total_sections = (total_frames - 1) // PREDICT_FRAMES
    sections = []
    for section_idx in range(total_sections):
        if section_idx == 0:
            new_frame_indices = list(range(0, FRAMES_PER_SECTION))
        else:
            start = section_idx * PREDICT_FRAMES + 1
            new_frame_indices = list(range(start, start + PREDICT_FRAMES))
        sections.append(new_frame_indices)
    return sections


def cluster_diagnostic(dino_features, candidate_frame_indices, rarity_neighbors_values):
    """Directly measure clustering granularity at a realistic candidate-pool
    size, before running any write-path simulation. Answers "is the
    clustering over-segmenting (near-1 candidate per cluster, so MCE's
    redundancy discount barely engages) independent of alpha/hist_freq_bias,
    which only matter once real cluster-size variance exists."""
    print(
        f"\nCluster diagnostic on a {len(candidate_frame_indices)}-candidate pool "
        f"(realistic eviction-time size):"
    )
    print(f"{'rarity_neighbors':>16} {'num_clusters':>13} {'mean_size':>10} {'size=1_frac':>12} {'max_size':>9}")
    for rarity_neighbors in rarity_neighbors_values:
        _, clusters = _historical_query_medoids(
            candidate_frame_indices, dino_features, int(rarity_neighbors)
        )
        sizes = np.array([len(members) for members in clusters], dtype=np.float64)
        singleton_frac = float(np.mean(sizes == 1))
        print(
            f"{int(rarity_neighbors):16d} {len(sizes):13d} {sizes.mean():10.2f} "
            f"{singleton_frac:12.3f} {int(sizes.max()):9d}"
        )


def simulate_mce(dino_features, rgb_features, c2ws, total_frames, budget, alpha, gamma, lambda_hist, query_stride, hist_freq_bias=0.0, rarity_neighbors=3):
    memory_buffer = FrameMemoryBuffer(policy="mce", budget=budget, pinned_frames={0})
    sections = section_boundaries(total_frames)

    admitted = set()
    alive_since = {}
    survival_records = []  # (frame_idx, sections_alive) once evicted
    final_ages = None

    for section_idx, new_frame_indices in enumerate(sections):
        section_end_frame = new_frame_indices[-1]
        current_memory = list(memory_buffer.candidates())
        prospective_memory = current_memory + [
            f for f in new_frame_indices if f not in current_memory
        ]
        for f in new_frame_indices:
            admitted.add(f)
            alive_since.setdefault(f, section_idx)

        future_query_frame_indices = range(
            section_end_frame + 1, total_frames, max(1, int(query_stride))
        )
        protected_frames = {section_end_frame}
        eviction_scores, _ = compute_marginal_coverage_eviction_scores(
            memory_frame_indices=prospective_memory,
            c2ws=c2ws,
            budget=budget,
            future_query_frame_indices=future_query_frame_indices,
            forced_keep_frames=protected_frames | {0},
            dino_features=dino_features,
            rgb_features=rgb_features,
            alpha=alpha,
            lambda_hist=lambda_hist,
            gamma=gamma,
            hist_freq_bias=hist_freq_bias,
            rarity_neighbors=rarity_neighbors,
            return_details=True,
        )
        before = set(prospective_memory)
        evicted = memory_buffer.update(
            new_frame_indices, eviction_scores=eviction_scores, protected_frames=protected_frames
        )
        for frame_idx in evicted:
            survival_records.append((frame_idx, section_idx - alive_since[frame_idx] + 1))

        if section_idx == len(sections) - 1:
            final_ages = [section_end_frame - f for f in memory_buffer.candidates()]

    # Anything still alive at the end "survived" the whole remaining horizon.
    last_section_idx = len(sections) - 1
    for frame_idx in memory_buffer.candidates():
        survival_records.append((frame_idx, last_section_idx - alive_since[frame_idx] + 1))

    survival = np.array([s for _, s in survival_records], dtype=np.float64)
    return {
        "retained_frames": sorted(memory_buffer.candidates()),
        "final_age_median": float(np.median(final_ages)) if final_ages else None,
        "final_age_p90": float(np.percentile(final_ages, 90)) if final_ages else None,
        "survival_sections_mean": float(survival.mean()) if survival.size else None,
        "survival_sections_p90": float(np.percentile(survival, 90)) if survival.size else None,
        "survival_gini": gini(survival) if survival.size else None,
        "unique_frames_ever_admitted": len(admitted),
        "num_sections": len(sections),
    }


def gini(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = values.size
    if n == 0 or values.sum() <= 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose_json", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--alpha", type=str, default="0.65")
    parser.add_argument("--gamma", type=str, default="0.25")
    parser.add_argument("--lambda_hist", type=str, default="")
    parser.add_argument(
        "--hist_freq_bias", type=str, default="0.0",
        help="Q_hist medoid weight exponent on cluster size: 0=paper default "
             "(equal weight per scene mode), 1=linear frequency-proportional "
             "(closer to what anchor-persistence heuristics like SLAM reward).",
    )
    parser.add_argument(
        "--rarity_neighbors", type=str, default="3",
        help="k for the clustering threshold estimate: higher values should "
             "coarsen clusters (fewer, bigger) if the default is over-"
             "segmenting near-duplicate candidates into singleton clusters.",
    )
    parser.add_argument("--query_stride", type=int, default=19)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from dataset.poses import load_c2ws_from_json

    print(f"Loading frames from {args.video}")
    frames = load_frames_as_pil(args.video, num_frames=args.num_frames)
    print(f"Loaded {len(frames)} frames")

    print("Extracting DINO/RGB features (one-time cost for this video)...")
    extractor = VisualMemoryFeatureExtractor(device=args.device)
    dino_batch, rgb_batch = extractor.encode_pil_images(frames)
    dino_features = {i: dino_batch[i] for i in range(len(frames))}
    rgb_features = {i: rgb_batch[i] for i in range(len(frames))}

    c2ws = load_c2ws_from_json(
        json_path=args.pose_json, start_frame=args.start_frame, num_frames=len(frames)
    )

    alphas = parse_float_list(args.alpha)
    gammas = parse_float_list(args.gamma)
    lambdas = parse_float_list(args.lambda_hist) if args.lambda_hist else [None]
    freq_biases = parse_float_list(args.hist_freq_bias)
    rarity_neighbors_values = [int(v) for v in parse_float_list(args.rarity_neighbors)]

    # Diagnostic first, cheap and direct: a realistic eviction-time candidate
    # pool (budget existing items + one section's worth of new frames), taken
    # from the middle of the video to avoid start-of-clip edge effects.
    pool_start = max(0, len(frames) // 2 - (args.budget + FRAMES_PER_SECTION) // 2)
    pool_end = min(len(frames), pool_start + args.budget + FRAMES_PER_SECTION)
    diagnostic_pool = list(range(pool_start, pool_end))
    cluster_diagnostic(dino_features, diagnostic_pool, rarity_neighbors_values)

    print(
        f"\n{'alpha':>6} {'gamma':>6} {'lambda':>7} {'freq_b':>7} {'rarity_k':>8} "
        f"{'age_med':>8} {'age_p90':>8} {'surv_mean':>10} {'surv_p90':>9} {'surv_gini':>10} {'unique':>7}"
    )
    for alpha in alphas:
        for gamma in gammas:
            for lambda_hist in lambdas:
                for hist_freq_bias in freq_biases:
                    for rarity_neighbors in rarity_neighbors_values:
                        result = simulate_mce(
                            dino_features,
                            rgb_features,
                            c2ws,
                            total_frames=len(frames),
                            budget=args.budget,
                            alpha=alpha,
                            gamma=gamma,
                            lambda_hist=lambda_hist,
                            query_stride=args.query_stride,
                            hist_freq_bias=hist_freq_bias,
                            rarity_neighbors=rarity_neighbors,
                        )
                        lambda_display = "auto" if lambda_hist is None else f"{lambda_hist:.2f}"
                        print(
                            f"{alpha:6.2f} {gamma:6.2f} {lambda_display:>7} {hist_freq_bias:7.2f} {rarity_neighbors:8d} "
                            f"{result['final_age_median']:8.1f} {result['final_age_p90']:8.1f} "
                            f"{result['survival_sections_mean']:10.2f} {result['survival_sections_p90']:9.1f} "
                            f"{result['survival_gini']:10.3f} {result['unique_frames_ever_admitted']:7d}"
                        )

    print(
        "\nTarget shape from real SLAM/RI b32 runs (read-path reuse stats, not "
        "directly comparable units, but same axis): SLAM/RI selected-age median "
        "~250-500, p90 ~1100-1600; FIFO/baseline sit far below that. Configs "
        "whose final_age_p90 stays low and survival_gini stays low are behaving "
        "like FIFO (churn); configs that push both up are heading toward SLAM/RI's "
        "anchor-persistence shape."
    )


if __name__ == "__main__":
    main()
