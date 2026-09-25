"""Inspect saved native-style round trips without generation or feature extraction."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.audit_gap_inputs import audit_trace
from paper.make_180s_gt_comparisons import bundle_path, decode_cached, digest, save_json
from utils.analyze_retrieval_quality_decomposition import reconstruct_candidate_banks
from utils.camera_rotation_utils import exact_roundtrip_c2ws, roundtrip_pairs
from utils.evaluate_context_memory import FVDRunner


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rotation_errors(poses, target, candidates):
    # trace(R_target.T @ R_candidate) is the Frobenius inner product.
    trace = np.einsum("ij,nij->n", poses[target, :3, :3], poses[candidates, :3, :3])
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def update_rows(events, frames):
    bank, result = {0}, []
    for s in range((frames - 1) // 76):
        new = set(range(s * 76 + 1, (s + 1) * 76 + 1))
        candidates = bank | new | {s * 76}
        evictions = [r for r in events if r.get("event") == "memory_eviction" and r["section_idx"] == s]
        removed = [int(r["evicted_memory_frame"]) for r in evictions]
        if len(removed) != len(set(removed)) or not set(removed) <= candidates:
            raise ValueError("Duplicate or unavailable eviction ID")
        bank = candidates - set(removed)
        if len(bank) != 32 or not {0, (s + 1) * 76} <= bank:
            raise ValueError("Invalid B32 post-update bank")
        result.append(dict(section=s, new_frames=len(new), new_frames_kept=len(new & bank),
                           kept_new_ids=sorted(new & bank), evicted=len(removed),
                           nearest_also_evicted=sum(r.get("eviction_nearest_covisible_frame") in removed for r in evictions)))
    return result


def retrieval_rows(events, poses, angle):
    count = len(poses)
    expected = exact_roundtrip_c2ws(poses[0], angle)
    if poses.shape != expected.shape or not np.allclose(poses, expected, atol=1e-9, rtol=0):
        raise ValueError("Saved poses do not follow the exact round-trip trajectory")
    if not np.array_equal(poses, poses[::-1]):
        raise ValueError("Outward/return poses are not identical")
    sections = (count - 1) // 76
    banks = reconstruct_candidate_banks(events, sections - 1, num_frames=count)
    rows, seen = [], set()
    for event in events:
        if event.get("event") != "context_access" or not event.get("selected"):
            continue
        s, q, selected = (int(event[k]) for k in ("section_idx", "target_frame", "selected_memory_frame"))
        if s not in banks or (s, q) in seen or not s * 76 < q <= (s + 1) * 76:
            raise ValueError("Duplicate or invalid query")
        seen.add((s, q))
        bank = banks[s]
        history = list(range(s * 76 - 3))
        if selected not in bank or len(bank) != event["candidate_count"] or not set(bank) <= set(history):
            raise ValueError("Invalid selected ID, bank count, or eligible history")
        distances = rotation_errors(poses, q, bank)
        best_col = int(np.argmin(distances))
        full_best = float(np.min(rotation_errors(poses, q, history)))
        selected_error = float(rotation_errors(poses, q, [selected])[0])
        bank_best = float(distances[best_col])
        if selected_error + 1e-5 < bank_best or bank_best + 1e-5 < full_best:
            raise ValueError("Invalid nested-oracle ordering")
        rows.append(dict(section=s, target_frame=q, selected_frame=selected,
                         phase="return" if q > count // 2 else "outward",
                         mirror_frame=count - 1 - q, mirror_retained=(count - 1 - q in bank),
                         candidate_count=len(bank), outward_candidates=sum(i <= count // 2 for i in bank),
                         selected_from_outward=selected <= count // 2,
                         selected_rotation_deg=selected_error, bank_best_rotation_deg=bank_best,
                         full_best_rotation_deg=full_best, bank_best_frame=bank[best_col],
                         retention_rotation_gap_deg=bank_best - full_best,
                         selection_rotation_gap_deg=selected_error - bank_best))
    expected_queries = {(s, q) for s in range(1, sections) for q in range(s * 76 + 1, (s + 1) * 76 + 1)}
    if seen != expected_queries:
        raise ValueError("Incomplete query coverage")
    return sorted(rows, key=lambda r: r["target_frame"]), banks


def load_case(folder):
    generation = json.loads((folder / "generation.json").read_text())
    measurement = json.loads((folder / "metrics.json").read_text())
    case = generation["case"]
    if generation["status"] != "complete" or case["id"] != folder.name:
        raise ValueError(f"Incomplete or mismatched generation: {folder}")
    signature = hashlib.sha256(json.dumps(generation, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if measurement["key"]["generation_sha256"] != signature:
        raise ValueError("Metrics refer to a different generation receipt")
    attempt = (folder / generation["attempt"]).resolve()
    if not attempt.is_relative_to(folder.resolve()):
        raise ValueError("Attempt outside case folder")
    verified = {}
    # The download intentionally excludes lossless PNGs. Never treat MP4 as metric input.
    for name in ("video.mp4", "poses.npy", "access.jsonl", "profile.jsonl"):
        path = bundle_path(attempt, name)
        if digest(path) != generation["files"][name]:
            raise ValueError(f"Changed generation artifact: {path}")
        verified[str(path)] = digest(path)
    for name in ("pairs.csv", "roundtrip_features.npz"):
        path = bundle_path(folder, name)
        if digest(path) != measurement["files"][name]:
            raise ValueError(f"Changed metric artifact: {path}")
        verified[str(path)] = digest(path)
    poses = np.load(attempt / "poses.npy", allow_pickle=False)
    if len(poses) != case["frames"]:
        raise ValueError("Frame/pose count mismatch")
    item = dict(scene=case["start"]["scene"], start_frame=case["start"]["start_frame"],
                duration_sec=(len(poses) - 1) / 30, num_frames=len(poses))
    audit_trace(attempt / "access.jsonl", item, "slam_covisibility", 32)
    events = [json.loads(line) for line in (attempt / "access.jsonl").read_text().splitlines() if line.strip()]
    queries, banks = retrieval_rows(events, poses, case["angle"])
    with (folder / "pairs.csv").open() as handle:
        pairs = list(csv.DictReader(handle))
    expected_pairs = roundtrip_pairs(len(poses))
    if [(int(r["outward_frame"]), int(r["return_frame"])) for r in pairs] != expected_pairs:
        raise ValueError("Pair indices do not match the saved evaluation contract")
    for row in pairs:
        for key in ("psnr_db", "ssim", "lpips"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]):
                raise ValueError("Nonfinite pair score")
    pair_by_target = {int(r["return_frame"]): r for r in pairs}
    for q in queries:
        q.update({k: pair_by_target.get(q["target_frame"], {}).get(k, "")
                  for k in ("psnr_db", "ssim", "lpips")})
    with np.load(folder / "roundtrip_features.npz", allow_pickle=False) as features:
        a, b = features["outward"], features["returning"]
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != 4 or not np.isfinite([a, b]).all():
        raise ValueError("Invalid four-clip feature cache")
    protocol = measurement["key"]["metric_identity"]
    if len(measurement["fvd_clip_pairs"]) != 4:
        raise ValueError("Invalid clip count")
    for clip in measurement["fvd_clip_pairs"]:
        if len(clip) != protocol["fvd_config"]["clip_length"] or any(tuple(p) not in expected_pairs for p in clip):
            raise ValueError("Clip contains unpaired frames")
        if any(clip[i + 1][0] - clip[i][0] != protocol["fvd_config"]["frame_stride"] for i in range(len(clip) - 1)):
            raise ValueError("Clip stride mismatch")
    return dict(case=case, attempt=attempt, queries=queries, banks=banks, pairs=pairs,
                outward=a, returning=b, verified=verified, protocol=protocol,
                updates=update_rows(events, len(poses)))


def frechet(a, b):
    # Only the existing NumPy scoring method is used, without loading Torch/I3D.
    return FVDRunner._frechet_distance(object.__new__(FVDRunner), a, b)


def section_rows(record):
    result = []
    for s in sorted(record["banks"]):
        queries = [q for q in record["queries"] if q["section"] == s]
        if not queries:
            continue
        values = dict(case_id=record["case"]["id"], angle=record["case"]["angle"], section=s,
                      phase=queries[0]["phase"], queries=len(queries), candidate_count=queries[0]["candidate_count"])
        for name in ("selected_rotation_deg", "bank_best_rotation_deg", "full_best_rotation_deg",
                     "retention_rotation_gap_deg", "selection_rotation_gap_deg", "selected_from_outward"):
            values[name] = float(np.mean([q[name] for q in queries]))
        values["worst_bank_best_rotation_deg"] = max(q["bank_best_rotation_deg"] for q in queries)
        for name in ("psnr_db", "ssim", "lpips"):
            measured = [q[name] for q in queries if q[name] != ""]
            values[name] = float(np.mean(measured)) if measured else ""
        result.append(values)
    return result


def render_examples(record, output):
    from PIL import Image, ImageDraw
    from paper.compare_local_rollouts import font, probe

    case = record["case"]
    video = record["attempt"] / "video.mp4"
    if probe(video) != dict(width=640, height=352, frames=case["frames"], fps=30.):
        raise ValueError(f"Video geometry/count mismatch: {video}")
    half = case["frames"] // 2
    first = [round(half * f) for f in (.125, .375, .625, .875)]
    target = [case["frames"] - 1 - i for i in first]
    queries = {q["target_frame"]: q for q in record["queries"]}
    selected = [queries[q]["selected_frame"] for q in target]
    indices = sorted(set(first + target + selected))
    paths, _ = decode_cached(video, indices, output / "frames" / case["id"])
    left, top, tw, th, gap, label = 175, 65, 384, 211, 10, 46
    canvas = Image.new("RGB", (left + 4 * (tw + gap), top + 3 * (th + label + gap) + 35), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{case['id']} | fixed angular samples, original MP4 pixels", font=font(23, True), fill="black")
    for row, (name, frames) in enumerate((("Outward", first), ("Return", target), ("Selected ID\n(video proxy)", selected))):
        y = top + row * (th + label + gap)
        draw.multiline_text((8, y + 70), name, font=font(19, True), fill="#252525")
        for col, index in enumerate(frames):
            x = left + col * (tw + gap)
            with Image.open(paths[index]) as image:
                canvas.paste(image.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS), (x, y))
            text = f"Frame {index}"
            if row == 1:
                text += f" | LPIPS {queries[index]['lpips']:.3f}"
            if row == 2:
                text += f" | pose mismatch {queries[target[col]]['selected_rotation_deg']:.1f} deg"
            draw.text((x, y + th + 5), text, font=font(16), fill="#252525")
    draw.text((12, canvas.height - 28), "Outward and return rows have identical requested poses. Neither is ground truth.", font=font(17), fill="#454545")
    path = output / (case["id"] + ".png")
    canvas.save(path)
    return path


def render_coverage(records, output):
    import matplotlib.pyplot as plt

    group = [r for r in records if r["case"]["angle"] == 360]
    fig, axes = plt.subplots(len(group), 1, figsize=(10, 9), sharex=True, sharey=True)
    for ax, record in zip(axes, group):
        queries = [q for q in record["queries"] if q["phase"] == "return"]
        x = [q["target_frame"] / 30 for q in queries]
        ax.plot(x, [q["selected_rotation_deg"] for q in queries], color="#3A718C", lw=1.4, label="Selected memory")
        ax.plot(x, [q["bank_best_rotation_deg"] for q in queries], color="#C36C43", lw=1.2,
                linestyle="--", label="Best retained pose")
        ax.plot(x, [q["full_best_rotation_deg"] for q in queries], color="#444444", lw=.7,
                label="Best eligible-history pose")
        ax.set_title(record["case"]["start"]["scene"], loc="left", fontsize=11, pad=4)
        ax.set_ylim(-3, 95)
        ax.set_yticks([0, 45, 90])
        ax.grid(axis="y", color="#E7E7E7", lw=.5)
        ax.spines[["top", "right"]].set_visible(False)
        for s in range(5, 8):
            ax.axvline((s * 76 + 1) / 30, color="#DDDDDD", linewidth=.6)
    axes[-1].set_xlabel("Generated time (seconds), return leg")
    fig.supylabel("Requested camera rotation mismatch (degrees)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.text(.5, .015, "Eligibility excludes continuation frames. Angles compare requested poses, not estimated generated cameras.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(.025, .03, 1, .96), h_pad=.8)
    fig.savefig(output / "return_pose_coverage.png", dpi=180)
    fig.savefig(output / "return_pose_coverage.pdf")
    plt.close(fig)


def run(source, output, render=True):
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Audit output must be separate from original results")
    plan = json.loads((source / "plan.json").read_text())
    protocol = json.loads((source / "metric_protocol.json").read_text())
    expected = {f"{s['scene']}_{s['start_frame']:04d}_{angle}deg" for s in plan["starts"] for angle in (90, 360)}
    folders = sorted((source / "cases").iterdir())
    if len(expected) != 10 or {p.name for p in folders if p.is_dir()} != expected:
        raise ValueError("Expected the complete ten-case native-roundtrip bundle")
    records = [load_case(p) for p in folders if p.is_dir()]
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    starts = {s["scene"]: s for s in plan["starts"]}
    if any(r["case"]["plan_sha256"] != plan_hash or r["case"]["start"] != starts[r["case"]["start"]["scene"]]
           for r in records):
        raise ValueError("Case does not belong to the frozen generation plan")
    if any(r["protocol"] != protocol for r in records):
        raise ValueError("Mixed feature/metric protocols")
    output.mkdir(parents=True, exist_ok=True)
    sections = [row for record in records for row in section_rows(record)]
    write_csv(output / "sections.csv", sections)
    write_csv(output / "queries.csv", [dict(case_id=r["case"]["id"], **q) for r in records for q in r["queries"]])
    updates = [dict(case_id=r["case"]["id"], angle=r["case"]["angle"], **u) for r in records for u in r["updates"]]
    write_csv(output / "updates.csv", [{**u, "kept_new_ids": json.dumps(u["kept_new_ids"])} for u in updates])
    save_json(output / "banks.json", {r["case"]["id"]: r["banks"] for r in records})
    fvd_rows, influence, scenes = [], [], []
    with (source / "roundtrip_fvd_diagnostic.csv").open() as handle:
        recorded_fvd = {int(r["angle_deg"]): float(r["roundtrip_fvd"]) for r in csv.DictReader(handle)}
    for angle in (90, 360):
        group = [r for r in records if r["case"]["angle"] == angle]
        a = np.concatenate([r["outward"] for r in group])
        b = np.concatenate([r["returning"] for r in group])
        value = frechet(a, b)
        if abs(value - recorded_fvd[angle]) > .001:
            raise ValueError("Cached-feature FVD does not reproduce recorded result")
        fvd_rows.append(dict(angle=angle, videos=len(group), clips_per_leg=len(a), fvd=value))
        for r in group:
            queries = [q for q in r["queries"] if q["phase"] == "return"]
            scenes.append(dict(case_id=r["case"]["id"], angle=angle,
                               **{m: float(np.mean([p[m] for p in r["pairs"]])) for m in ("psnr_db", "ssim", "lpips")},
                               return_selected_rotation_deg=float(np.mean([q["selected_rotation_deg"] for q in queries])),
                               return_bank_best_rotation_deg=float(np.mean([q["bank_best_rotation_deg"] for q in queries])),
                               return_selection_gap_deg=float(np.mean([q["selection_rotation_gap_deg"] for q in queries])),
                               return_worst_bank_best_rotation_deg=max(q["bank_best_rotation_deg"] for q in queries),
                               within_scene_fvd_diagnostic=frechet(r["outward"], r["returning"])))
            rest = [x for x in group if x is not r]
            influence.append(dict(angle=angle, omitted_case=r["case"]["id"], videos=4,
                                  fvd=frechet(np.concatenate([x["outward"] for x in rest]),
                                              np.concatenate([x["returning"] for x in rest]))))
    write_csv(output / "fvd_reproduction.csv", fvd_rows)
    write_csv(output / "scene_summary.csv", scenes)
    write_csv(output / "leave_one_scene_out.csv", influence)
    if render:
        render_coverage(records, output)
        for record in records:
            print(f"Rendering {record['case']['id']}", flush=True)
            render_examples(record, output)
    hashes = {str(source / p): digest(source / p) for p in ("plan.json", "metric_protocol.json", "roundtrip_fvd_diagnostic.csv")}
    for r in records:
        hashes.update(r["verified"])
        for name in ("metrics.json", "generation.json"):
            path = source / "cases" / r["case"]["id"] / name
            hashes[str(path)] = digest(path)
    save_json(output / "provenance.json", dict(source=str(source.resolve()), verified_sha256=hashes,
              audit_sha256=digest(Path(__file__)), original_frames_downloaded=False,
              feature_extraction=False, generation=False, fixed_display_fractions=[.125, .375, .625, .875],
              limits=["MP4 images are visual proxies, not the lossless pixels used for saved metrics.",
                      "Rotation gaps describe commanded poses, not generated camera estimates or appearance utility.",
                      "The angular oracle is not the production Monte Carlo overlap objective.",
                      "No matched unbounded run or causal intervention is present.",
                      "FVD is outward-versus-return, not established as the paper's quality FVD.",
                      "Four-clip per-scene and leave-one-out values are diagnostics, not replacement results."]))
    lines = ["# Native Round-Trip Audit", "", "All ten cases are retained. No generation or feature extraction was run.",
             "Saved non-PNG generation artifacts and pair/feature caches match their receipts.",
             "Exact symmetric poses, complete reads, bank membership and logged counts pass validation.", "",
             "## Cached FVD", "", "| Angle | Videos | Clips per leg | Reproduced round-trip FVD |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['angle']} | {r['videos']} | {r['clips_per_leg']} | {r['fvd']:.3f} |" for r in fvd_rows]
    lines += ["", "## Return-Leg Camera Coverage", "", "Full history contains matching outward poses except for continuation-frame exclusions immediately after the turnaround.",
              "The following angles are averages over all return queries, including the final frame.", "",
              "| Scene (360 deg) | Selected mismatch | Best retained mismatch | Selection gap | Worst best-retained mismatch |", "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['case_id']} | {r['return_selected_rotation_deg']:.2f} | {r['return_bank_best_rotation_deg']:.2f} | {r['return_selection_gap_deg']:.2f} | {r['return_worst_bank_best_rotation_deg']:.2f} |" for r in scenes if r["angle"] == 360]
    lines += ["", "Interpretation: missing camera coverage is directly observed; its causal contribution to FVD is not measured.",
              "The reader optimizes FOV overlap, not angular distance. This is a pose diagnostic, not the DINO retention/selection figure.",
              "A new retention variant would need a separately labeled, matched evaluation; no existing scores were changed.", "",
              "## Bulk-Eviction Evidence", "", "Updates that keep only the protected endpoint from a new outward chunk:", ""]
    for u in updates:
        if u["angle"] == 360 and u["section"] < 4 and u["new_frames_kept"] == 1:
            lines.append(f"- {u['case_id']}, section {u['section']}: only frame {u['kept_new_ids'][0]} survives out of 76 new frames.")
    lines += ["", "Production scores are computed once on the pre-eviction candidates; FrameMemoryBuffer then evicts repeatedly without rescoring.",
              "The update log also records when a removed frame's nearest covisible neighbor is removed in the same batch (updates.csv).",
              "That alone does not prove no other substitute survives, but the angular coverage gaps directly measure missing requested views.",
              "A plausible test is sequential rescoring on the same pose/appearance affinity after each deletion, keeping B32 and all generation settings fixed.",
              "This would be a new update variant, not a correction to an incorrect stored score; an improvement in generation/FVD is untested.", "",
              "## Visual Review", "", "Fixed 1/8, 3/8, 5/8, 7/8 outward-leg samples; paired return views and selected memory IDs shown below each.",
              "Images are MP4 previews without exposure changes or alignment. Selected-ID thumbnails may differ from stored pixels at re-decoded chunk boundaries.",
              "The synthetic trajectory is not the dataset's original path: trace target_dataset_frame fields are not GT correspondences.", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"Audit complete: {output / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output, not args.no_render)


if __name__ == "__main__":
    main()
