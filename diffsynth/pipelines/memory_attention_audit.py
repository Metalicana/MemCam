import math
import random
from collections import defaultdict

import torch


def _evenly_spaced_indices(length, count, device):
    if length < 1:
        raise ValueError("Cannot sample from an empty target sequence")
    count = min(int(count), int(length))
    if count < 1:
        raise ValueError("query_count must be at least 1")
    if count == length:
        return torch.arange(length, device=device, dtype=torch.long)
    positions = (torch.arange(count, device=device, dtype=torch.float64) + 0.5)
    positions = torch.floor(positions * length / count).to(torch.long)
    return positions.clamp_(max=length - 1)


@torch.no_grad()
def sample_target_to_context_attention(
    q,
    k,
    *,
    num_heads,
    context_spatial,
    context_frame_indices,
    query_count=128,
    query_chunk_size=16,
):
    """Recover sampled target-query to context-key attention from projected q/k.

    The production attention kernel is left untouched. This function recomputes only
    a small, deterministic subset of attention rows and reduces context pixels back
    to the retrieved memory frame that produced each context slot.
    """
    if q.ndim != 3 or k.ndim != 3 or q.shape != k.shape:
        raise ValueError("q and k must have matching [batch, sequence, channels] shapes")
    if q.shape[0] != 1:
        raise ValueError("The attention audit currently supports batch size 1")
    if q.shape[-1] % num_heads != 0:
        raise ValueError("Attention channels must be divisible by num_heads")
    if context_spatial < 1:
        raise ValueError("context_spatial must be at least 1")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be at least 1")

    frame_indices = [int(frame_idx) for frame_idx in context_frame_indices]
    num_slots = len(frame_indices)
    context_tokens = num_slots * int(context_spatial)
    sequence_length = q.shape[1]
    target_tokens = sequence_length - context_tokens
    if num_slots < 1 or context_tokens >= sequence_length:
        raise ValueError(
            "Context layout does not leave any target tokens: "
            f"slots={num_slots}, spatial={context_spatial}, sequence={sequence_length}"
        )

    head_dim = q.shape[-1] // num_heads
    q_heads = q.reshape(1, sequence_length, num_heads, head_dim).permute(0, 2, 1, 3)
    k_heads = k.reshape(1, sequence_length, num_heads, head_dim).permute(0, 2, 1, 3)
    target_indices = _evenly_spaced_indices(target_tokens, query_count, q.device)
    target_indices = target_indices + context_tokens

    slot_attention_sum = torch.zeros(num_slots, device=q.device, dtype=torch.float32)
    observations = 0
    scale = 1.0 / math.sqrt(head_dim)
    key_transpose = k_heads.transpose(-1, -2)

    for start in range(0, target_indices.numel(), query_chunk_size):
        query_indices = target_indices[start : start + query_chunk_size]
        query = q_heads.index_select(2, query_indices)
        logits = torch.matmul(query, key_transpose) * scale
        probabilities = torch.softmax(logits.float(), dim=-1)
        context_probabilities = probabilities[..., :context_tokens]
        slot_probabilities = context_probabilities.reshape(
            1,
            num_heads,
            query_indices.numel(),
            num_slots,
            context_spatial,
        ).sum(dim=-1)
        slot_attention_sum += slot_probabilities.sum(dim=(0, 1, 2))
        observations += num_heads * query_indices.numel()

    slot_attention = (slot_attention_sum / observations).cpu().tolist()
    frame_rows = defaultdict(
        lambda: {"context_slots": [], "attention_total": 0.0}
    )
    for slot_idx, (frame_idx, attention) in enumerate(
        zip(frame_indices, slot_attention)
    ):
        frame_rows[frame_idx]["context_slots"].append(slot_idx)
        frame_rows[frame_idx]["attention_total"] += float(attention)

    memory_scores = []
    for frame_idx in sorted(frame_rows):
        row = frame_rows[frame_idx]
        slot_count = len(row["context_slots"])
        memory_scores.append(
            {
                "memory_frame": frame_idx,
                "context_slots": row["context_slots"],
                "slot_count": slot_count,
                "attention_total": row["attention_total"],
                "attention_per_slot": row["attention_total"] / slot_count,
            }
        )

    return {
        "query_count": int(target_indices.numel()),
        "context_token_count": int(context_tokens),
        "target_token_count": int(target_tokens),
        "context_attention_mass": float(sum(slot_attention)),
        "slot_attention": slot_attention,
        "memory_scores": memory_scores,
    }


class MemoryAttentionCollector:
    def __init__(
        self,
        *,
        context_spatial,
        context_frame_indices,
        query_count,
        query_chunk_size=16,
    ):
        self.context_spatial = int(context_spatial)
        self.context_frame_indices = [int(value) for value in context_frame_indices]
        self.query_count = int(query_count)
        self.query_chunk_size = int(query_chunk_size)
        self.layer_results = {}

    def callback(self, layer_idx):
        def capture(q, k, num_heads):
            self.layer_results[int(layer_idx)] = sample_target_to_context_attention(
                q,
                k,
                num_heads=num_heads,
                context_spatial=self.context_spatial,
                context_frame_indices=self.context_frame_indices,
                query_count=self.query_count,
                query_chunk_size=self.query_chunk_size,
            )

        return capture

    def aggregate(self):
        if not self.layer_results:
            raise RuntimeError("No attention layers were captured")

        ordered_layers = sorted(self.layer_results)
        slot_count = len(self.context_frame_indices)
        slot_attention = [0.0] * slot_count
        context_mass = 0.0
        for layer_idx in ordered_layers:
            result = self.layer_results[layer_idx]
            if len(result["slot_attention"]) != slot_count:
                raise RuntimeError("Attention layers produced incompatible context layouts")
            context_mass += result["context_attention_mass"]
            for slot_idx, value in enumerate(result["slot_attention"]):
                slot_attention[slot_idx] += float(value)

        layer_count = len(ordered_layers)
        slot_attention = [value / layer_count for value in slot_attention]
        context_mass /= layer_count
        frame_rows = defaultdict(
            lambda: {"context_slots": [], "attention_total": 0.0}
        )
        for slot_idx, (frame_idx, attention) in enumerate(
            zip(self.context_frame_indices, slot_attention)
        ):
            frame_rows[frame_idx]["context_slots"].append(slot_idx)
            frame_rows[frame_idx]["attention_total"] += attention

        memory_scores = []
        for frame_idx in sorted(frame_rows):
            row = frame_rows[frame_idx]
            count = len(row["context_slots"])
            memory_scores.append(
                {
                    "memory_frame": frame_idx,
                    "context_slots": row["context_slots"],
                    "slot_count": count,
                    "attention_total": row["attention_total"],
                    "attention_per_slot": row["attention_total"] / count,
                }
            )

        representative = self.layer_results[ordered_layers[0]]
        return {
            "probe_layers": ordered_layers,
            "query_count": representative["query_count"],
            "context_token_count": representative["context_token_count"],
            "target_token_count": representative["target_token_count"],
            "context_attention_mass": context_mass,
            "slot_attention": slot_attention,
            "memory_scores": memory_scores,
        }


def add_retrieval_controls(memory_scores, selected_contexts, section_start_frame):
    overlaps_by_slot = {
        int(slot_idx): float(overlap)
        for slot_idx, _, memory_frame, overlap in selected_contexts
        if memory_frame is not None
    }
    target_frames_by_slot = {
        int(slot_idx): int(target_frame)
        for slot_idx, target_frame, memory_frame, _ in selected_contexts
        if memory_frame is not None
    }

    enriched = []
    for source_row in memory_scores:
        row = dict(source_row)
        slots = row["context_slots"]
        overlaps = [overlaps_by_slot[slot] for slot in slots if slot in overlaps_by_slot]
        target_frames = [
            target_frames_by_slot[slot]
            for slot in slots
            if slot in target_frames_by_slot
        ]
        memory_frame = int(row["memory_frame"])
        row["retrieval_overlap_mean"] = (
            sum(overlaps) / len(overlaps) if overlaps else None
        )
        row["retrieval_overlap_max"] = max(overlaps) if overlaps else None
        row["memory_age_mean"] = (
            sum(target_frame - memory_frame for target_frame in target_frames)
            / len(target_frames)
            if target_frames
            else float(section_start_frame - memory_frame)
        )
        enriched.append(row)
    return enriched


def select_intervention_candidates(memory_scores, seed):
    """Choose distinct high-, low-, and random-attention memories."""
    if not memory_scores:
        return []
    ordered = sorted(
        memory_scores,
        key=lambda row: (float(row["attention_total"]), int(row["memory_frame"])),
    )
    chosen = []
    used = set()

    def add(row, role):
        frame_idx = int(row["memory_frame"])
        if frame_idx in used:
            return
        candidate = dict(row)
        candidate["intervention_role"] = role
        chosen.append(candidate)
        used.add(frame_idx)

    add(ordered[-1], "top_attention")
    add(ordered[0], "bottom_attention")
    remaining = [row for row in ordered if int(row["memory_frame"]) not in used]
    if remaining:
        add(random.Random(int(seed)).choice(remaining), "random_attention")
    return chosen
