"""STRUCTURED_A11_IDENTITY_V1 — DeepSets residual on [z_j, z_i, a11(j,i)].

Does not replace B0. Trains only token MLP + aggregator + gate + delta head.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.lastfm_lp.binary.user_hcr_aggregation import _truncate_history
from src.lastfm_lp.pipeline.features_hcr_v2 import _hist_for_row, ensure_cross_fit

MAX_HISTORY = 25
TOKEN_HIDDEN = 64
TOKEN_OUT = 32
GATE_HIDDEN = 64
EMBED_DIM = 64
TOKEN_DIM = EMBED_DIM * 2 + 1  # z_j, z_i, a11
GATE_TARGET_MEAN = 0.07
MODE_TRUE = "S1_TRUE"
MODE_ZERO = "S1_ZERO"
MODE_SHUFFLED = "S1_SHUFFLED"


def _hist_list(u: int, i: int, y: int, model_train: dict[int, set[int]]) -> list[int]:
    """Canonical history with train self-exclusion + stable item order."""
    return sorted(_truncate_history(_hist_for_row(u, i, y, model_train), MAX_HISTORY))


def materialize_token_banks(
    bundle: dict[str, Any],
    out_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> Path:
    """Write hist_ids / mask / a11_true for each split (padded to max_history)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cf = ensure_cross_fit(bundle)
    model_train = bundle["model_train"]
    pop = bundle["popularity"].astype(np.int64)

    for split in splits:
        ids_p = out_dir / f"hist_ids_{split}.npy"
        mask_p = out_dir / f"hist_mask_{split}.npy"
        a11_p = out_dir / f"a11_true_{split}.npy"
        meta_p = out_dir / f"bank_meta_{split}.json"
        if ids_p.exists() and mask_p.exists() and a11_p.exists() and meta_p.exists():
            print(f"[bank] reuse {split}", flush=True)
            continue

        pairs = bundle[f"{split}_pairs"]
        users = pairs["user_id"].astype(np.int64)
        items = pairs["item_id"].astype(np.int64)
        labels = pairs["label"].astype(np.int64)
        n = len(users)
        hist_ids = np.full((n, MAX_HISTORY), -1, dtype=np.int64)
        hist_mask = np.zeros((n, MAX_HISTORY), dtype=np.float32)
        a11 = np.zeros((n, MAX_HISTORY), dtype=np.float32)

        # Group rows that share (fold-index key, identical history tuple).
        groups: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        for r in range(n):
            u = int(users[r])
            hist = _hist_list(u, int(items[r]), int(labels[r]), model_train)
            # fold id for train; -1 for val/test (full index)
            if split in {"val", "valid", "validation", "test"}:
                fold_key = -1
            else:
                fold_key = int(cf.user_to_fold.get(u, -1))
            groups[(fold_key, tuple(hist))].append(r)

        for (fold_key, hist_t), rows in tqdm(groups.items(), desc=f"bank:{split}", leave=False):
            hist = list(hist_t)
            H = len(hist)
            if H == 0:
                continue
            if fold_key < 0:
                pw = cf.full_index
            else:
                pw = cf.fold_indices[fold_key]
            cands = items[rows]
            a_block, _ = pw.a11_energy_block(
                np.asarray(hist, dtype=np.int64),
                cands.astype(np.int64),
            )
            hist_arr = np.asarray(hist, dtype=np.int64)
            for local, r in enumerate(rows):
                hist_ids[r, :H] = hist_arr
                hist_mask[r, :H] = 1.0
                a11[r, :H] = a_block[:, local].astype(np.float32)

        np.save(ids_p, hist_ids)
        np.save(mask_p, hist_mask)
        np.save(a11_p, a11)
        meta = {
            "split": split,
            "n_rows": n,
            "max_history": MAX_HISTORY,
            "n_groups": len(groups),
            "mean_H": float(hist_mask.sum(axis=1).mean()),
            "a11_mean": float(a11[hist_mask > 0].mean()) if (hist_mask > 0).any() else 0.0,
            "a11_std": float(a11[hist_mask > 0].std()) if (hist_mask > 0).any() else 0.0,
            "pop_used_for_support_bins": True,
            "n_items_pop": int(pop.shape[0]),
        }
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[bank] wrote {split} n={n} mean_H={meta['mean_H']:.2f}", flush=True)

    (out_dir / "BANKS_COMPLETE").write_text("1\n", encoding="utf-8")
    return out_dir


def _support_bin(pop: int) -> int:
    if pop <= 5:
        return 0
    if pop <= 20:
        return 1
    if pop <= 100:
        return 2
    if pop <= 500:
        return 3
    return 4


def materialize_shuffled_a11(
    bundle: dict[str, Any],
    bank_dir: Path,
    out_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    seed: int = 2026,
) -> Path:
    """Destroy (j,i)->a11 within each split; preserve support-bin marginals."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = Path(bank_dir)
    pop = bundle["popularity"].astype(np.int64)
    rng = np.random.default_rng(seed)

    for split in splits:
        out_p = out_dir / f"a11_shuffled_{split}.npy"
        if out_p.exists():
            print(f"[shuffle] reuse {split}", flush=True)
            continue
        hist_ids = np.load(bank_dir / f"hist_ids_{split}.npy")
        hist_mask = np.load(bank_dir / f"hist_mask_{split}.npy")
        a11 = np.load(bank_dir / f"a11_true_{split}.npy").copy()
        items = bundle[f"{split}_pairs"]["item_id"].astype(np.int64)
        n, H = a11.shape
        # Collect valid tokens by (hist_support_bin, cand_support_bin)
        buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for r in range(n):
            cand_b = _support_bin(int(pop[int(items[r])]))
            for h in range(H):
                if hist_mask[r, h] <= 0:
                    continue
                j = int(hist_ids[r, h])
                buckets[(_support_bin(int(pop[j])), cand_b)].append((r, h))
        shuffled = a11.copy()
        n_moved = 0
        for key, positions in buckets.items():
            if len(positions) < 2:
                continue
            vals = np.array([a11[r, h] for r, h in positions], dtype=np.float32)
            perm = rng.permutation(len(vals))
            for t, (r, h) in enumerate(positions):
                shuffled[r, h] = vals[perm[t]]
            n_moved += int((perm != np.arange(len(vals))).sum())
        np.save(out_p, shuffled)
        meta = {
            "split": split,
            "seed": seed,
            "n_buckets": len(buckets),
            "n_tokens_moved_est": n_moved,
            "corr_true_shuffled": float(
                np.corrcoef(a11[hist_mask > 0], shuffled[hist_mask > 0])[0, 1]
            )
            if (hist_mask > 0).any()
            else float("nan"),
            "note": "Within-split shuffle by (hist_pop_bin, cand_pop_bin); no cross-split mixing",
        }
        (out_dir / f"a11_shuffled_{split}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[shuffle] {split} corr(true,shuffled)={meta['corr_true_shuffled']:.4f}", flush=True)
    return out_dir


class StructuredDeepSetsBranch(nn.Module):
    """Token MLP → masked mean → gate × delta residual."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        token_hidden: int = TOKEN_HIDDEN,
        token_out: int = TOKEN_OUT,
        gate_hidden: int = GATE_HIDDEN,
        gate_bias_init: float | None = None,
    ) -> None:
        super().__init__()
        token_dim = embed_dim * 2 + 1
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, token_out),
        )
        # gate inputs: z_u, z_i, h_struct  (+ optional compact graph ctx later)
        gate_in = embed_dim * 2 + token_out
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(token_out, token_out),
            nn.GELU(),
            nn.Linear(token_out, 1),
        )
        # Init gate near GATE_TARGET_MEAN
        if gate_bias_init is None:
            p = GATE_TARGET_MEAN
            gate_bias_init = float(np.log(p / (1.0 - p)))
        last = self.gate_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, gate_bias_init)
        self.gate_bias_init = float(gate_bias_init)
        self.gate_target_mean = float(GATE_TARGET_MEAN)

    def forward(
        self,
        z_u: torch.Tensor,  # [B, d]
        z_i: torch.Tensor,  # [B, d]
        z_hist: torch.Tensor,  # [B, H, d]
        a11: torch.Tensor,  # [B, H]
        mask: torch.Tensor,  # [B, H]
    ) -> dict[str, torch.Tensor]:
        B, H, d = z_hist.shape
        z_i_exp = z_i.unsqueeze(1).expand(B, H, d)
        a = a11.unsqueeze(-1)
        tokens = torch.cat([z_hist, z_i_exp, a], dim=-1)  # [B,H,2d+1]
        h_ji = self.token_mlp(tokens)  # [B,H,out]
        m = mask.unsqueeze(-1)
        h_sum = (h_ji * m).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        h_struct = h_sum / denom  # mean over valid H only
        gate_in = torch.cat([z_u, z_i, h_struct], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)
        delta = self.delta_head(h_struct).squeeze(-1)
        return {
            "h_struct": h_struct,
            "gate": gate,
            "delta_struct": delta,
            "token_out": h_ji,
            "residual": gate * delta,
        }


class FrozenB0PlusStructured(nn.Module):
    """Frozen B0 RecommendationModel + trainable structured residual."""

    def __init__(self, b0: nn.Module, branch: StructuredDeepSetsBranch) -> None:
        super().__init__()
        self.b0 = b0
        self.branch = branch
        for p in self.b0.parameters():
            p.requires_grad = False

    def encode_all(self) -> torch.Tensor:
        self.b0.eval()
        with torch.no_grad():
            return self.b0.encoder.encode_all()

    def base_logits(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        A: torch.Tensor,
        H: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        self.b0.eval()
        with torch.no_grad():
            out = self.b0(user_idx, item_idx, A, H, z=z)
        return out["logits"]

    def forward_residual(
        self,
        z: torch.Tensor,
        users: torch.Tensor,  # raw user ids [B]
        items: torch.Tensor,  # raw item ids [B]
        hist_ids: torch.Tensor,  # [B,H] item ids (-1 pad)
        hist_mask: torch.Tensor,
        a11: torch.Tensor,
        item_offset: int,
    ) -> dict[str, torch.Tensor]:
        z_u = z[users.long()]
        z_i = z[items.long() + item_offset]
        # gather history embeddings; pad positions unused due to mask
        hist_node = hist_ids.clone()
        hist_node = torch.where(hist_node >= 0, hist_node + item_offset, torch.zeros_like(hist_node))
        z_hist = z[hist_node.long()]  # [B,H,d]
        return self.branch(z_u, z_i, z_hist, a11, hist_mask)


def count_trainable_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def count_all_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def run_leakage_checks(
    bundle: dict[str, Any],
    bank_dir: Path,
    shuffle_dir: Path,
) -> dict[str, Any]:
    """Pre-train leakage / protocol checks. Any fail → STOP."""
    cf = ensure_cross_fit(bundle)
    bank_dir = Path(bank_dir)
    shuffle_dir = Path(shuffle_dir)
    report: dict[str, Any] = {"ok": True, "checks": {}}

    # 1. HCR fit population excludes expected fold (train)
    u_sample = int(bundle["train_pairs"]["user_id"][0])
    fold = cf.user_to_fold[u_sample]
    idx_train = cf.index_for(u_sample, split="train")
    idx_val = cf.index_for(u_sample, split="val")
    c1 = {
        "user": u_sample,
        "fold": int(fold),
        "train_index_is_leave_one_fold": idx_train is cf.fold_indices[fold],
        "val_index_is_full": idx_val is cf.full_index,
    }
    c1["pass"] = bool(c1["train_index_is_leave_one_fold"] and c1["val_index_is_full"])
    report["checks"]["1_cross_fit_index"] = c1

    # 2. val/test interactions not in HCR fit (model_train is fit; val holdout items
    #    for a user should not appear in that user's model_train set)
    model_train = bundle["model_train"]
    val_u = bundle["val_pairs"]["user_id"]
    val_i = bundle["val_pairs"]["item_id"]
    val_y = bundle["val_pairs"]["label"]
    pos_in_train = 0
    pos_checked = 0
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(val_y), size=min(5000, len(val_y)), replace=False)
    for r in sample_idx:
        if int(val_y[r]) != 1:
            continue
        pos_checked += 1
        u, i = int(val_u[r]), int(val_i[r])
        if i in model_train.get(u, ()):
            pos_in_train += 1
    c2 = {
        "val_pos_checked": pos_checked,
        "val_pos_also_in_model_train": pos_in_train,
        "pass": pos_in_train == 0,
        "note": "Val positives must be holdouts not present in model_train history",
    }
    report["checks"]["2_val_not_in_fit_history"] = c2

    # 3. train candidate self-information removed
    tr_u = bundle["train_pairs"]["user_id"]
    tr_i = bundle["train_pairs"]["item_id"]
    tr_y = bundle["train_pairs"]["label"]
    hist_ids = np.load(bank_dir / "hist_ids_train.npy")
    hist_mask = np.load(bank_dir / "hist_mask_train.npy")
    bad_self = 0
    checked = 0
    sample_tr = rng.choice(len(tr_y), size=min(10000, len(tr_y)), replace=False)
    for r in sample_tr:
        if int(tr_y[r]) != 1:
            continue
        checked += 1
        i = int(tr_i[r])
        ids = hist_ids[r][hist_mask[r] > 0]
        if i in set(ids.tolist()):
            bad_self += 1
    c3 = {
        "train_pos_checked": checked,
        "candidate_still_in_history": bad_self,
        "pass": bad_self == 0,
    }
    report["checks"]["3_train_self_exclusion"] = c3

    # 4. z_j from frozen encoder protocol — documented; checked at load time
    report["checks"]["4_z_from_frozen_hgt"] = {
        "pass": True,
        "note": "z from frozen B0 HGT encode_all; encoder requires_grad=False",
    }

    # 5. shuffle does not cross split boundaries
    a_tr_t = np.load(bank_dir / "a11_true_train.npy")
    a_tr_s = np.load(shuffle_dir / "a11_shuffled_train.npy")
    a_va_t = np.load(bank_dir / "a11_true_val.npy")
    a_va_s = np.load(shuffle_dir / "a11_shuffled_val.npy")
    # Shuffled files exist per-split with independent RNG streams started from same seed
    # but applied separately — pass if shapes match and train≠val arrays
    c5 = {
        "train_shape_match": a_tr_t.shape == a_tr_s.shape,
        "val_shape_match": a_va_t.shape == a_va_s.shape,
        "train_changed": not np.allclose(a_tr_t, a_tr_s),
        "val_changed": not np.allclose(a_va_t, a_va_s),
        "pass": True,
        "note": "Shuffle applied independently per split file; no train↔val mixing",
    }
    c5["pass"] = bool(
        c5["train_shape_match"]
        and c5["val_shape_match"]
        and c5["train_changed"]
        and c5["val_changed"]
    )
    report["checks"]["5_shuffle_split_boundary"] = c5

    # 6. label does not affect token selection beyond train self-exclusion
    # (history from model_train only; labels only used for y==1 self-removal)
    report["checks"]["6_label_token_selection"] = {
        "pass": True,
        "note": "History from model_train; label only triggers candidate self-exclusion on train positives",
    }

    # 7. padding masked
    m = np.load(bank_dir / "hist_mask_val.npy")
    ids = np.load(bank_dir / "hist_ids_val.npy")
    pad_ok = bool(((m <= 0) | (ids >= 0)).all())
    # padded slots are -1
    pad_ids_neg = bool((ids[m <= 0] == -1).all()) if (m <= 0).any() else True
    report["checks"]["7_padding_mask"] = {
        "pass": pad_ok and pad_ids_neg,
        "pad_fraction": float((m <= 0).mean()),
    }

    # 8. row identity: hist belongs to user
    bad_hist = 0
    sample_v = rng.choice(len(val_u), size=min(2000, len(val_u)), replace=False)
    for r in sample_v:
        u = int(val_u[r])
        allowed = model_train.get(u, set())
        ids_r = ids[r][m[r] > 0]
        for j in ids_r.tolist():
            if int(j) not in allowed:
                bad_hist += 1
                break
    report["checks"]["8_history_belongs_to_user"] = {
        "rows_checked": int(len(sample_v)),
        "rows_with_foreign_history": bad_hist,
        "pass": bad_hist == 0,
    }

    report["ok"] = all(c.get("pass", False) for c in report["checks"].values())
    return report


def decide_seed101(
    metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Decision gate after seed101. metrics keys: B0, S1_TRUE, S1_ZERO, S1_SHUFFLED."""
    b0 = metrics["B0"]["NDCG@20"]
    t = metrics["S1_TRUE"]["NDCG@20"]
    z = metrics["S1_ZERO"]["NDCG@20"]
    s = metrics["S1_SHUFFLED"]["NDCG@20"]
    d_base = t - b0
    d_zero = t - z
    d_shuf = t - s
    out = {
        "D_base": d_base,
        "D_zero": d_zero,
        "D_shuffle": d_shuf,
        "next": "STOP",
        "run_seed202": False,
        "run_seed303": False,
    }
    # CASE A
    if d_base >= 0.002 and d_zero > 0 and d_shuf > 0:
        out["verdict"] = "STRUCTURED_A11_SIGNAL_PRESENT"
        out["next"] = "run seed202 (TRUE/ZERO/SHUFFLED)"
        out["run_seed202"] = True
        return out
    # CASE B
    if 0.001 <= d_base < 0.002 and t > z and t > s:
        out["verdict"] = "INCONCLUSIVE_POSITIVE"
        out["next"] = "run seed202 and seed303"
        out["run_seed202"] = True
        out["run_seed303"] = True
        return out
    # CASE E — identity encoder interesting without HCR
    if (z - b0) >= 0.002 and d_zero <= 0:
        out["verdict"] = "HISTORY_IDENTITY_ENCODER_SIGNAL_PRESENT"
        out["HISTORY_IDENTITY_ENCODER_SIGNAL"] = "PRESENT"
        out["HCR_INCREMENTAL_SIGNAL"] = "NOT_ESTABLISHED"
        return out
    # CASE C — architecture only
    if d_base > 0 and (abs(d_zero) < 0.001 or abs(d_shuf) < 0.001 or d_zero <= 0 or d_shuf <= 0):
        if t > b0 and (d_zero <= 0.0005 or d_shuf <= 0.0005):
            out["verdict"] = "ARCHITECTURE_GAIN_NOT_HCR_GAIN"
            return out
    # CASE D
    if t <= b0:
        out["verdict"] = "STRUCTURED_A11_DEEPSETS_NOT_SUPPORTED"
        return out
    out["verdict"] = "STRUCTURED_A11_DEEPSETS_NOT_SUPPORTED"
    return out


ModeName = Literal["S1_TRUE", "S1_ZERO", "S1_SHUFFLED"]
