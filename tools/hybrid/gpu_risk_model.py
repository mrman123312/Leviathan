#!/usr/bin/env python3
"""GPU-assisted reply/risk scoring for Leviathan Hybrid.

The model is intentionally advisory. CPU alpha-beta remains authoritative.
If no trained checkpoint is available, the scorer uses a deterministic
hand-coded prior so the hybrid controller remains functional and safe.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

FEATURE_NAMES = (
    "rank",
    "score_cp",
    "score_gap_cp",
    "depth",
    "seldepth",
    "depth_gap",
    "nodes_log10",
    "nps_log10",
    "hashfull_frac",
    "pv_len",
    "predicted_reply",
    "mate_flag",
)


@dataclass(frozen=True)
class ReplyFeatures:
    rank: float
    score_cp: float
    score_gap_cp: float
    depth: float
    seldepth: float
    nodes: float
    nps: float
    hashfull: float
    pv_len: float
    predicted_reply: float
    mate_flag: float = 0.0

    def vector(self) -> List[float]:
        return [
            float(self.rank),
            float(self.score_cp),
            float(self.score_gap_cp),
            float(self.depth),
            float(self.seldepth),
            float(self.seldepth - self.depth),
            math.log10(max(1.0, float(self.nodes))),
            math.log10(max(1.0, float(self.nps))),
            max(0.0, min(1.0, float(self.hashfull) / 1000.0)),
            float(self.pv_len),
            float(self.predicted_reply),
            float(self.mate_flag),
        ]


class GpuRiskScorer:
    """Rank opponent replies by likelihood * verification value."""

    def __init__(self, device: str = "auto", checkpoint: str | None = None):
        self.requested_device = device
        self.checkpoint = checkpoint
        self.torch = None
        self.device = "cpu"
        self.model = None
        self.mode = "heuristic"
        self._init_torch()
        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _init_torch(self) -> None:
        try:
            import torch  # type: ignore
        except Exception:
            return
        self.torch = torch
        if self.requested_device == "off":
            return
        if self.requested_device == "cpu":
            self.device = "cpu"
        elif self.requested_device in ("cuda", "auto") and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def _load_checkpoint(self, checkpoint: str) -> None:
        if self.torch is None:
            return
        torch = self.torch
        path = Path(checkpoint)
        if not path.exists():
            return
        payload = torch.load(path, map_location=self.device)
        in_features = len(FEATURE_NAMES)
        hidden = int(payload.get("hidden", 32)) if isinstance(payload, dict) else 32
        model = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2),
        )
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        try:
            model.load_state_dict(state)
        except Exception:
            return
        model.to(self.device)
        model.eval()
        self.model = model
        self.mode = "checkpoint"

    def score(self, rows: Sequence[ReplyFeatures]) -> List[dict]:
        if not rows:
            return []
        if self.model is not None and self.torch is not None:
            return self._score_model(rows)
        return self._score_heuristic(rows)

    def _score_model(self, rows: Sequence[ReplyFeatures]) -> List[dict]:
        torch = self.torch
        assert torch is not None and self.model is not None
        x = torch.tensor([r.vector() for r in rows], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits = self.model(x)
            reply_prob = torch.softmax(logits[:, 0], dim=0)
            risk = torch.sigmoid(logits[:, 1])
            utility = reply_prob * (1.0 + risk)
        rp = reply_prob.detach().cpu().tolist()
        rr = risk.detach().cpu().tolist()
        uu = utility.detach().cpu().tolist()
        return [
            {"reply_probability": float(p), "risk": float(r), "utility": float(u), "device": self.device, "mode": self.mode}
            for p, r, u in zip(rp, rr, uu)
        ]

    def _score_heuristic(self, rows: Sequence[ReplyFeatures]) -> List[dict]:
        logits = []
        risks = []
        for r in rows:
            gap = max(0.0, r.score_gap_cp)
            logits.append(-gap / 70.0 + 0.35 * r.predicted_reply - 0.08 * max(0.0, r.rank - 1.0))
            depth_gap = max(0.0, r.seldepth - r.depth)
            closeness = math.exp(-gap / 90.0)
            risk_raw = -0.8 + 1.25 * closeness + 0.035 * depth_gap + 0.25 * r.mate_flag
            risks.append(1.0 / (1.0 + math.exp(-risk_raw)))
        mx = max(logits)
        exps = [math.exp(v - mx) for v in logits]
        z = sum(exps) or 1.0
        probs = [v / z for v in exps]
        utilities = [p * (1.0 + r) for p, r in zip(probs, risks)]
        if self.torch is not None and self.device == "cuda":
            torch = self.torch
            p = torch.tensor(probs, dtype=torch.float32, device="cuda")
            r = torch.tensor(risks, dtype=torch.float32, device="cuda")
            utilities = (p * (1.0 + r)).cpu().tolist()
        return [
            {"reply_probability": float(p), "risk": float(r), "utility": float(u), "device": self.device, "mode": self.mode}
            for p, r, u in zip(probs, risks, utilities)
        ]

    def describe(self) -> dict:
        return {
            "requested_device": self.requested_device,
            "device": self.device,
            "mode": self.mode,
            "checkpoint": self.checkpoint,
            "torch_available": self.torch is not None,
            "cuda_available": bool(self.torch is not None and self.torch.cuda.is_available()),
        }


def self_test(device: str) -> int:
    scorer = GpuRiskScorer(device=device)
    rows = [
        ReplyFeatures(1, 35, 0, 14, 22, 50000, 750000, 120, 8, 1),
        ReplyFeatures(2, 22, 13, 14, 25, 50000, 750000, 120, 8, 0),
        ReplyFeatures(3, -40, 75, 14, 19, 50000, 750000, 120, 7, 0),
    ]
    out = scorer.score(rows)
    print(json.dumps({"runtime": scorer.describe(), "scores": out}, indent=2))
    if len(out) != 3 or not math.isclose(sum(x["reply_probability"] for x in out), 1.0, rel_tol=1e-5):
        return 2
    if out[0]["utility"] <= out[2]["utility"]:
        return 3
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "off"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test(args.device)
    print(json.dumps(GpuRiskScorer(device=args.device).describe(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
