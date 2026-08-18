#!/usr/bin/env python3
"""P18.8 hybrid controller: fail-closed learned-advisor authority.

P18.7 proved the dedicated-scout lifecycle stable, but two decisive GPU-on
losses both became draws when the learned checkpoint was removed. The
provisional checkpoint also had zero validation rows for the risk head and no
validated regret MAE. P18.8 therefore makes advisor authority explicit and
fail-closed.

Default mode is ``shadow``:
- the proven heuristic scheduler remains authoritative;
- the learned DirectML checkpoint may score the same candidates for telemetry;
- learned scores can NEVER choose/promote the authoritative warm scout.

``qualified`` mode exists for future checkpoints, but it only activates learned
authority when the checkpoint has actual risk/regret validation and prospective
metrics. Otherwise it falls back to the heuristic scheduler automatically.
"""
from __future__ import annotations
import math

try:
    from leviathan_hybrid_uci_v5 import *
except ImportError:
    from .leviathan_hybrid_uci_v5 import *


class HybridProxyV6(HybridProxyV5):
    def __init__(self, args):
        super().__init__(args)
        self.advisor_authority_requested = getattr(args, "advisor_authority", "shadow")
        self.learned_scorer = self.scorer
        self.authority_report = self._checkpoint_authority_report(self.learned_scorer)

        # Heuristic scheduler is the fail-closed baseline. It is exactly the
        # advisor used by the existing --gpu-device off counterfactual.
        self.baseline_scorer = GpuRiskScorer("off", None)
        learned_allowed = (
            self.advisor_authority_requested == "qualified"
            and self.authority_report["qualified"]
        )
        self.learned_authority_active = bool(learned_allowed)
        self.scorer = self.learned_scorer if learned_allowed else self.baseline_scorer

        self.log(
            "advisor_authority_v6",
            requested=self.advisor_authority_requested,
            learned_authority_active=self.learned_authority_active,
            report=self.authority_report,
            authoritative=self.scorer.describe(),
            shadow=self.learned_scorer.describe(),
        )

    @staticmethod
    def _finite_number(value):
        try:
            return value is not None and math.isfinite(float(value))
        except Exception:
            return False

    def _checkpoint_authority_report(self, scorer):
        desc = scorer.describe()
        metrics = desc.get("metrics") or {}
        valid = metrics.get("valid") or {}
        prospective = metrics.get("prospective")
        risk_rows = int(valid.get("risk_labeled_rows") or 0)
        risk_auc = valid.get("risk_auc")
        risk_pr = valid.get("risk_pr_auc")
        regret_mae = valid.get("regret_mae_cp")
        promoted = metrics.get("promote") is True

        checks = {
            "checkpoint_loaded": desc.get("mode") == "checkpoint",
            "trainer_promoted": promoted,
            "risk_validation_rows_ge_200": risk_rows >= 200,
            "risk_auc_present": self._finite_number(risk_auc),
            "risk_pr_auc_present": self._finite_number(risk_pr),
            "regret_mae_present": self._finite_number(regret_mae),
            "prospective_metrics_present": isinstance(prospective, dict) and bool(prospective),
        }
        return {
            "qualified": all(checks.values()),
            "checks": checks,
            "risk_labeled_rows": risk_rows,
            "risk_auc": risk_auc,
            "risk_pr_auc": risk_pr,
            "regret_mae_cp": regret_mae,
        }

    def _prepare_multi_ponder(self, s):
        # V5 builds the real portfolio using self.scorer. In shadow mode that is
        # deliberately the heuristic counterfactual scorer, so the learned model
        # cannot influence branch selection, thread allocation, or promotion.
        super()._prepare_multi_ponder(s)

        if self.learned_authority_active:
            return
        if self.learned_scorer is self.baseline_scorer:
            return
        if s.cancel.is_set() or self.session is not s or not s.candidates:
            return

        try:
            candidates = list(s.candidates)
            best = max(c.score_cp for c in candidates)
            feats = [
                ReplyFeatures(
                    c.rank,
                    c.score_cp,
                    max(0, best - c.score_cp),
                    c.depth,
                    c.seldepth,
                    c.nodes,
                    c.nps,
                    c.hashfull,
                    c.pv_len,
                    1.0 if c.predicted else 0.0,
                    float(c.mate_flag),
                )
                for c in candidates
            ]
            shadow_scores = self.learned_scorer.score(feats)
            self.log(
                "advisor_shadow_scores_v6",
                generation=s.generation,
                authority="heuristic",
                learned_authority_active=False,
                checkpoint_report=self.authority_report,
                candidates=[
                    {
                        "move": c.move,
                        "rank": c.rank,
                        "authoritative_reply_probability": c.reply_probability,
                        "authoritative_risk": c.risk,
                        "authoritative_expected_regret_cp": getattr(c, "expected_regret_cp", 0.0),
                        "authoritative_utility": c.utility,
                        "shadow_reply_probability": float(x["reply_probability"]),
                        "shadow_risk": float(x["risk"]),
                        "shadow_expected_regret_cp": float(x.get("expected_regret_cp", 0.0)),
                    }
                    for c, x in zip(candidates, shadow_scores)
                ],
                shadow=self.learned_scorer.describe(),
            )
        except Exception as exc:
            # Shadow telemetry is never allowed to affect the authoritative game.
            self.log("advisor_shadow_error_v6", generation=s.generation, error=repr(exc))


def build_v6_parser():
    p = build_v2_parser()
    p.description = "Leviathan P18.8 fail-closed advisor-authority proxy"
    p.add_argument(
        "--advisor-authority",
        choices=("shadow", "qualified"),
        default="shadow",
        help="shadow keeps heuristic authority; qualified enables learned authority only if checkpoint safety gates pass",
    )
    return p


def main():
    return HybridProxyV6(build_v6_parser().parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
