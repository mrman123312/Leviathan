#!/usr/bin/env python3
"""Strict fail-closed wrapper for P19.2 loss-envelope certification."""
from __future__ import annotations

try:
    import run_p19_loss_envelope_v3 as base
except ImportError:
    from . import run_p19_loss_envelope_v3 as base


class StrictLossEnvelopeFunnel(base.LossEnvelopeFunnel):
    def census_replies(self, board, c, lev_color, reply_nodes, deep_nodes, token):
        after = board.copy(stack=True)
        after.push(c.move)
        if after.is_game_over(claim_draw=True):
            outcome = after.outcome(claim_draw=True)
            if outcome is not None and outcome.winner is None:
                c.worst_loss_pm = 0
                c.worst_cp = 0
                c.risk_delta_pm = 0
                c.certified = True
                return

        legal_n = after.legal_moves.count()
        p = base.root_views(self.p09, after, reply_nodes, legal_n, lev_color, self.tok(token + "-p09-replies"))
        s = base.root_views(self.sf, after, reply_nodes, legal_n, lev_color, self.tok(token + "-sf-replies"))
        union = list(dict.fromkeys(list(p.keys()) + list(s.keys())))
        seen = set(union)
        for m in after.legal_moves:
            if m not in seen:
                union.append(m)
                seen.add(m)

        replies = {}
        for m in union:
            rr = base.ReplyRisk(m, p.get(m), s.get(m))
            # A requested full-MultiPV line that did not arrive is unknown, and
            # unknown is never allowed to certify safety.
            if rr.broad_p09 is None:
                rr.broad_p09 = base.View(-base.MATE_SCORE, 1000, True)
            if rr.broad_sf is None:
                rr.broad_sf = base.View(-base.MATE_SCORE, 1000, True)
            replies[m] = rr

        dangerous = sorted(
            replies.values(),
            key=lambda r: (r.worst_loss(), -r.worst_cp()),
            reverse=True,
        )
        for i, rr in enumerate(dangerous[: min(self.dangerous_replies, len(dangerous))]):
            # Missing broad lines already hard-refute this certificate. There is
            # no reason to spend deep nodes pretending that absence is evidence.
            if rr.hard_refuted() and (
                rr.broad_p09 is not None and rr.broad_p09.loss_pm == 1000
                or rr.broad_sf is not None and rr.broad_sf.loss_pm == 1000
            ):
                continue
            leaf = after.copy(stack=True)
            leaf.push(rr.move)
            rr.deep_p09 = base.analyse_one(
                self.p09, leaf, deep_nodes, lev_color, self.tok(f"{token}-deep-p09-{i}")
            )
            rr.deep_sf = base.analyse_one(
                self.sf, leaf, deep_nodes, lev_color, self.tok(f"{token}-deep-sf-{i}")
            )

        c.replies = replies
        c.worst_loss_pm = max((r.worst_loss() for r in replies.values()), default=1000)
        c.worst_cp = min((r.worst_cp() for r in replies.values()), default=-base.MATE_SCORE)
        c.hard_refuted = any(r.hard_refuted() for r in replies.values())


base.LossEnvelopeFunnel = StrictLossEnvelopeFunnel

if __name__ == "__main__":
    raise SystemExit(base.main())
