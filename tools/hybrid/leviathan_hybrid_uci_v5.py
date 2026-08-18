#!/usr/bin/env python3
"""P18.7 hybrid controller: dedicated speculative scouts.

V4 fixed the foreground bestmove publication race. This version removes the
remaining process-role aliasing: the current authoritative foreground engine is
never used as a speculative ponder scout. Ponder work is performed only by
idle/dedicated child engines. A matching scout may still be atomically promoted
for the next foreground search, at which point the old foreground engine becomes
an idle scout.

This preserves the P18 architecture and CPU budget while ensuring that no engine
process is simultaneously owned by the foreground relay and speculative pool.
"""
from __future__ import annotations

try:
    from leviathan_hybrid_uci_v4 import *
except ImportError:
    from .leviathan_hybrid_uci_v4 import *


class HybridProxyV5(HybridProxyV4):
    def prewarm_workers(self):
        """Pre-create a full dedicated scout pool; primary stays foreground-only."""
        try:
            self.ensure_predictor()
            target = max(1, min(self.max_scouts, self.full_threads))
            self.ensure_spares(target)
            self.log(
                "workers_prewarmed_v5",
                scouts=len(self.spare_engines),
                predictor=True,
                dedicated_primary=True,
            )
        except Exception as exc:
            self.log("prewarm_error_v5", error=repr(exc))

    def _prepare_multi_ponder(self, s):
        """Build P18 portfolio only on idle scout processes, never self.primary."""
        local_assign = {}
        try:
            k = max(1, min(self.max_scouts, self.full_threads))
            cands = self.generate_reply_candidates(s, max(k + 4, k))
            if s.cancel.is_set() or self.session is not s:
                return
            if not cands:
                if s.predicted_reply:
                    cands = [ReplyCandidate(s.predicted_reply, 1, 0, 0, 0, 0, 0, 0, 0, 0, True)]
                else:
                    return
            if s.predicted_reply and all(c.move != s.predicted_reply for c in cands):
                best = cands[0].score_cp if cands else 0
                cands.append(
                    ReplyCandidate(
                        s.predicted_reply, len(cands) + 1, best - 10,
                        0, 0, 0, 0, 0, 0, 0, True,
                    )
                )
            for c in cands:
                c.predicted = c.predicted or c.move == s.predicted_reply

            best = max(c.score_cp for c in cands)
            feats = [
                ReplyFeatures(
                    c.rank, c.score_cp, max(0, best - c.score_cp), c.depth,
                    c.seldepth, c.nodes, c.nps, c.hashfull, c.pv_len,
                    1.0 if c.predicted else 0.0, float(c.mate_flag),
                )
                for c in cands
            ]
            scores = self.scorer.score(feats)
            if s.cancel.is_set() or self.session is not s:
                return

            for c, x in zip(cands, scores):
                c.reply_probability = float(x["reply_probability"])
                c.risk = float(x["risk"])
                c.expected_regret_cp = float(x.get("expected_regret_cp", 25 * c.risk))
                c.utility = self.candidate_value(c)

            cands.sort(key=lambda c: (c.utility, c.predicted, -c.rank), reverse=True)
            selected = cands[:k]
            if s.predicted_reply and all(c.move != s.predicted_reply for c in selected):
                selected[-1] = next(c for c in cands if c.move == s.predicted_reply)
            selected.sort(key=lambda c: c.utility, reverse=True)
            alloc = allocate_integer_budget(self.full_threads, [c.utility for c in selected], 1)

            # CRITICAL V5 INVARIANT: never include self.primary in the ponder pool.
            engines = list(self.ensure_spares(len(selected)))
            if any(e is self.primary for e in engines):
                # This should only be possible if external code corrupted role lists.
                engines = [e for e in engines if e is not self.primary]
                while len(engines) < len(selected):
                    # ensure_spares cannot create past its current length target when
                    # primary was accidentally present in the list, so create a clean
                    # extra child directly and publish it into the spare pool.
                    with self.pool_lock:
                        label = f"scout-{len(self.spare_engines) + 1}"
                        e = EngineProcess(self.args.engine, label)
                        self.init_engine(e, True)
                        self.spare_engines.append(e)
                        engines.append(e)

            for cand, threads, e in zip(selected, alloc, engines):
                if s.cancel.is_set() or self.session is not s:
                    break
                if e is self.primary:
                    raise RuntimeError("V5 invariant violated: primary selected as ponder scout")
                h = min(self.full_hash, self.scout_hash)
                self.configure_search_engine(e, max(1, threads), h, 1)
                if s.cancel.is_set() or self.session is not s:
                    break

                pos = append_move_to_position(s.after_our_move_cmd, cand.move)
                e.drain()
                e.send(pos)
                e.send(s.go_ponder_cmd)
                a = ScoutAssignment(cand, e, norm(pos), max(1, threads), h, time.monotonic())
                local_assign[norm(pos)] = a

                if s.cancel.is_set() or self.session is not s:
                    self.stop_assignment(a, False)
                    break

            if s.cancel.is_set() or self.session is not s:
                for a in local_assign.values():
                    if not a.stopped:
                        self.stop_assignment(a, False)
                return

            with self.state_lock:
                if s.cancel.is_set() or self.session is not s:
                    publish = False
                else:
                    s.candidates = selected
                    s.assignments = dict(local_assign)
                    s.ready.set()
                    publish = True

            if not publish:
                for a in local_assign.values():
                    if not a.stopped:
                        self.stop_assignment(a, False)
                return

            self.log(
                "ponder_pool_ready_v5",
                generation=s.generation,
                setup_ms=int((time.monotonic() - s.started_at) * 1000),
                dedicated_primary=True,
                primary_label=self.primary.label,
                scout_labels=[a.engine.label for a in local_assign.values()],
                candidates=[
                    {
                        "move": c.move,
                        "rank": c.rank,
                        "reply_probability": c.reply_probability,
                        "risk": c.risk,
                        "expected_regret_cp": c.expected_regret_cp,
                        "utility": c.utility,
                        "threads": alloc[i],
                    }
                    for i, c in enumerate(selected[:len(local_assign)])
                ],
                gpu=self.scorer.describe(),
            )

            # Benchmark launchers keep this at zero. If annealing is revisited,
            # it operates only on dedicated scouts under this architecture.
            if self.anneal_seconds > 0 and len(local_assign) > self.min_final_scouts:
                threading.Thread(
                    target=self._anneal_pool,
                    args=(s,),
                    daemon=True,
                    name=f"ponder-anneal-{s.generation}",
                ).start()

        except Exception as exc:
            for a in local_assign.values():
                if not a.stopped:
                    try:
                        self.stop_assignment(a, False)
                    except Exception:
                        pass
            self.log("ponder_prepare_error_v5", generation=s.generation, error=repr(exc))
            s.ready.set()


def main():
    return HybridProxyV5(build_v2_parser().parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
