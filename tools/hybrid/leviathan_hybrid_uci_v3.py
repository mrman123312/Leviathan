#!/usr/bin/env python3
"""P18.5 hybrid controller: race-safe fixed/multi-ponder lifecycle.

This keeps P18.2 expected-regret candidate ranking/portfolio logic but hardens the
UCI lifecycle around python-chess managed pondering. In particular, a GUI `stop`
never reconfigures or starts a child engine merely to manufacture a bestmove, and
pool construction re-checks cancellation after every blocking engine operation.
"""
from __future__ import annotations

try:
    from leviathan_hybrid_uci_v2 import *
except ImportError:
    from .leviathan_hybrid_uci_v2 import *


class HybridProxyV3(HybridProxyV2):
    def _stop_session_for_gui(self, s):
        """Detach a ponder session and return one UCI cancellation acknowledgement.

        python-chess only needs the cancelled ponder command to finish before it
        can start the next play command. We prefer the predicted scout's real
        bestmove, but `bestmove 0000` is a safe cancellation acknowledgement when
        the speculative pool was not ready yet. Crucially, this path never sends
        setoption/isready/go to an engine that may still be searching.
        """
        with self.state_lock:
            if self.session is s:
                self.session = None
            self.warm_match = None
            s.gui_stop_seen = True
            s.cancel.set()
            assignments = list(s.assignments.values())
            predicted = s.assignments.get(norm(s.predicted_position_cmd)) if s.assignments else None

        ack = None
        if predicted is not None and not predicted.stopped:
            ack = self.stop_assignment(predicted, False)

        for a in assignments:
            if a is predicted or a.stopped:
                continue
            self.stop_assignment(a, False)

        self.emit(ack or "bestmove 0000")
        self.log(
            "ponder_stop_ack_v3",
            generation=s.generation,
            predicted_reply=s.predicted_reply,
            assignments=len(assignments),
            used_real_bestmove=bool(ack),
        )

    def handle_stop(self):
        with self.state_lock:
            fg = self.foreground_engine
            s = self.session

        # Normal foreground search: its relay thread owns the response and will
        # emit the real bestmove after Stockfish receives stop.
        if fg is not None:
            try:
                fg.send("stop")
            except Exception as exc:
                self.log("foreground_stop_error_v3", error=repr(exc))
                self.emit("bestmove 0000")
            return

        if s is not None:
            self._stop_session_for_gui(s)
            return

        # Defensive fallback for a stray stop. Do not create a new search.
        try:
            self.primary.send("stop")
        except Exception:
            pass
        self.emit("bestmove 0000")
        self.log("stray_stop_ack_v3")

    def _prepare_multi_ponder(self, s):
        """P18.2 portfolio preparation with cancellation barriers around I/O."""
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
                cands.append(ReplyCandidate(s.predicted_reply, len(cands) + 1, best - 10, 0, 0, 0, 0, 0, 0, 0, True))
            for c in cands:
                c.predicted = c.predicted or c.move == s.predicted_reply
            best = max(c.score_cp for c in cands)
            feats = [
                ReplyFeatures(
                    c.rank, c.score_cp, max(0, best - c.score_cp), c.depth,
                    c.seldepth, c.nodes, c.nps, c.hashfull, c.pv_len,
                    1.0 if c.predicted else 0.0, float(c.mate_flag)
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
            engines = [self.primary] + self.ensure_spares(max(0, len(selected) - 1))

            for cand, threads, e in zip(selected, alloc, engines):
                if s.cancel.is_set() or self.session is not s:
                    break
                h = self.full_hash if e is self.primary else min(self.full_hash, self.scout_hash)
                self.configure_search_engine(e, max(1, threads), h, 1)

                # configure_search_engine() can block on readyok. A GUI stop may
                # arrive while it is blocked, so re-check before starting search.
                if s.cancel.is_set() or self.session is not s:
                    break

                pos = append_move_to_position(s.after_our_move_cmd, cand.move)
                e.drain()
                e.send(pos)
                e.send(s.go_ponder_cmd)
                a = ScoutAssignment(cand, e, norm(pos), max(1, threads), h, time.monotonic())
                local_assign[norm(pos)] = a

                # Close the tiny race between `go ponder` and publication.
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
                "ponder_pool_ready_v3",
                generation=s.generation,
                setup_ms=int((time.monotonic() - s.started_at) * 1000),
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

            # Annealing remains available for later research, but the benchmark
            # launcher sets --anneal-seconds 0 until its separate concurrency
            # implementation is proven.
            if self.anneal_seconds > 0 and len(local_assign) > self.min_final_scouts:
                threading.Thread(target=self._anneal_pool, args=(s,), daemon=True, name=f"ponder-anneal-{s.generation}").start()
        except Exception as exc:
            for a in local_assign.values():
                if not a.stopped:
                    try:
                        self.stop_assignment(a, False)
                    except Exception:
                        pass
            self.log("ponder_prepare_error_v3", generation=s.generation, error=repr(exc))
            s.ready.set()


def main():
    return HybridProxyV3(build_v2_parser().parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
