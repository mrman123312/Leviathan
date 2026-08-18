#!/usr/bin/env python3
"""P18.6 hybrid controller: commit foreground bestmove state before GUI emission.

V3 hardened ponder cancellation, but the base foreground relay still emitted a
`bestmove` line to the GUI before committing the internal state that the next
python-chess ponder command depends on. Because python-chess can react to
`bestmove` immediately, the next `position`/`go ponder` could race the relay and
reconfigure/read from the same child engine while the relay still owned it.

V4 closes that handoff race by making bestmove publication two-phase:
1. parse + commit all relay-owned state under state_lock;
2. only then emit bestmove to the GUI.

Non-bestmove info lines are still streamed immediately.
"""
from __future__ import annotations

try:
    from leviathan_hybrid_uci_v3 import *
except ImportError:
    from .leviathan_hybrid_uci_v3 import *


class HybridProxyV4(HybridProxyV3):
    def _relay_loop(self, e):
        try:
            while True:
                line = e.read(3600)
                if line == "__LEV_ENGINE_EOF__":
                    with self.state_lock:
                        if self.foreground_engine is e:
                            self.foreground_engine = None
                    self.log("relay_eof_v4", engine=e.label)
                    self.emit("bestmove 0000")
                    break

                if line.startswith("bestmove"):
                    best, ponder = parse_bestmove(line)
                    # CRITICAL ORDERING: commit every field consumed by the next
                    # GUI command before publishing bestmove. Once emit() flushes,
                    # python-chess may immediately send position/go ponder.
                    with self.state_lock:
                        position = self.current_search_position_cmd
                        self.last_bestmove = best
                        self.last_ponder_move = ponder
                        if position and best and best != "0000":
                            try:
                                self.after_our_move_cmd = append_move_to_position(position, best)
                            except ValueError:
                                self.after_our_move_cmd = None
                        else:
                            self.after_our_move_cmd = None
                        if self.foreground_engine is e:
                            self.foreground_engine = None

                    self.log(
                        "relay_bestmove_committed_v4",
                        engine=e.label,
                        bestmove=best,
                        ponder=ponder,
                        position=position,
                        after_our_move=self.after_our_move_cmd,
                    )
                    self.emit(line)
                    break

                # Info/string output cannot trigger a new python-chess play command.
                self.emit(line)

        except Exception as exc:
            with self.state_lock:
                if self.foreground_engine is e:
                    self.foreground_engine = None
            self.log("relay_error_v4", error=repr(exc), engine=e.label)
            self.emit("bestmove 0000")


def main():
    return HybridProxyV4(build_v2_parser().parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
