#!/usr/bin/env python3
"""Print Leviathan's current architecture embodiment and parameter-ecology status."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leviathan.architecture_maturity import GATE_ORDER, load_maturity_plan  # noqa: E402
from leviathan.cognitive_kernel import load_cognitive_kernel_spec  # noqa: E402
from leviathan.parameter_cells import load_parameter_cell_spec  # noqa: E402


SYMBOLS = {
    "passed": "PASS",
    "partial": "PART",
    "not_started": "----",
}


def main() -> int:
    maturity = load_maturity_plan()
    cells = load_parameter_cell_spec()
    kernel = load_cognitive_kernel_spec()

    print("LEVIATHAN ARCHITECTURE STATUS")
    print("=" * 96)
    print("Gate order: " + " -> ".join(GATE_ORDER))
    print()
    header = f"{'Layer':<6} {'Score':>5}  " + "  ".join(f"{gate[:5]:>5}" for gate in GATE_ORDER) + "  Name"
    print(header)
    print("-" * len(header))
    for layer in maturity.layers:
        gates = "  ".join(f"{SYMBOLS[layer.gates[gate].value]:>5}" for gate in GATE_ORDER)
        print(f"{layer.layer_id:<6} {layer.score:>5.1f}  {gates}  {layer.name}")

    print("\nDevelopment order:")
    print("  " + " -> ".join(maturity.build_order))

    print("\nPARAMETER ECOLOGY")
    print("=" * 96)
    roadmap = cells["roadmap"]
    for stage in roadmap["stages"]:
        entry = roadmap[f"stage_{stage}"]
        print(f"MoP-{stage}: {entry['name']:<34} {entry['meaning']}")

    invariant = cells["invariant"]
    live = cells["live_reference"]
    print("\nCell insertion invariant:")
    print(f"  one cognitive model    : {cells['single_cognitive_model']}")
    print(f"  independent agents     : {invariant['independent_agents']}")
    print(f"  original router kept   : {invariant['original_router_retained']}")
    print(f"  ancestral tile kept    : {invariant['pretrained_tile_computation_retained']}")
    print(f"  one final output       : {invariant['one_final_output']}")

    print("\nLive reference execution:")
    print(
        "  arbitrary ancestral cells : "
        f"{live['independent_route_executes_ancestral_swiglu_tiles']}"
    )
    print(
        "  peer communication        : "
        f"{live['peer_communication_runs_inside_packed_expert_forward']}"
    )
    print(
        "  disagreement recruitment  : "
        f"{live['recruited_cell_executes_ancestral_swiglu_tile']}"
    )
    print(
        "  post-recruit discussion   : "
        f"{live['recruited_cells_join_second_token_local_communication_round']}"
    )
    print(f"  ephemeral local state     : {live['local_state_is_ephemeral']}")

    print("\nBehavioral influence at insertion:")
    for key in (
        "independent_route_influence_at_insertion",
        "communication_influence_at_insertion",
        "recruitment_influence_at_insertion",
        "local_state_influence_at_insertion",
        "refinement_influence_at_insertion",
    ):
        print(f"  {key:<43} {live[key]}")
    print("  (all must remain 0.0 until the relevant migration gate earns control)")

    print("\nCOGNITIVE KERNEL")
    print("=" * 96)
    print(f"Canonical model : {kernel['canonical_model_id']}")
    print(f"Semantic models : {kernel['invariant']['semantic_model_count']}")
    print("Pipeline:")
    for index, stage in enumerate(kernel["pipeline"]["stages"], start=1):
        print(f"  {index}. {stage}")

    print("\nNo layer is 'achieved' until all five embodiment gates pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
