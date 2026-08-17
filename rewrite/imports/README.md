# Leviathan Imported Donor Material

This directory is the only approved landing zone for source/model/data artifacts copied or adapted from external donors.

## Layout

```text
rewrite/imports/<donor-id>/<artifact>
rewrite/imports/<donor-id>/<artifact>.provenance.json
```

The `<donor-id>` must exist in `rewrite/donors/DONOR_REGISTRY.json`.

## Required provenance sidecar

```json
{
  "donor_id": "caissa",
  "source_repo": "Witek902/Caissa",
  "source_revision": "<immutable tag or commit SHA>",
  "source_path": "src/example.cpp",
  "license": "MIT",
  "reuse_mode": "adapted",
  "leviathan_owner": "evaluation",
  "import_reason": "What capability this artifact uniquely contributes",
  "tests_required": ["unit", "A/B", "fixed-node"],
  "version_gate_verified": true,
  "changes": "Summary of Leviathan modifications",
  "attribution": "Copyright/license notice required by upstream"
}
```

`reuse_mode` is one of `copied`, `adapted`, `reimplemented`, `model`, `dataset`, or `tool`.

## Policy

- Ideas may be studied from every registered donor.
- GPLv3/GPLv3-or-later and MIT source may be directly reused under the project policy, with required notices and corresponding-source obligations.
- Current AGPL donors are **reference-only** unless Leviathan explicitly changes its licensing/distribution policy.
- Version-gated donors such as pre-20 Viridithas must pin a qualifying immutable revision and set `version_gate_verified=true`.
- Network/model/data licenses are audited separately from engine source.
- Pawnocchio code is GPLv3, but its assets are CC-BY-ND-4.0; do not treat those assets as modifiable donor material.
- No imported mechanism is promoted merely because it is strong upstream. It must pass Leviathan's local correctness, A/B, interaction, and strength gates.

Run:

```bash
python3 rewrite/tools/audit_donors.py rewrite/donors/DONOR_REGISTRY.json rewrite/imports
```

The CI donor audit must remain green before imported material can merge.
