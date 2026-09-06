# Transported frozen recurrence: response to the real RTX FP16 failure

## Empirical negative result

The first real RTX 3060 run of the Frozen Bedrock feature lab successfully executed the untouched `Qwen/Qwen3-1.7B-Base` donor, then failed on the first raw two-pass frozen-band route with:

`FloatingPointError: Nonfinite bounded update`

The failure occurred before the output trust-region projection could make the update safe. The raw implementation fed the output of the final selected decoder band directly back into the first layer of that earlier band. On the real FP16 pretrained model, that state was sufficiently off the layer band's normal activation distribution to produce non-finite replay values.

This is a useful falsification: **raw off-manifold band loopback is not a safe default for Qwen3-1.7B FP16.** The donor itself was healthy.

## Architectural repair: manifold-constrained transport

Let

- `e` be the actual hidden state entering the donor layer band,
- `F` be the frozen pretrained layer band,
- `a = F(e)` be its ordinary donor output,
- `c_r` be the current recurrent output state.

Instead of evaluating `F(c_r)` directly, construct a re-entry state near the donor's observed input:

`z_r = project_near(e, c_r, rho_in)`

and then evaluate:

`u_r = F(z_r)`

The recurrent innovation is measured relative to the original band output:

`i_r = u_r - a`

and applied at the output under a second trust region:

`c_(r+1) = project_near(a, c_r + gain * i_r, rho_out)`

The input and output trust regions are independent. The implementation also caps re-entry pointwise relative to the donor input's observed maximum magnitude as an FP16 guard.

This is not a theorem that all possible frozen matrix products remain finite. Therefore every replay layer is checked. If any reused layer produces a non-finite tensor, the entire experimental recurrent route returns the untouched donor band output `a` and records `route_status = donor_fallback_nonfinite`.

## Why this is logically better than raw repetition

A pretrained layer has only been optimized on the distribution of states it receives from preceding layers. Feeding a much later state directly into an earlier layer creates a large, unjustified distribution shift. Transported recurrence instead probes the same nonlinear function in a bounded neighborhood of an input that actually occurred during the donor forward pass.

That does not prove improved reasoning. It makes the experiment better posed and numerically safer.

## Lab changes

The Windows feature lab now:

- uses `StableFrozenExecutor` for experimental recurrence;
- tests 2-pass and 4-pass transported recurrence;
- tests adaptive transported recurrence;
- retains a guarded anchored-difference comparison;
- warms each CUDA route before timing;
- records peak VRAM, re-entry radius and fallback count;
- catches failures per route so one experiment cannot terminate the full run.

The raw `FrozenExecutor` remains in the repository as a historical/control implementation. It is no longer the lab default.

## Evidence before the next RTX run

Local CPU suite after the repair: **151/151 tests pass**, including the original 97 project tests. New stability tests verify:

1. transported re-entry obeys the configured relative L2 radius on the test fixture;
2. transported recurrence changes computation without modifying donor tensors;
3. deliberately injected non-finite replay falls back to exact donor output;
4. prefix causality remains intact on the fixture.

These are mechanism tests. The repaired recurrence still requires the real RTX run before any claim about pretrained Qwen numerical stability, output quality or wall-clock cost.
