# ADR 0004: Spec-Derived Expected-Mix Window vs. Fixed Tolerance Constant

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

During a normal `RollingUpdate`, a serving fleet intentionally contains a mix of old and new model versions while waves of replacement replicas pass their readiness gates. If an auditor alerts the instant any replica's hash differs from the newly repointed `production` alias, it cries wolf on every single deployment.

The naive fix is a hardcoded tolerance constant (for example, "ignore version mismatches for 15 steps"). A static constant fails in two directions:

1. If a rollout has a small wave capacity or slow readiness gate (`N=8, maxSurge=1, maxUnavailable=0, readiness_delay=2`, which legitimately requires 24 steps), a 15-step constant raises false alarms on an honest deploy.
2. If a fast rollout (`N=8, maxSurge=2, maxUnavailable=1, readiness_delay=1`, which completes in 6 steps) stalls at step 4, a 15-step constant leaves the fleet serving stale weights undetected for 9 extra steps.

## Decision

Compute the expected mixed-fleet window directly from the `RolloutSpec`:

$$\text{expected\_window}(N, S, U, d) = \left\lceil \frac{N}{S + U} \right\rceil \times (d + 1)$$

where $N$ is `fleet_size`, $S$ is `max_surge`, $U$ is `max_unavailable`, and $d$ is `readiness_delay_steps`. Furthermore, inspect `claimed_version` alongside `loaded_sha256`: any replica that already claims the new target version while holding old weight bytes is flagged `STALE` immediately on step 1 without waiting for the rollout window to expire.

## Consequences

- **Positive:** Zero false alarms across all honest rollouts (including slow rollouts with $d=2$), while the auditor detects the stalled 8-replica rollout at step 17 (the exact first step after its 16-step theoretical window closes).
- **Negative:** Requires the auditor to read the deployment's `RolloutSpec` parameters rather than operating as a spec-agnostic poller.
