# CBS replication-fidelity check

- Detectors: divergence_reversal
- Match window: ±24.0h around entry_date
- Trades in corpus: **2**

**Match rate: 0% (0/2 assessed)**

## Per-trade results

| id | ticker | dir | entry_date | matched | detector | lead | gap |
|---|---|---|---|---|---|---|---|
| hype_long_2026_04_05 | HYPE | long | 2026-04-05 | ✗ | — | — | no_trigger_in_window |
| spx_long_2026_04_05 | SPX | long | 2026-04-05 | ✗ | — | — | no_trigger_in_window |

## Per-trade diagnosis

### hype_long_2026_04_05 — HYPE long on 2026-04-05

- CBS setup type: `consolidation_break_then_trend`
- Bars scanned: 1129
- Same-direction triggers in window: 0
- Opposite-direction triggers in window: 0
- **Gap:** no_trigger_in_window

### spx_long_2026_04_05 — SPX long on 2026-04-05

- CBS setup type: `v_reversal_then_consolidation_break`
- Bars scanned: 1129
- Same-direction triggers in window: 0
- Opposite-direction triggers in window: 0
- **Gap:** no_trigger_in_window
