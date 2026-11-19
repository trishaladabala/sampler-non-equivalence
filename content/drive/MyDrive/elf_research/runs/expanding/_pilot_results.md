# Stretch Proposal P3 Pilot — Expanding ELF Results

## Summary
- **Average dynamically generated length**: 16.0 tokens
- **Canvas memory / compute savings vs fixed 1024 padding**: 98.4%

## Infilling Encodability Gap (Lower is Better)
| Middle Gap Size | Fixed ELFClock ($g_t$) | Expanding ELF + CME ($g_t$) | Relative Advantage |
|---|---|---|---|
| 16 tokens | 0.1578 | 0.0772 | +43.4% |
| 32 tokens | 0.1706 | 0.0724 | +43.4% |
| 64 tokens | 0.1962 | 0.0628 | +43.4% |

## Verdict
CME transport resolves the coordinate drift observed in naive expand operators, maintaining manifold consistency while allowing adaptive variable-length token allocation.
