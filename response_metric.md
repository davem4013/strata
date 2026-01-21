# STRATA Response Metric (Canonical Definition)

**Primary metric:** `response = microprice - mid`. This captures how the top-of-book tilts in reaction to geometric distortion (residual) without reusing residual magnitude itself.

## Exact formula

Given best bid/ask prices (`bid`, `ask`) and their displayed sizes (`bid_size`, `ask_size`):

```
mid        = (bid + ask) / 2
microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
response   = microprice - mid
```

- Requires `bid_size > 0` and `ask_size > 0`. If either side is missing/zero, the response is undefined for that tick (defer to upstream handling).
- Uses only book geometry; no residual terms are included, keeping the metric independent of residual level.

## Units

- Price units of the underlying (e.g., USD) or equivalently ticks. No scaling or normalization is applied.

## Sign convention

- Positive `response`: bid-heavy book (microprice above mid) ⇒ upward pressure toward higher prices.
- Negative `response`: ask-heavy book (microprice below mid) ⇒ downward pressure toward lower prices.
- Near zero: balanced book / neutral pressure.

## Expected behavior by regime

- **Calm regimes:** tight spreads; `|response|` small and mean-reverting around zero. Brief flickers reflect microstructure noise more than intent; autocorrelation low.
- **Stressed regimes:** spreads widen or one side thins; `|response|` can spike and persist as liquidity pulls; sustained sign indicates directional pressure (inventory shedding or urgent demand). Magnitude often scales with spread width; expect higher variance and occasional jumps as quotes refresh.
