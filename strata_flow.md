                    ┌──────────────┐
                    │  Streamers   │
                    └──────┬───────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼────────┐                   ┌────────▼─────────┐
│ Underlying Ring │                   │ Options Ring     │
│ Buffer          │                   │ Buffer           │
└───────┬─────────┘                   └────────┬─────────┘
        │                                      │
        └──────────────┬──────────────────────┘
                       │
                ┌──────▼───────┐
                │  Analytics   │
                │ (compute)    │
                └──────┬───────┘
                       │
        ┌──────────────┴──────────────┐
        │                              │
┌───────▼────────┐              ┌──────▼──────────┐
│ State Buffer   │              │ Streaming Plots │
│ (StateFrame)   │              │ (Dash / Plotly) │
└───────┬────────┘              └─────────────────┘
        │                              │
        │                              │  (Regime control /
        │                              │   interpretation params)
        │                              ▼
┌───────▼────────┐              ┌─────────────────┐
│ STRATA Buffer  │◀─────────────│ Regime Control  │
│ (basins,       │              │ (from UI)       │
│ residuals)     │              └─────────────────┘
└───────┬────────┘
        │
┌───────▼────────┐
│ React Plots    │
│ (structure)    │
└────────────────┘




                ┌──────────────────────┐
                │  Analytics (compute) │
                │──────────────────────│
                │  Observed Surface    │  ◄── Market data only
                │──────────┬───────────│
                │  Surface Fit          │◄── Fit parameters (from GUI)
                │──────────┬───────────│
                │  Residual Surface     │
                └──────────┬───────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼────────┐                     ┌──────▼──────────┐
│ State Buffer   │                     │ Streaming Plots │
│ (Surface +     │◄────────────────────│ (Dash / Plotly) │
│ Residuals)     │   visualization      │  + controls     │
└───────┬────────┘                     └─────────────────┘
        │                                      ▲
        │                                      │
        │                         (fit / view parameters)
        │                                      │
┌───────▼────────┐                     ┌──────┴──────────┐
│ STRATA Buffer  │◀────────────────────│ Regime / Lens   │
│ (basins,       │    interpretation    │ Control (UI)   │
│ residual geom) │                     └─────────────────┘

