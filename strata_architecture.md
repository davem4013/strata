# STRATA Architecture

> **STRATA is not an AI. STRATA is the world the AIs live in.**

This document formalizes the STRATA architecture as it exists today. It is intentionally opinionated and minimal. Its purpose is to **preserve architectural correctness** as the system grows.

---

## 1. What STRATA Is

STRATA is an in-memory state kernel for market reality.

It provides:

* A stable, append-only representation of market state
* A substrate that multiple cognitive engines can observe
* A separation between *truth*, *interpretation*, and *decision*

STRATA does **not**:

* make decisions
* execute trades
* infer strategy
* optimize outcomes

Those belong elsewhere.

---

## 2. STRATA as an OS Stack (Conceptual Model)

```
┌─────────────────────────────────────────────┐
│               Interfaces                    │
│                                             │
│  Dash / React / Plotly / CLI / Alerts       │
│  REST / WebSockets                           │
│                                             │
└─────────────────────────────────────────────┘
                     ▲
                     │ read-only
                     │
┌─────────────────────────────────────────────┐
│            Cognitive Engines                │
│                                             │
│  • Regime Classifier                         │
│  • Risk Engine                               │
│  • Vol Surface Interpreter                  │
│  • Agreement / Disagreement Models           │
│  • (future) LLM Supervisors                  │
│                                             │
│  Stateless or softly stateful                │
│  Never authoritative                         │
└─────────────────────────────────────────────┘
                     ▲
                     │ observe / append
                     │
┌─────────────────────────────────────────────┐
│          STRATA State Kernel (RAM)           │
│                                             │
│  Ring Buffers                                │
│  ├─ SurfaceStateBuffer                      │
│  ├─ StrataStateBuffer                       │
│  ├─ RegimeStateBuffer                       │
│                                             │
│  Immutable Snapshots                         │
│  Monotonic Time                              │
│  No Decisions                                │
│                                             │
└─────────────────────────────────────────────┘
                     ▲
                     │ ingest
                     │
┌─────────────────────────────────────────────┐
│          Data & Reality Layer                │
│                                             │
│  Market Feeds                                │
│  Synthetic Regimes                           │
│  Historical Replays                          │
│                                             │
└─────────────────────────────────────────────┘
```

**Rule:** Nothing above the kernel is allowed to pretend it *is* the kernel.

---

## 3. STRATA as a Dataflow Graph (Operational View)

This view explains how data moves through the system in real time.

```
 Market Stream
      │
      ▼
┌───────────────┐
│ Option Buffer │
└───────────────┘
      │
      ▼
┌──────────────────────┐
│ IV / Surface Builder │
└──────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│ SurfaceState (snapshot)      │
│  - timestamp                 │
│  - spot                      │
│  - iv_surface                │
│  - expiries / strikes        │
└──────────────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│ StrataState Committer        │
│  - residuals                 │
│  - basin geometry            │
│  - position                  │
│  - risk                      │
└──────────────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│ Regime Tracker               │
│  - stable / drifting / ...   │
└──────────────────────────────┘
```

**Important:**

* If any upstream state is missing, downstream APIs return **404**.
* This is correctness, not failure.

---

## 4. STRATA as a Memory Model (Most Important)

STRATA behaves like an operating system’s internal memory tables.

```
+--------------------------------------------------+
|                  RAM                             |
|                                                  |
|  [ SurfaceStateBuffer ]                          |
|    t0 → snapshot                                 |
|    t1 → snapshot                                 |
|    t2 → snapshot                                 |
|                                                  |
|  [ StrataStateBuffer ]                           |
|    t0 → basin, residuals, risk                   |
|    t1 → basin, residuals, risk                   |
|                                                  |
|  [ RegimeStateBuffer ]                           |
|    t0 → STABLE                                   |
|    t1 → TRANSITION                               |
|                                                  |
+--------------------------------------------------+
```

Properties:

* append-only
* bounded (ring buffers)
* monotonic timestamps
* inspectable
* forkable (future)

This is intentionally database-free.

---

## 5. Kernel Rules (Non‑Negotiable)

### STRATA kernel **may**:

* ingest raw or derived state
* store immutable snapshots
* expose read-only APIs

### STRATA kernel **must not**:

* infer strategy
* issue recommendations
* mutate past state
* execute trades

### Cognitive engines:

* may read STRATA
* may append derived state via committers
* must never overwrite kernel truth

### Interfaces:

* are strictly read-only
* must tolerate missing state
* must not compute analytics

---

## 6. Why This Exists

Markets are not driven by classical supply/demand.
They are driven by **liquidity, hedging pressure, and derivatives geometry**.

To reason about that honestly, the system needs:

* memory
* geometry
* temporal continuity

STRATA is the minimum structure that makes that possible.

---

## 7. Forward Compatibility

This architecture explicitly supports future additions:

* LLM supervisors (read-only)
* Multi-agent disagreement engines
* Copy-on-write state forks (simulation)
* Historical replay buffers
* Cross-asset regime correlation

All without breaking the kernel contract.

---

## 8. One Guiding Sentence

> **STRATA is the stable world. Everything else is a visitor.**

If a design decision violates that sentence, it is wrong.

---

*End of document.*

