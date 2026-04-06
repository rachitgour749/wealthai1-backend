# WealthAI Backend Structure (Post-Refactoring)

This document visualizes the folder structure after implementing the 4-layer architecture.

## 📂 wealthai1-backend

### 🧠 CoreLogic (The Foundation)
*Refactored to be hierarchical and modular.*
```text
CoreLogic/
├── WealthAIBase.py              # [NEW] Level 1: Universal Base Class
├── IndianExchange.py            # [NEW] Level 2: Indian Market Logic
├── Segments/                    # [NEW] Level 3: Asset Class Logic
│   ├── EquitySegment.py         # [NEW] Delivery Trading (Stocks/ETFs)
│   └── DerivativesSegment.py    # [NEW] F&O Trading (Placeholder)
└── ...
```

### ♟️ Strategies (The Implementation)
*Refactored to inherit from the new base classes.*
```text
Strategies/
├── Rotation/                    # [NEW] Rotation Strategy Base
│   └── RotationStrategy.py      # [NEW] Level 4: Base logic for Rotation
├── RS/                          # [NEW] RS Strategy Base
│   └── RSStrategy.py            # [NEW] Level 4: Base logic for RS
│
├── Rotation_ETF/                # Existing Strategy (Refactored)
│   └── services/
│       └── backtester.py        # Inherits from RotationStrategy
│
├── Rotation_Stocks/             # Existing Strategy (Refactored)
│   └── services/
│       └── backtester.py        # Inherits from RotationStrategy
│
├── RS_ETF/                      # Existing Strategy (Refactored)
│   └── rs_etf_backtester_core.py # Inherits from RSStrategy
│
└── RS_Stocks/                   # Existing Strategy (Refactored)
    └── rs_backtester_core.py    # Inherits from RSStrategy
```

### 🧮 Calculators (Integrated)
*Logic moved to `EquitySegment`, but files kept for backward compatibility or specific utility.*
```text
Calculators/
├── cost_calculator.py           # Logic integrated into EquitySegment
└── tax_calculator.py            # Logic integrated into EquitySegment
```

---
**Key Changes:**
1.  **New Layers:** `WealthAIBase` -> `IndianExchange` -> `EquitySegment`.
2.  **New Strategy Bases:** `RotationStrategy` and `RSStrategy` to reduce code duplication.
3.  **Clean Separation:** Core logic is no longer mixed with strategy logic.
