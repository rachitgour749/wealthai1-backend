# RS Strategy Stop Loss Flow Diagrams

## Daily Mode Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      DAILY STOP LOSS MODE                    │
│                    (daily_stop_loss_check: true)             │
└─────────────────────────────────────────────────────────────┘

Week Timeline:
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│  Monday  │ Tuesday  │Wednesday │ Thursday │  Friday  │  Monday  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

Monday:
  ├─ Check Stop Loss
  ├─ Execute SL exits immediately ⚡
  └─ Update positions

Tuesday:
  ├─ Check Stop Loss
  ├─ Execute SL exits immediately ⚡
  └─ Update positions

Wednesday:
  ├─ Check Stop Loss
  ├─ Execute SL exits immediately ⚡
  └─ Update positions

Thursday:
  ├─ Check Stop Loss
  ├─ Execute SL exits immediately ⚡
  └─ Update positions

Friday (Signal Day):
  ├─ Check Stop Loss
  ├─ Execute SL exits immediately ⚡
  ├─ Generate RS signals
  │   ├─ Calculate RS scores
  │   ├─ Rank stocks/ETFs
  │   ├─ Determine entries
  │   └─ Determine exits
  └─ Store signals for Monday

Next Monday (Execution Day):
  └─ Execute RS signals
      ├─ Sell exits
      └─ Buy entries

═══════════════════════════════════════════════════════════════

Characteristics:
✅ Immediate risk protection
✅ Stop losses execute same day
⚠️  More frequent trading
⚠️  Higher transaction costs
📊 Stop loss trades: Independent of RS signals
```

## Weekly Mode Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     WEEKLY STOP LOSS MODE                    │
│                   (daily_stop_loss_check: false)             │
└─────────────────────────────────────────────────────────────┘

Week Timeline:
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│  Monday  │ Tuesday  │Wednesday │ Thursday │  Friday  │  Monday  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

Monday:
  ├─ Check Stop Loss
  └─ Accumulate to weekly_stop_loss_exits[] 📝

Tuesday:
  ├─ Check Stop Loss
  └─ Accumulate to weekly_stop_loss_exits[] 📝

Wednesday:
  ├─ Check Stop Loss
  └─ Accumulate to weekly_stop_loss_exits[] 📝

Thursday:
  ├─ Check Stop Loss
  └─ Accumulate to weekly_stop_loss_exits[] 📝

Friday (Signal Day):
  ├─ Check Stop Loss
  ├─ Combine: weekly_stop_loss_exits[] + today's SL
  ├─ Generate RS signals
  │   ├─ Calculate RS scores
  │   ├─ Rank stocks/ETFs
  │   ├─ Determine entries
  │   └─ Determine exits
  ├─ Merge: all_exits = RS_exits + SL_exits 🔀
  ├─ Store combined signals for Monday
  └─ Reset weekly_stop_loss_exits[] = []

Next Monday (Execution Day):
  └─ Execute ALL signals together
      ├─ Sell ALL exits (RS + Stop Loss)
      └─ Buy entries

═══════════════════════════════════════════════════════════════

Characteristics:
✅ Pure weekly rebalancing
✅ Fewer total trades
✅ Lower transaction costs
⚠️  Delayed stop loss execution
⚠️  Higher intra-week risk
📊 Stop loss trades: Combined with RS signals
```

## Configuration Loading Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Configuration Loading Priority                  │
└─────────────────────────────────────────────────────────────┘

Start Backtest
      │
      ├─ Check config_dict parameter
      │       │
      │       ├─ Has 'daily_stop_loss_check'? ──YES──> Use it (Priority 1) ✅
      │       │
      │       └─ NO
      │           │
      │           ├─ Load rs_config.json
      │           │       │
      │           │       ├─ File exists? ──YES──> Load value (Priority 2) ✅
      │           │       │
      │           │       └─ NO
      │           │           │
      │           │           └─ Use default: true (Priority 3) ✅
      │
      └─ Set self.daily_stop_loss_check
              │
              └─ Log: "Stop Loss Mode: Daily/Weekly Check"
```

## Stop Loss Check Logic

```
┌─────────────────────────────────────────────────────────────┐
│                  Stop Loss Check Decision                    │
└─────────────────────────────────────────────────────────────┘

For each trading day:
      │
      ├─ if daily_stop_loss_check == True:
      │       │
      │       ├─ Check all positions
      │       ├─ Find positions where: current_price <= stop_loss_price
      │       └─ Execute trades immediately
      │           └─ Reason: "Stop Loss (Daily)"
      │
      └─ else (Weekly mode):
              │
              ├─ if NOT Friday:
              │       │
              │       ├─ Check all positions
              │       ├─ Find positions where: current_price <= stop_loss_price
              │       └─ Accumulate to weekly_stop_loss_exits[]
              │
              └─ if Friday (Signal Day):
                      │
                      ├─ Check all positions
                      ├─ Find today's stop loss hits
                      ├─ Combine: all_SL = weekly_stop_loss_exits[] + today's
                      ├─ Generate RS signals → RS_exits
                      ├─ Merge: final_exits = RS_exits + all_SL
                      ├─ Store for Monday execution
                      └─ Reset weekly_stop_loss_exits[] = []
```

## Example Scenario

```
┌─────────────────────────────────────────────────────────────┐
│              Example: 3 Positions Hit Stop Loss              │
└─────────────────────────────────────────────────────────────┘

Portfolio:
  • RELIANCE (Stop Loss: ₹2,500)
  • TCS (Stop Loss: ₹3,800)
  • INFY (Stop Loss: ₹1,450)
  • HDFCBANK (Stop Loss: ₹1,650)
  • ICICIBANK (Stop Loss: ₹1,100)

═══════════════════════════════════════════════════════════════

DAILY MODE:
─────────────────────────────────────────────────────────────
Monday:
  RELIANCE drops to ₹2,480 → SELL immediately ⚡

Wednesday:
  TCS drops to ₹3,790 → SELL immediately ⚡

Friday:
  INFY drops to ₹1,440 → SELL immediately ⚡
  Generate RS signals:
    - Exit: HDFCBANK (RS rank dropped)
    - Entry: WIPRO (High RS rank)
  Store for Monday

Next Monday:
  Execute RS signals:
    - SELL HDFCBANK
    - BUY WIPRO

Total Trades: 5 (3 SL + 1 RS exit + 1 RS entry)

═══════════════════════════════════════════════════════════════

WEEKLY MODE:
─────────────────────────────────────────────────────────────
Monday:
  RELIANCE drops to ₹2,480 → Accumulate 📝

Wednesday:
  TCS drops to ₹3,790 → Accumulate 📝

Friday:
  INFY drops to ₹1,440 → Accumulate 📝
  Generate RS signals:
    - Exit: HDFCBANK (RS rank dropped)
  Combine exits:
    - SL exits: RELIANCE, TCS, INFY
    - RS exits: HDFCBANK
    - Total: 4 exits
    - Entry: WIPRO
  Store for Monday

Next Monday:
  Execute ALL signals together:
    - SELL RELIANCE (Stop Loss)
    - SELL TCS (Stop Loss)
    - SELL INFY (Stop Loss)
    - SELL HDFCBANK (RS signal)
    - BUY WIPRO (RS signal)

Total Trades: 5 (same total, but all on Monday)

═══════════════════════════════════════════════════════════════

Key Difference:
  Daily:  Trades spread across Mon, Wed, Fri, Mon (4 days)
  Weekly: All trades on Monday (1 day)
```

## Code Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Code Architecture                         │
└─────────────────────────────────────────────────────────────┘

Strategies/
│
├── RS/                                    ← Configuration Hub
│   ├── rs_config.json                    ← Main config file
│   ├── rs_config_loader.py               ← Loader utility
│   │   └── get_rs_config()               ← Singleton function
│   ├── README.md                         ← Documentation
│   ├── IMPLEMENTATION_SUMMARY.md         ← Tech details
│   └── QUICK_REFERENCE.md                ← Quick guide
│
├── RS_ETF/
│   └── rs_etf_backtester_core.py
│       ├── import get_rs_config()        ← Import config
│       ├── __init__()
│       │   ├── Load rs_config            ← Load on init
│       │   └── Set daily_stop_loss_check
│       └── run_backtest()
│           └── if daily_stop_loss_check: ← Conditional logic
│               ├── Daily mode
│               └── Weekly mode
│
└── RS_Stocks/
    └── rs_backtester_core.py
        ├── import get_rs_config()        ← Import config
        ├── __init__()
        │   ├── Load rs_config            ← Load on init
        │   └── Set daily_stop_loss_check
        └── run_backtest()
            └── if daily_stop_loss_check: ← Conditional logic
                ├── Daily mode
                └── Weekly mode
```

---

**Legend:**
- ⚡ Immediate execution
- 📝 Accumulate for later
- 🔀 Merge/Combine
- ✅ Advantage
- ⚠️  Consideration
- 📊 Note
