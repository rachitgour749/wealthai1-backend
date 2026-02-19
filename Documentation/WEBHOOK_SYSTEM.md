# WealthAI Backend v2 - Webhook System Documentation

This document details the workflow, APIs, and validation logic for the external webhook system, designed to execute trades from platforms like TradingView.

## 1. Overview

The Webhook System allows external signal providers (like TradingView) to trigger trades for authorized users. It integrates with:
-   **Subscription Service**: To validate user access.
-   **Strategy Manager**: To verify strategy status and configuration.
-   **Broker Manager**: To execute trades via broker APIs.
-   **Portfolio Service**: To record executed trades.

## 2. API Endpoints

### 2.1 Create External Strategy Configuration
**Endpoint:** `POST /api/webhook/create_webhook`

Used to register a new external strategy instance for a user.

**Request Payload:**
```json
{
  "user_id": "user@example.com",
  "strategy_type": "External_Strategy",
  "strategy_name": "My TradingView Strategy",
  "reference_capital": 100000,
  "client_info": {
    "CLIENT_ID_1": 50000,
    "CLIENT_ID_2": 50000
  },
  "webhook": "https://api.wealthai1.in/api/webhook/wealthai1.in/trade_execute" 
}
```

**Outcome:**
-   Creates a `SavedInstance` record.
-   Sets `status` to `'running'`.
-   Sets `source` to `'other'`.
-   Returns a unique `run_id` (e.g., `EXT_...`).

---

### 2.2 Execute Trade Webhook
**Endpoint:** `POST /api/webhook/wealthai1.in/trade_execute`

This is the main endpoint called by the external signal provider (e.g., TradingView alert).

**Request Payload:**
```json
{
  "signal_id": "SIG_UNIQUE_ID",
  "strategy_name": "My TradingView Strategy",
  "timestamp": "2026-02-19T10:15:30Z",
  "symbol": "PHARMABEES",
  "exchange": "NSE",
  "order_side": "BUY",
  "authorized_emails": [
    "user1@example.com",
    "user2@example.com"
  ],
  "clients": {
    "CLIENT_ID_1": "10", 
    "CLIENT_ID_2": "0" 
  }
}
```

## 3. Workflow & Validation Logic

When a request hits the `/trade_execute` endpoint, the system processes each email in `authorized_emails` sequentially:

### Step 1: Subscription Validation
*   **Check:** Queries `ProductManager` table for the user's email.
*   **Validation:**
    *   **Product Code**: Must be `'M'` (MarketAI1).
    *   **Expiry**: `subscription_end_date` must be strictly *greater* than current UTC time.
    *   **Status**: Must be `ACTIVE` or `TRIAL`.
*   **Failure:** Logs "Invalid product code", "Subscription expired", or "Unauthorized".

### Step 2: Strategy Validation
*   **Check:** Queries `saved_instances` table for:
    *   `user_id` == `email`
    *   `strategy_name` == `payload.strategy_name`
*   **Validation:**
    *   **Exists?**: If no record found -> Error: "No strategy found with name..."
    *   **Source Check**: Must be `source == 'other'`. If not -> Error: "Strategy found but source is '{source}', expected 'other'."
    *   **Status Check**: Must be `status == 'running'`. If not -> Error: "Strategy ... is currently '{status}', expected 'running'."
*   **Failure:** Skips the user with a specific error message.

### Step 3: Broker Session Check
*   **Check:** Retrieves active session from `broker_sessions` table.
*   **Validation:** valid `access_token` must exist.
*   **Failure:** Logs "Broker session inactive or missing".

### Step 4:  Quantity Calculation
Determines the quantity to trade for the specific client ID associated with the user's broker session.

1.  **Payload Check**:
    *   Looks up `payload.clients[client_id]`.
    *   If a positive integer is found, it is used.
2.  **Fallback Calculation** (Dynamic Sizing):
    *   If payload quantity is `0` or missing:
    *   Fetch `reference_capital` from the user's `SavedInstance` record.
    *   Fetch Real-Time Price for `payload.symbol` (via Yahoo Finance / `PriceService`).
    *   Calculate: `Quantity = INT( Reference Capital / Current Price )`.
*   **Failure:** If quantity is still <= 0 (e.g., capital is 0 or price fetch failed), the trade is skipped.

### Step 5: Order Execution
*   Places the order via the Broker API (e.g., Angel One, Zerodha) using `dispatch_place_order`.
*   **Order Type**: MARKET.
*   **Product Type**: DELIVERY (default).

### Step 6: Portfolio Recording
*   If the broker returns "success":
    *   Creates a record in `portfolio_trades` table.
    *   **Run ID**: Uses the `SavedInstance.run_id` (e.g., `EXT_...`) to link the trade to the strategy instance.
    *   **Price**: Uses execution price from broker or fallback to real-time price.
    *   **Costs**: Automatically calculates estimated Brokerage and Taxes.

## 4. Troubleshooting Common Errors

| Error Message | Cause | Solution |
| :--- | :--- | :--- |
| `Missing mandatory header: Authorization` | The endpoint requires a Bearer token but none provided. | **Fixed:** Auth is now optional for this webhook. |
| `can't compare offset-naive and offset-aware datetimes` | Database has timezone-aware dates, code used naive `utcnow()`. | **Fixed:** Code now uses `datetime.now(timezone.utc)`. |
| `No strategy found with name ...` | Name mismatch between Payload and `SavedInstance` in DB. | Ensure `strategy_name` in webhook matching exactly with DB record. |
| `Strategy found but source is 'internal'` | Strategy was created as an internal algo, not external webhook. | Update `saved_instances` set `source='other'` for this row. |
| `Strategy ... is currently 'stopped'` | User has stopped the strategy. | User must "Deploy" or "Start" the strategy from dashboard. |
| `No quantity defined ...` | Payload sent `0` and fallback calc failed (e.g., 0 capital). | Ensure `reference_capital` > 0 in strategy config OR send explicit quantity. |
