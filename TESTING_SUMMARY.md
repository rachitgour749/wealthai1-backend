# Testing Summary for Notification Preferences

## ✅ Implementation Complete!

All notification preference columns have been successfully added to the database.

---

## What Was Done

### 1. Database Migration ✅
- Added `email_notification` and `telegram_notification` columns to `etf_saved_strategy`
- Added `email_notification` and `telegram_notification` columns to `rs_etf_instance`
- Both columns are BOOLEAN type with DEFAULT FALSE

### 2. Code Changes ✅
- Updated SQLAlchemy models
- Updated deployment endpoints to accept notification preferences
- Added backward compatibility (defaults to False)

---

## How to Test

### Step 1: Verify Database Columns

Run the verification script:
```bash
python test_notification_columns.py
```

**Expected Output:**
```
✅ Notification Columns Found:

Table                     Column                    Type            Default         Nullable
-----------------------------------------------------------------------------------------------
etf_saved_strategy        email_notification        boolean         false           NO
etf_saved_strategy        telegram_notification     boolean         false           NO
rs_etf_instance           email_notification        boolean         false           NO
rs_etf_instance           telegram_notification     boolean         false           NO

✅ etf_saved_strategy: 2/2 columns found
✅ rs_etf_instance: 2/2 columns found

🎉 All notification columns successfully added!
```

---

### Step 2: Test API Endpoints

**Important:** Make sure your FastAPI server is running first!

```bash
# Start your server
python main.py
# OR
uvicorn main:app --reload
```

#### Test 1: RS Strategy with Email Notification

```bash
curl -X POST http://localhost:8000/api/deployment/save-rs-deployment \
  -H "Content-Type: application/json" \
  -d "{\"user_email\":\"test@example.com\",\"strategy_name\":\"Test Email Notif\",\"webhook_url\":\"https://webhook.example.com\",\"client_information_json\":\"{}\",\"email_notification\":true,\"telegram_notification\":false}"
```

#### Test 2: Rotation ETF with Telegram Notification

```bash
curl -X POST http://localhost:8000/api/deployment/live-signals/save-deployment \
  -H "Content-Type: application/json" \
  -d "{\"user_email\":\"test@example.com\",\"strategy_name\":\"Test Telegram Notif\",\"webhook_url\":\"https://webhook.example.com\",\"client_information_json\":\"{}\",\"email_notification\":false,\"telegram_notification\":true,\"reference_capital\":\"1000000\",\"strategy_type\":\"Rotation ETF\"}"
```

#### Test 3: Both Notifications Enabled

```bash
curl -X POST http://localhost:8000/api/deployment/save-rs-deployment \
  -H "Content-Type: application/json" \
  -d "{\"user_email\":\"test@example.com\",\"strategy_name\":\"Test Both Notif\",\"webhook_url\":\"https://webhook.example.com\",\"client_information_json\":\"{}\",\"email_notification\":true,\"telegram_notification\":true}"
```

---

### Step 3: Verify Database Values

Create a file `check_notifications.py`:

```python
from Databases.app_data_db_connection import get_session
from sqlalchemy import text

session = get_session()

print("\n" + "="*80)
print("NOTIFICATION PREFERENCES IN DATABASE")
print("="*80)

# Check RS strategies
print("\nRS ETF Strategies:")
print("-"*80)
result = session.execute(text("""
    SELECT strategy_name, email_notification, telegram_notification, status
    FROM rs_etf_instance 
    WHERE strategy_name LIKE 'Test%'
    ORDER BY strategy_name
"""))

for row in result.fetchall():
    email = "✅" if row[1] else "❌"
    telegram = "✅" if row[2] else "❌"
    print(f"{row[0]:<30} Email: {email}  Telegram: {telegram}  Status: {row[3]}")

# Check Rotation ETF strategies
print("\nRotation ETF Strategies:")
print("-"*80)
result = session.execute(text("""
    SELECT strategy_name, email_notification, telegram_notification, status
    FROM etf_saved_strategy 
    WHERE strategy_name LIKE 'Test%'
    ORDER BY strategy_name
"""))

for row in result.fetchall():
    email = "✅" if row[1] else "❌"
    telegram = "✅" if row[2] else "❌"
    print(f"{row[0]:<30} Email: {email}  Telegram: {telegram}  Status: {row[3]}")

session.close()
print("\n" + "="*80)
```

Run: `python check_notifications.py`

---

## Quick Test Commands

### All-in-One Test Script

Save as `quick_test.sh` (or run commands one by one):

```bash
# 1. Verify columns exist
echo "Step 1: Verifying database columns..."
python test_notification_columns.py

# 2. Test RS deployment
echo "\nStep 2: Testing RS deployment..."
curl -X POST http://localhost:8000/api/deployment/save-rs-deployment \
  -H "Content-Type: application/json" \
  -d '{"user_email":"test@example.com","strategy_name":"Quick Test RS","webhook_url":"https://webhook.example.com","client_information_json":"{}","email_notification":true,"telegram_notification":true}'

# 3. Test Rotation ETF deployment
echo "\nStep 3: Testing Rotation ETF deployment..."
curl -X POST http://localhost:8000/api/deployment/live-signals/save-deployment \
  -H "Content-Type: application/json" \
  -d '{"user_email":"test@example.com","strategy_name":"Quick Test ETF","webhook_url":"https://webhook.example.com","client_information_json":"{}","email_notification":true,"telegram_notification":false,"reference_capital":"1000000","strategy_type":"Rotation ETF"}'

# 4. Verify database
echo "\nStep 4: Checking database values..."
python check_notifications.py
```

---

## Expected Results

After running the tests, you should see:

### Database Verification
```
✅ etf_saved_strategy: 2/2 columns found
✅ rs_etf_instance: 2/2 columns found
🎉 All notification columns successfully added!
```

### API Responses
```json
{
  "success": true,
  "message": "...",
  "run_id": "..."
}
```

### Database Values
```
RS ETF Strategies:
--------------------------------------------------------------------------------
Quick Test RS                  Email: ✅  Telegram: ✅  Status: running

Rotation ETF Strategies:
--------------------------------------------------------------------------------
Quick Test ETF                 Email: ✅  Telegram: ❌  Status: running
```

---

## Cleanup Test Data

After testing, remove test strategies:

```sql
DELETE FROM rs_etf_instance WHERE strategy_name LIKE 'Test%' OR strategy_name LIKE 'Quick Test%';
DELETE FROM etf_saved_strategy WHERE strategy_name LIKE 'Test%' OR strategy_name LIKE 'Quick Test%';
```

---

## Next Steps

Now that notification preferences are stored, you can:

1. **Read notification preferences** when executing signals
2. **Send emails** when `email_notification = true`
3. **Send Telegram messages** when `telegram_notification = true`
4. **Update frontend** to include notification toggle switches

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Columns not found | Run `python migrate_notification_columns.py` |
| API returns 404 | Make sure FastAPI server is running |
| "Strategy not found" error | Create strategy first before deploying |
| Columns show NULL | Re-run `python add_etf_notification_columns.py` |

---

## Files Created for Testing

- `test_notification_columns.py` - Verify database columns
- `migrate_notification_columns.py` - Add columns to database
- `add_etf_notification_columns.py` - Manually add ETF table columns
- `check_notifications.py` - Verify notification values in database
- `API_TESTING_GUIDE.md` - Detailed API testing guide
