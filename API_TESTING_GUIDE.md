# API Testing Guide for Notification Preferences

## Quick Start Testing

### Prerequisites
1. Backend server must be running on `http://localhost:8000`
2. Database migration completed (run `migrate_notification_columns.py`)

---

## Test 1: RS Strategy Deployment with Notifications

### Test Email Notification Enabled

```bash
curl -X POST http://localhost:8000/api/deployment/save-rs-deployment \
  -H "Content-Type: application/json" \
  -d "{
    \"user_email\": \"test@example.com\",
    \"strategy_name\": \"Test RS ETF Email\",
    \"webhook_url\": \"https://webhook.example.com\",
    \"client_information_json\": \"{}\",
    \"email_notification\": true,
    \"telegram_notification\": false
  }"
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Rs Etf Instance deployment saved successfully",
  "run_id": "RS_ETF_Test_RS_ETF_Email_1703345206"
}
```

---

### Test Telegram Notification Enabled

```bash
curl -X POST http://localhost:8000/api/deployment/save-rs-deployment \
  -H "Content-Type: application/json" \
  -d "{
    \"user_email\": \"test@example.com\",
    \"strategy_name\": \"Test RS ETF Telegram\",
    \"webhook_url\": \"https://webhook.example.com\",
    \"client_information_json\": \"{}\",
    \"email_notification\": false,
    \"telegram_notification\": true
  }"
```

---

### Test Both Notifications Enabled

```bash
curl -X POST http://localhost:8000/api/deployment/save-rs-deployment \
  -H "Content-Type: application/json" \
  -d "{
    \"user_email\": \"test@example.com\",
    \"strategy_name\": \"Test RS ETF Both\",
    \"webhook_url\": \"https://webhook.example.com\",
    \"client_information_json\": \"{}\",
    \"email_notification\": true,
    \"telegram_notification\": true
  }"
```

---

### Test Default Values (No Notification Fields)

```bash
curl -X POST http://localhost:8000/api/deployment/save-rs-deployment \
  -H "Content-Type: application/json" \
  -d "{
    \"user_email\": \"test@example.com\",
    \"strategy_name\": \"Test RS ETF Default\",
    \"webhook_url\": \"https://webhook.example.com\",
    \"client_information_json\": \"{}\"
  }"
```

**Expected:** Both notification fields should be `false` in database

---

## Test 2: Rotation ETF Strategy Deployment

### Test with Notifications

```bash
curl -X POST http://localhost:8000/api/deployment/live-signals/save-deployment \
  -H "Content-Type: application/json" \
  -d "{
    \"user_email\": \"test@example.com\",
    \"strategy_name\": \"Test Rotation ETF\",
    \"webhook_url\": \"https://webhook.example.com\",
    \"client_information_json\": \"{}\",
    \"email_notification\": true,
    \"telegram_notification\": false,
    \"reference_capital\": \"1000000\",
    \"strategy_type\": \"Rotation ETF\",
    \"etf_count\": 5,
    \"etf_names\": [\"NIFTYBEES\", \"JUNIORBEES\"]
  }"
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Etf Saved Strategy deployment saved successfully",
  "run_id": "run_etfs_rotation_strategy_Test_Rotation_ETF_1703345207",
  "execution_date": "2025-12-24",
  "last_execution_date": "N/A",
  "next_execution_date": "2025-12-24",
  "table": "etf_saved_strategy"
}
```

---

## Verify Database Values

### Using Python Script

Create `verify_notifications.py`:

```python
from Databases.app_data_db_connection import get_session
from sqlalchemy import text

session = get_session()

# Check RS ETF strategies
print("RS ETF Strategies:")
print("-" * 80)
result = session.execute(text("""
    SELECT strategy_name, email_notification, telegram_notification 
    FROM rs_etf_instance 
    WHERE strategy_name LIKE 'Test%'
    ORDER BY strategy_name
"""))

for row in result.fetchall():
    print(f"{row[0]:<30} Email: {row[1]:<6} Telegram: {row[2]}")

# Check Rotation ETF strategies
print("\nRotation ETF Strategies:")
print("-" * 80)
result = session.execute(text("""
    SELECT strategy_name, email_notification, telegram_notification 
    FROM etf_saved_strategy 
    WHERE strategy_name LIKE 'Test%'
    ORDER BY strategy_name
"""))

for row in result.fetchall():
    print(f"{row[0]:<30} Email: {row[1]:<6} Telegram: {row[2]}")

session.close()
```

Run: `python verify_notifications.py`

---

### Using SQL Query

```sql
-- Check RS ETF strategies
SELECT 
    strategy_name, 
    email_notification, 
    telegram_notification,
    status,
    run_id
FROM rs_etf_instance 
WHERE strategy_name LIKE 'Test%'
ORDER BY strategy_name;

-- Check Rotation ETF strategies
SELECT 
    strategy_name, 
    email_notification, 
    telegram_notification,
    status,
    run_id
FROM etf_saved_strategy 
WHERE strategy_name LIKE 'Test%'
ORDER BY strategy_name;
```

---

## Expected Test Results

| Strategy Name | Email | Telegram | Notes |
|--------------|-------|----------|-------|
| Test RS ETF Email | true | false | Email only |
| Test RS ETF Telegram | false | true | Telegram only |
| Test RS ETF Both | true | true | Both enabled |
| Test RS ETF Default | false | false | Default values |
| Test Rotation ETF | true | false | Rotation strategy |

---

## Cleanup Test Data

After testing, clean up test strategies:

```sql
-- Delete test RS strategies
DELETE FROM rs_etf_instance WHERE strategy_name LIKE 'Test%';

-- Delete test Rotation strategies
DELETE FROM etf_saved_strategy WHERE strategy_name LIKE 'Test%';
```

---

## Troubleshooting

### Issue: "Strategy not found"
**Cause:** Strategy must be saved first before deploying  
**Solution:** Use the save-strategy endpoint first, then deploy

### Issue: Columns not found in database
**Cause:** Migration not run  
**Solution:** Run `python migrate_notification_columns.py`

### Issue: Default values not working
**Cause:** Old database records  
**Solution:** Re-run migration or manually update existing records:
```sql
UPDATE etf_saved_strategy 
SET email_notification = false, telegram_notification = false 
WHERE email_notification IS NULL;
```
