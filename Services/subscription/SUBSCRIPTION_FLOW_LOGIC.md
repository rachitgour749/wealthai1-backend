# Complete Subscription Management Flow & Logic

## Overview
This document describes the complete flow for subscription management with product-level activation using `plan_master`, `plan_prod_mapping`, `prod_master`, `subscription`, and `product_subscriptions` tables.

---

## Database Schema Relationships

### Tables Involved:
1. **`plan_master`** - Master plan table
   - `plan_code` (PK)
   - `plan_name`
   - `extension_days`
   - `description`

2. **`plan_prod_mapping`** - Maps plans to products
   - `plan_code` (FK → plan_master.plan_code)
   - `product_id` (FK → prod_master.product_id)
   - Links which products are included in each plan

3. **`prod_master`** - Master product table
   - `product_id` (PK)
   - `product_code` (e.g., "MARKETAI", "CHATAI", "TRADAI", "AUTOMATIONAI")
   - `product_name`
   - Other product details

4. **`subscription`** - User's main subscription record
   - `user_email` (PK)
   - `user_name`
   - `plan_code` (FK → plan_master.plan_code)
   - `subscription_start_date`
   - `subscription_end_date`
   - `is_trial` (Boolean)
   - `created_at`, `updated_at`

5. **`product_subscriptions`** - Individual product subscriptions per user
   - `id` (PK)
   - `user_email` (FK → subscription.user_email)
   - `product_code` (Enum: MARKETAI, CHATAI, TRADAI, AUTOMATIONAI)
   - `subscription_type` (TRIAL, PAID, BUNDLE)
   - `status` (TRIAL, ACTIVE, EXPIRED, CANCELLED)
   - `plan_code` (String - from subscription.plan_code)
   - `trial_start_date`, `trial_end_date`
   - `paid_start_date`, `paid_end_date`
   - `trial_duration_days`
   - `is_bundle_subscription` (Boolean)
   - `chatai_key` (for CHATAI product)
   - `total_tokens`, `used_tokens` (for CHATAI)
   - `created_at`, `updated_at`

---

## Flow 1: Trial Activation (`/api/subscription/activate-product-trial`)

### Request:
```json
{
  "user_email": "user@example.com",
  "user_name": "User Name",
  "plan_code": 8  // Trial bundle plan
}
```

### Complete Flow Logic:

#### Step 1: Validate Plan Code
- Query `plan_master` table to validate `plan_code` exists
- Retrieve: `plan_name`, `extension_days`, `description`
- If invalid → Return error

#### Step 2: Get Products for Plan Code
- Query `plan_prod_mapping` table with `plan_code`
- Join with `prod_master` to get `product_code` values
- Example for plan_code 8:
  ```sql
  SELECT pm.product_code 
  FROM plan_prod_mapping ppm
  JOIN prod_master pm ON ppm.product_id = pm.product_id
  WHERE ppm.plan_code = 8
  ```
- Result: List of product codes (e.g., ["TRADAI", "MARKETAI", "AUTOMATIONAI"])

#### Step 3: Update/Create Main Subscription
- Check if `subscription` record exists for `user_email`
- **If EXISTS:**
  - Update `plan_code` = new plan_code
  - Update `user_name` = new user_name
  - Set `subscription_start_date` = NOW()
  - Set `subscription_end_date` = NOW() + `extension_days` (from plan_master)
  - Set `is_trial` = TRUE
  - Update `updated_at` = NOW()
  
- **If NOT EXISTS:**
  - Create new `subscription` record
  - Set all fields as above

#### Step 4: Create/Update Product Subscriptions
For each product_code from Step 2:

- Check if `product_subscriptions` record exists for (`user_email`, `product_code`)
  
- **If EXISTS:**
  - Update existing record:
    - `plan_code` = subscription.plan_code (as string)
    - `subscription_type` = TRIAL
    - `status` = TRIAL
    - `trial_start_date` = NOW()
    - `trial_end_date` = NOW() + `extension_days`
    - `trial_duration_days` = `extension_days`
    - `is_bundle_subscription` = TRUE (if multiple products)
    - `updated_at` = NOW()
    - For CHATAI: Generate/update `chatai_key` if needed
  
- **If NOT EXISTS:**
  - Create new `product_subscriptions` record:
    - `id` = `ps_{user_email}_{product_code}_{timestamp}`
    - `user_email` = user_email (lowercase)
    - `product_code` = product_code enum value
    - `subscription_type` = TRIAL
    - `status` = TRIAL
    - `plan_code` = subscription.plan_code (as string)
    - `trial_start_date` = NOW()
    - `trial_end_date` = NOW() + `extension_days`
    - `trial_duration_days` = `extension_days`
    - `is_bundle_subscription` = TRUE (if plan has multiple products)
    - `chatai_key` = Generated if product_code == CHATAI
    - `total_tokens` = 0, `used_tokens` = 0
    - `created_at` = NOW(), `updated_at` = NOW()

#### Step 5: Return Response
```json
{
  "user_email": "user@example.com",
  "startDate": "2024-01-01T00:00:00Z",
  "endDate": "2024-01-08T00:00:00Z",
  "planCode": 8,
  "planName": "All Products Mega Bundle Trial"
}
```

---

## Flow 2: Payment-Based Product Activation

### Scenario: User makes payment for plan_code 3 (ChatAI1 only)

### Request (from Payment Service):
```json
{
  "user_email": "rs3501194@gmail.com",
  "plan_code": 3,
  "payment_id": "pay_123456",
  "payment_status": "success"
}
```

### Complete Flow Logic:

#### Step 1: Validate Plan Code
- Query `plan_master` to validate `plan_code = 3`
- Retrieve: `plan_name` = "ChatAI1", `extension_days` = 365

#### Step 2: Get Products for Plan Code
- Query `plan_prod_mapping` + `prod_master` for `plan_code = 3`
- Result: `["CHATAI"]` (only ChatAI product)

#### Step 3: Update Main Subscription
- Update `subscription` table:
  - `plan_code` = 3
  - `subscription_start_date` = NOW()
  - `subscription_end_date` = NOW() + 365 days
  - `is_trial` = FALSE (paid subscription)
  - `updated_at` = NOW()

#### Step 4: Update Product Subscriptions
**CRITICAL LOGIC:**
- For products IN the plan (CHATAI):
  - Check if `product_subscriptions` exists for (`user_email`, "CHATAI")
  - **If EXISTS:**
    - Update to PAID:
      - `subscription_type` = PAID
      - `status` = ACTIVE
      - `plan_code` = "3"
      - `paid_start_date` = NOW()
      - `paid_end_date` = NOW() + 365 days
      - `payment_id` = payment_id
      - `payment_status` = "success"
      - Keep `trial_start_date`, `trial_end_date` (historical data)
      - `updated_at` = NOW()
  - **If NOT EXISTS:**
    - Create new record with PAID subscription details

- For products NOT IN the plan (TRADAI, MARKETAI, AUTOMATI  ONAI):
  - Check existing `product_subscriptions` for these products
  - **If EXISTS and status is TRIAL:**
    - Keep trial record (don't delete)
    - Trial will expire naturally based on `trial_end_date`
  - **If EXISTS and status is ACTIVE (from previous paid plan):**
    - Update status based on expiry:
      - If `paid_end_date` < NOW() → Set `status` = EXPIRED
      - If `paid_end_date` >= NOW() → Keep ACTIVE (user has valid paid access)
  - **If NOT EXISTS:**
    - Do nothing (user never had access to this product)

#### Step 5: Result
- User `rs3501194@gmail.com` now has:
  - `subscription.plan_code` = 3
  - `product_subscriptions` for CHATAI: ACTIVE (paid, 365 days)
  - `product_subscriptions` for other products: EXPIRED or TRIAL (if trial not expired)

---

## Key Business Rules

### 1. Plan Code to Products Mapping
- Always query `plan_prod_mapping` + `prod_master` to get products for a plan
- Never hardcode product mappings
- Support dynamic plan configurations

### 2. Product Subscription Status Priority
1. **PAID ACTIVE** - Highest priority (paid subscription active)
2. **TRIAL ACTIVE** - Second priority (trial not expired)
3. **EXPIRED** - No access
4. **CANCELLED** - No access

### 3. Multiple Product Subscriptions
- User can have multiple `product_subscriptions` records (one per product)
- Each product subscription is independent
- Bundle subscriptions share same `plan_code` but separate records

### 4. Trial vs Paid Transition
- When user pays for a plan:
  - Products IN the plan → Upgrade to PAID
  - Products NOT IN the plan → Keep existing status (don't auto-expire unless already expired)
  - Trial records are preserved for historical tracking

### 5. Plan Code Updates
- When `subscription.plan_code` changes:
  - Update all `product_subscriptions.plan_code` for products in NEW plan
  - Products not in new plan keep their existing `plan_code` (historical reference)

---

## Database Query Examples

### Get Products for Plan Code:
```sql
SELECT 
    pm.product_code,
    pm.product_name,
    pm.product_id
FROM plan_prod_mapping ppm
INNER JOIN prod_master pm ON ppm.product_id = pm.product_id
WHERE ppm.plan_code = :plan_code
```

### Get All User Product Subscriptions:
```sql
SELECT 
    ps.*,
    s.plan_code as subscription_plan_code
FROM product_subscriptions ps
LEFT JOIN subscription s ON ps.user_email = s.user_email
WHERE ps.user_email = :user_email
```

### Check Product Access:
```sql
SELECT 
    ps.*,
    CASE 
        WHEN ps.status = 'ACTIVE' AND ps.paid_end_date > NOW() THEN 'paid_active'
        WHEN ps.status = 'TRIAL' AND ps.trial_end_date > NOW() THEN 'trial_active'
        ELSE 'no_access'
    END as access_type
FROM product_subscriptions ps
WHERE ps.user_email = :user_email
  AND ps.product_code = :product_code
```

---

## Error Handling

### Invalid Plan Code:
- Return: `400 Bad Request` - "Plan code {plan_code} does not exist in plan_master"

### Database Errors:
- Rollback all transactions
- Return: `500 Internal Server Error` - "Failed to activate product trial: {error}"

### Missing Products in Plan:
- If `plan_prod_mapping` returns no products for a plan_code:
  - Log warning
  - Still update `subscription` table
  - Create no `product_subscriptions` records
  - Return success (user has subscription but no product access)

---

## Transaction Management

### Critical: Use Database Transactions
- All operations (subscription update + product_subscriptions create/update) must be in ONE transaction
- If any step fails → Rollback all changes
- Ensure data consistency

### Transaction Flow:
```
BEGIN TRANSACTION
  → Validate plan_code
  → Get products for plan
  → Update/Create subscription
  → For each product:
      → Create/Update product_subscription
COMMIT TRANSACTION
```

---

## Summary

1. **Trial Activation**: Updates `subscription` + creates/updates `product_subscriptions` for all products in plan
2. **Payment Activation**: Updates `subscription` + upgrades matching `product_subscriptions` to PAID, leaves others unchanged
3. **Product Access**: Always check `product_subscriptions` table, not just `subscription` table
4. **Plan Mapping**: Always query `plan_prod_mapping` + `prod_master` dynamically
5. **Status Priority**: PAID > TRIAL > EXPIRED

