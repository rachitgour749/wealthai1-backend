# Subscription System with 30-Day Trial

This subscription system provides automatic 30-day trial creation for Google-authenticated users without requiring payment integration.

## Features

- **Automatic Trial Creation**: 30-day free trial starts automatically when users login via Google OAuth
- **Subscription Management**: Track trial status, expiration, and user access
- **Route Protection**: Protect endpoints based on subscription status
- **Google OAuth Integration**: Seamless integration with Google authentication
- **No Payment Required**: Focus on trial management without payment complexity

## Architecture

```
subscription/
├── models.py              # Pydantic models for requests/responses
├── database.py            # Database models and operations
├── service.py             # Business logic layer
├── api.py                 # Subscription API endpoints
├── auth_integration.py    # Google OAuth integration
├── google_auth_api.py     # Google auth specific endpoints
├── middleware.py          # Route protection middleware
├── protected_endpoints.py # Example protected endpoints
└── README.md             # This file
```

## Database Schema

The system uses SQLite with the following subscription table:

```sql
CREATE TABLE subscriptions (
    id VARCHAR PRIMARY KEY,
    user_email VARCHAR UNIQUE NOT NULL,
    user_name VARCHAR,
    plan VARCHAR DEFAULT 'free',
    status VARCHAR DEFAULT 'trial',
    trial_start_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    trial_end_date DATETIME,
    subscription_start_date DATETIME,
    subscription_end_date DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE
);
```

## API Endpoints

### Google Authentication Integration

#### POST `/api/auth/google-login`
Handles Google OAuth login and automatically creates/checks subscription.

**Headers:**
```
Authorization: Bearer <google_oauth_token>
```

**Response:**
```json
{
  "success": true,
  "message": "Welcome! Your 30-day free trial has started.",
  "user": {
    "email": "user@example.com",
    "name": "John Doe"
  },
  "subscription": {
    "status": "trial",
    "plan": "free",
    "is_trial_active": true,
    "can_access_premium": true,
    "days_remaining": 30,
    "trial_end_date": "2024-02-15T10:30:00"
  },
  "trial_created": true
}
```

#### GET `/api/auth/user-info`
Get current user information and subscription status.

#### GET `/api/auth/check-access`
Check if user has access to premium features.

#### POST `/api/auth/start-trial`
Manually start a trial for the authenticated user.

### Subscription Management

#### POST `/api/subscription/create`
Create a new subscription with 30-day trial.

#### GET `/api/subscription/status`
Get subscription status for the authenticated user.

#### POST `/api/subscription/extend-trial`
Extend trial period for a user.

#### POST `/api/subscription/upgrade`
Upgrade user subscription plan.

#### GET `/api/subscription/analytics`
Get subscription analytics (admin endpoint).

### Protected Endpoints

#### GET `/api/protected/premium-feature`
Example endpoint requiring premium access (trial or paid).

#### GET `/api/protected/trial-only-feature`
Example endpoint requiring active trial.

#### GET `/api/protected/basic-feature`
Example endpoint requiring basic access (always available).

#### GET `/api/protected/user-dashboard`
User dashboard with subscription information.

## Usage Examples

### 1. Frontend Integration

```javascript
// After Google OAuth login
const response = await fetch('/api/auth/google-login', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${googleToken}`,
    'Content-Type': 'application/json'
  }
});

const data = await response.json();
if (data.success) {
  // User has trial access
  console.log(`Trial active: ${data.subscription.is_trial_active}`);
  console.log(`Days remaining: ${data.subscription.days_remaining}`);
}
```

### 2. Protecting Backend Endpoints

```python
from subscription.middleware import require_premium_access

@app.get("/api/premium-data")
async def get_premium_data(
    access_info: Dict[str, Any] = Depends(require_premium_access)
):
    # This endpoint requires premium access (trial or paid)
    user_email = access_info["user_email"]
    days_remaining = access_info["days_remaining"]
    
    return {
        "premium_data": "sensitive information",
        "user": user_email,
        "days_remaining": days_remaining
    }
```

### 3. Checking Access in Frontend

```javascript
// Check if user can access premium features
const checkAccess = async () => {
  const response = await fetch('/api/auth/check-access', {
    headers: {
      'Authorization': `Bearer ${googleToken}`
    }
  });
  
  const data = await response.json();
  if (data.has_access) {
    // Show premium features
    showPremiumContent();
  } else {
    // Show upgrade prompt
    showUpgradePrompt();
  }
};
```

## Subscription Statuses

- **TRIAL**: User is in 30-day trial period
- **ACTIVE**: User has active paid subscription
- **EXPIRED**: Trial or subscription has expired
- **CANCELLED**: Subscription was cancelled
- **SUSPENDED**: Subscription is suspended

## Subscription Plans

- **FREE**: Free plan with trial access
- **BASIC**: Basic paid plan
- **PREMIUM**: Premium paid plan
- **ENTERPRISE**: Enterprise plan

## Middleware Dependencies

### `require_premium_access`
Requires active trial or paid subscription.

### `require_trial_access`
Requires active trial specifically.

### `require_basic_access`
Always available (basic features).

### `get_user_with_subscription`
Gets user info with subscription details.

## Configuration

The system uses the existing `unified_etf_data.sqlite` database. No additional configuration is required.

## Trial Period Logic

1. **New User Login**: 30-day trial automatically created
2. **Existing User**: Returns current subscription status
3. **Trial Expiration**: Status automatically updated to EXPIRED
4. **Access Control**: Premium features blocked after trial expires

## Error Handling

The system provides detailed error messages for:
- Invalid authentication tokens
- Expired trials
- Missing subscriptions
- Access denied scenarios

## Security Considerations

- Google OAuth tokens should be properly verified in production
- Implement proper JWT token validation
- Add rate limiting for trial creation
- Log all subscription-related activities

## Monitoring and Analytics

The system provides analytics endpoints for:
- Total users and trial users
- Conversion rates
- Plan distribution
- Trial duration tracking

## Integration with Existing System

The subscription system integrates seamlessly with your existing backend:

1. **Database**: Uses the same SQLite database
2. **Authentication**: Works with Google OAuth flow
3. **API Structure**: Follows your existing FastAPI patterns
4. **Error Handling**: Consistent with your error handling approach

## Next Steps

1. **Production Setup**: Implement proper Google OAuth token verification
2. **Payment Integration**: Add payment processing when ready
3. **Email Notifications**: Add trial expiration notifications
4. **Admin Dashboard**: Create admin interface for subscription management
5. **Analytics Dashboard**: Build analytics visualization

