# Strategy Logging Configuration Guide

## Overview

The centralized logging system allows you to control which types of logs are printed for each strategy by editing simple JSON configuration files.

## Quick Start

### 1. Locate Your Strategy's Config File

Each strategy has its own `logging_config.json` file:
- `Strategies/Rotation_ETF/logging_config.json`
- `Strategies/Rotation_Stocks/logging_config.json`
- `Strategies/RS_ETF/logging_config.json`
- `Strategies/RS_Stocks/logging_config.json`
- `Strategies/SuperTrend/logging_config.json`
- `Strategies/customStrategy/logging_config.json`

### 2. Edit the Configuration

Open the config file and modify the settings:

```json
{
  "enabled": true,
  "categories": {
    "debug": false,
    "info": true,
    "progress": true,
    "error": true,
    "trade": true,
    "performance": true
  }
}
```

### 3. Control Log Categories

**Set `true` to enable, `false` to disable:**

- **`debug`**: Detailed debugging information (table queries, metadata loading, etc.)
- **`info`**: General informational messages (data loading, warnings)
- **`progress`**: Progress updates and status messages
- **`error`**: Error messages (**always enabled**, cannot be disabled)
- **`trade`**: Trade execution logs (buy/sell transactions)
- **`performance`**: Performance metrics and statistics

### 4. Global Enable/Disable

Set `"enabled": false` to disable ALL logs except errors:

```json
{
  "enabled": false,
  "categories": {
    ...
  }
}
```

## Examples

### Example 1: Quiet Mode (Errors Only)

```json
{
  "enabled": false,
  "categories": {
    "debug": false,
    "info": false,
    "progress": false,
    "error": true,
    "trade": false,
    "performance": false
  }
}
```

### Example 2: Trading Focus (Trades + Performance)

```json
{
  "enabled": true,
  "categories": {
    "debug": false,
    "info": false,
    "progress": false,
    "error": true,
    "trade": true,
    "performance": true
  }
}
```

### Example 3: Debug Mode (Everything)

```json
{
  "enabled": true,
  "categories": {
    "debug": true,
    "info": true,
    "progress": true,
    "error": true,
    "trade": true,
    "performance": true
  }
}
```

### Example 4: Production Mode (Info + Trades + Performance)

```json
{
  "enabled": true,
  "categories": {
    "debug": false,
    "info": true,
    "progress": true,
    "error": true,
    "trade": true,
    "performance": true
  }
}
```

## Notes

- **No code changes required**: Just edit the JSON file and restart your strategy
- **Error logs always print**: Critical errors will always be displayed regardless of settings
- **Per-strategy control**: Each strategy has independent logging configuration
- **Default behavior**: If config file is missing, all logs are enabled (backward compatible)

## Troubleshooting

**Q: Logs still printing after disabling?**
- Make sure you saved the JSON file
- Restart the strategy/application
- Check that the JSON syntax is valid (no trailing commas, proper quotes)

**Q: No logs at all?**
- Check that `"enabled": true` in the config
- Verify at least one category is set to `true`
- Error logs should always appear

**Q: JSON syntax error?**
- Use a JSON validator (jsonlint.com)
- Ensure all boolean values are lowercase (`true`/`false`, not `True`/`False`)
- No trailing commas after last item in objects/arrays
