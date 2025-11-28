# Unused/Deprecated Files List

This document lists all files that are currently not in use, deprecated, or can be safely removed.

---

## 🗑️ Files Safe to Delete

### Backup Files
These are backup files created during refactoring and are no longer needed:

1. **`Strategies/Rotation_ETF/services/backtester.py.backup_20251126_102809`**
   - **Type**: Backup file
   - **Created**: 2025-11-26
   - **Reason**: Backup during refactoring
   - **Status**: ✅ Safe to delete

2. **`Strategies/Rotation_Stocks/services/backtester.py.backup_20251126_102809`**
   - **Type**: Backup file
   - **Created**: 2025-11-26
   - **Reason**: Backup during refactoring
   - **Status**: ✅ Safe to delete

### Test/Fix Files
These are temporary files created to fix issues and are no longer needed:

3. **`test_etf_fix.py`**
   - **Type**: Test script
   - **Purpose**: Test ETF backtester initialization
   - **Status**: ✅ Safe to delete (issue fixed)

4. **`FIX_ETF_BACKTESTER.py`**
   - **Type**: Manual fix instructions
   - **Purpose**: Documentation for manual fix
   - **Status**: ✅ Safe to delete (issue fixed)

5. **`MANUAL_FIX_REQUIRED.txt`**
   - **Type**: Documentation file
   - **Purpose**: Manual fix instructions
   - **Status**: ✅ Safe to delete (issue fixed)

---

## ⚠️ Deprecated Code (Keep for Now)

### Database Models

6. **`ETFUnified` Model** (`Databases/market_data_db_connection.py`, lines 27-46)
   - **Status**: ⚠️ DEPRECATED
   - **Reason**: Replaced by `ETFData` model
   - **Action**: Kept for backward compatibility
   - **Recommendation**: Remove after confirming no dependencies

### Methods

7. **`get_nifty50_custom_stocks()`** (`Strategies/RS_ETF/rs_etf_backtester_core.py`, lines 406-421)
   - **Status**: ⚠️ DEPRECATED
   - **Reason**: Not used for ETF strategy (marked as deprecated in code)
   - **Action**: Can be removed (returns empty list)

---

## 🔮 Placeholder Code (Future Implementation)

### Derivatives Segment

8. **`DerivativesSegment` Class** (`Segments/DerivativesSegment.py`)
   - **Status**: 🔮 PLACEHOLDER
   - **Purpose**: Future F&O (Futures & Options) support
   - **Methods**:
     - `calculate_margin_requirements()`: Placeholder
     - `calculate_fo_costs()`: Placeholder
     - `handle_expiry()`: Placeholder
   - **Action**: Keep for future implementation

---

## 📝 Code Cleanup Opportunities

### Unused Imports
- Various files may have unused imports (can be cleaned with linter)
- Recommendation: Run `pylint` or `flake8` to identify

### Commented Code
- Some strategy files have commented-out code
- Recommendation: Remove if not needed, or document why kept

---

## 🧹 Cleanup Script

To safely remove unused files, you can use:

```bash
# Remove backup files
rm Strategies/Rotation_ETF/services/backtester.py.backup_20251126_102809
rm Strategies/Rotation_Stocks/services/backtester.py.backup_20251126_102809

# Remove test/fix files
rm test_etf_fix.py
rm FIX_ETF_BACKTESTER.py
rm MANUAL_FIX_REQUIRED.txt
```

---

## ✅ Verification Checklist

Before deleting files:

- [ ] Verify no imports reference these files
- [ ] Check git history for important changes
- [ ] Ensure backups are in version control
- [ ] Test application after deletion
- [ ] Update documentation if needed

---

**Last Updated**: 2025-01-XX  
**Review Frequency**: Quarterly

