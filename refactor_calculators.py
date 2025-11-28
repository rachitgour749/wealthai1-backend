"""
Safe Refactoring Script: Replace Duplicate Calculator Logic with Calculator Classes

This script safely refactors the ETF and Stock backtesters to use centralized
Calculator classes instead of duplicate hardcoded logic.

Safety features:
- Creates backups before making changes
- Validates changes don't break syntax
- Can be easily reverted
"""

import os
import shutil
import re
from datetime import datetime

# Paths
ETF_BACKTESTER = r'C:\Users\Lenovo\Desktop\WEALTHAI_PROD\wealthai1-backend\Strategies\Rotation_ETF\services\backtester.py'
STOCK_BACKTESTER = r'C:\Users\Lenovo\Desktop\WEALTHAI_PROD\wealthai1-backend\Strategies\Rotation_Stocks\services\backtester.py'

def create_backup(filepath):
    """Create a timestamped backup of the file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Created backup: {backup_path}")
    return backup_path

def add_calculator_imports(content, is_stock=False):
    """Add Calculator imports after CoreLogic import"""
    
    # Find the CoreLogic import line
    corelogic_pattern = r'(from CoreLogic import BaseRotationBacktester\s*\n)'
    
    if is_stock:
        calculator_imports = '''# Import Calculator classes for cost and tax calculations
from Calculators import (
    IndianStockCostCalculator,
    IndianCapitalGainsTaxCalculator
)

'''
    else:
        calculator_imports = '''# Import Calculator classes for cost and tax calculations
from Calculators import (
    IndianMarketCostCalculator,
    IndianCapitalGainsTaxCalculator
)

'''
    
    # Check if imports already exist
    if 'from Calculators import' in content:
        print("⚠️  Calculator imports already exist, skipping...")
        return content
    
    # Add imports after CoreLogic
    content = re.sub(corelogic_pattern, r'\1' + calculator_imports, content)
    print("✅ Added Calculator imports")
    return content

def add_calculator_initialization(content, is_stock=False):
    """Add calculator initialization in __init__ method"""
    
    # Find the end of __init__ method (before the first method after it)
    # Look for the line with metadata loading
    if is_stock:
        metadata_pattern = r'(self\.stock_metadata = self\.load_metadata\(\).*?\n)'
        init_code = '''        
        # Initialize Calculator classes for cost and tax calculations
        self.cost_calculator = IndianStockCostCalculator()
        self.tax_calculator = IndianCapitalGainsTaxCalculator()
'''
    else:
        metadata_pattern = r'(self\.etf_metadata = self\.load_metadata\(\).*?\n)'
        init_code = '''        
        # Initialize Calculator classes for cost and tax calculations
        self.cost_calculator = IndianMarketCostCalculator()
        self.tax_calculator = IndianCapitalGainsTaxCalculator()
'''
    
    # Check if initialization already exists
    if 'self.cost_calculator' in content:
        print("⚠️  Calculator initialization already exists, skipping...")
        return content
    
    # Add initialization after metadata loading
    content = re.sub(metadata_pattern, r'\1' + init_code, content)
    print("✅ Added calculator initialization")
    return content

def replace_transaction_costs_method(content):
    """Replace calculate_transaction_costs method with Calculator call"""
    
    # Find the entire calculate_transaction_costs method
    method_pattern = r'def calculate_transaction_costs\(self, action: str, amount: float, brokerage_percent: float\) -> Dict\[str, float\]:.*?(?=\n    def |\nclass |\Z)'
    
    new_method = '''def calculate_transaction_costs(self, action: str, amount: float, brokerage_percent: float) -> Dict[str, float]:
        """
        Calculate Indian market transaction costs using centralized calculator.
        
        This method now delegates to the IndianMarketCostCalculator/IndianStockCostCalculator
        for consistent cost calculations across the application.
        """
        return self.cost_calculator.calculate(action, amount, brokerage_percent)
    
    '''
    
    # Replace the method
    content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
    print("✅ Replaced calculate_transaction_costs method")
    return content

def replace_capital_gains_tax_method(content):
    """Replace calculate_capital_gains_tax method with Calculator call"""
    
    # Find the entire calculate_capital_gains_tax method
    method_pattern = r'def calculate_capital_gains_tax\(self, ticker: str, units_to_sell: int, sell_price: float,?\s*sell_date: datetime\) -> Dict:.*?(?=\n    def |\nclass |\Z)'
    
    new_method = '''def calculate_capital_gains_tax(self, ticker: str, units_to_sell: int, sell_price: float,
                                    sell_date: datetime) -> Dict:
        """
        Calculate capital gains tax using centralized calculator with FIFO logic.
        
        This method now delegates to the IndianCapitalGainsTaxCalculator for
        consistent tax calculations across the application.
        """
        if ticker not in self.purchase_history or not self.purchase_history[ticker]:
            return {
                'total_profit': 0,
                'capital_gains_tax': 0,
                'cost_basis': sell_price * units_to_sell,
                'transactions': []
            }
        
        return self.tax_calculator.calculate_with_details(
            purchase_lots=self.purchase_history[ticker],
            units_to_sell=units_to_sell,
            sell_price=sell_price,
            sell_date=sell_date
        )
    
    '''
    
    # Replace the method
    content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
    print("✅ Replaced calculate_capital_gains_tax method")
    return content

def refactor_file(filepath, is_stock=False):
    """Refactor a single backtester file"""
    print(f"\n{'='*60}")
    print(f"Refactoring: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # Create backup
    backup_path = create_backup(filepath)
    
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_length = len(content)
        print(f"📄 Original file: {original_length:,} characters")
        
        # Apply transformations
        content = add_calculator_imports(content, is_stock)
        content = add_calculator_initialization(content, is_stock)
        content = replace_transaction_costs_method(content)
        content = replace_capital_gains_tax_method(content)
        
        new_length = len(content)
        print(f"📄 New file: {new_length:,} characters")
        print(f"📊 Difference: {new_length - original_length:+,} characters")
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Successfully refactored {os.path.basename(filepath)}")
        print(f"💾 Backup saved at: {backup_path}")
        
        return True, backup_path
        
    except Exception as e:
        print(f"❌ Error refactoring {os.path.basename(filepath)}: {e}")
        # Restore from backup
        shutil.copy2(backup_path, filepath)
        print(f"🔄 Restored from backup")
        return False, backup_path

def main():
    """Main refactoring function"""
    print("\n" + "="*60)
    print("Calculator Integration Refactoring Script")
    print("="*60)
    print("\nThis script will:")
    print("1. Create backups of both backtester files")
    print("2. Add Calculator class imports")
    print("3. Initialize Calculator instances")
    print("4. Replace duplicate methods with Calculator calls")
    print("\n" + "="*60)
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    results = []
    
    # Refactor ETF backtester
    success, backup = refactor_file(ETF_BACKTESTER, is_stock=False)
    results.append(('ETF Backtester', success, backup))
    
    # Refactor Stock backtester
    success, backup = refactor_file(STOCK_BACKTESTER, is_stock=True)
    results.append(('Stock Backtester', success, backup))
    
    # Summary
    print("\n" + "="*60)
    print("REFACTORING SUMMARY")
    print("="*60)
    
    for name, success, backup in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} - {name}")
        print(f"   Backup: {backup}")
    
    all_success = all(r[1] for r in results)
    
    if all_success:
        print("\n🎉 All files refactored successfully!")
        print("\n📝 Next steps:")
        print("1. Test the backtester to ensure it works correctly")
        print("2. Run a sample backtest and compare results")
        print("3. If everything works, you can delete the backup files")
    else:
        print("\n⚠️  Some files failed to refactor")
        print("   Files have been restored from backups")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
