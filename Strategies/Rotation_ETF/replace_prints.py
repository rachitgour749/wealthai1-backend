"""
Script to replace all print statements with logger calls in backtester.py
"""

import re
import sys

def categorize_print(line):
    """Determine which logger method to use based on print content"""
    line_lower = line.lower()
    
    # Error messages
    if '[error]' in line_lower or 'error' in line_lower and ('exception' in line_lower or 'failed' in line_lower or 'traceback' in line_lower):
        return 'error'
    
    # Debug messages
    if '[debug]' in line_lower:
        return 'debug'
    
    # Warning messages
    if '[warning]' in line_lower or '⚠️' in line or '❌' in line:
        return 'info'
    
    # Trade/execution messages
    if any(word in line_lower for word in ['buy', 'sell', 'sold', 'bought', 'trade', 'executed', 'transaction']):
        return 'trade'
    
    # Performance/metrics messages
    if any(word in line_lower for word in ['cagr', 'return', 'sharpe', 'drawdown', 'performance', 'profit', 'loss', 'nav']):
        return 'performance'
    
    # Progress messages
    if any(word in line_lower for word in ['loading', 'processing', 'calculating', 'running', 'starting', 'completed', '✅', '🔄', '📊', '📈', '📉', '💰', '🎯']):
        return 'progress'
    
    # Default to info
    return 'info'

def replace_print_with_logger(content):
    """Replace print statements with appropriate logger calls"""
    lines = content.split('\n')
    modified_lines = []
    
    for i, line in enumerate(lines):
        # Skip if line doesn't contain print(
        if 'print(' not in line:
            modified_lines.append(line)
            continue
        
        # Skip if it's already using logger
        if 'self.logger' in line:
            modified_lines.append(line)
            continue
        
        # Skip the Plotly warning (it's at module level, not in class)
        if 'Plotly not available' in line:
            modified_lines.append(line)
            continue
        
        # Extract indentation
        indent = len(line) - len(line.lstrip())
        indent_str = ' ' * indent
        
        # Determine logger method
        logger_method = categorize_print(line)
        
        # Handle different print patterns
        # Pattern 1: print(f"...")
        if match := re.search(r'print\(f"([^"]+)"\)', line):
            message = match.group(1)
            # Remove [DEBUG], [ERROR], [WARNING] prefixes
            message = re.sub(r'\[(DEBUG|ERROR|WARNING)\]\s*', '', message)
            new_line = f'{indent_str}self.logger.{logger_method}(f"{message}")'
            modified_lines.append(new_line)
        
        # Pattern 2: print("...")
        elif match := re.search(r'print\("([^"]+)"\)', line):
            message = match.group(1)
            # Remove [DEBUG], [ERROR], [WARNING] prefixes
            message = re.sub(r'\[(DEBUG|ERROR|WARNING)\]\s*', '', message)
            new_line = f'{indent_str}self.logger.{logger_method}("{message}")'
            modified_lines.append(new_line)
        
        # Pattern 3: print(variable) or complex expressions
        else:
            # Keep the original line as-is if it's too complex
            modified_lines.append(line)
    
    return '\n'.join(modified_lines)

if __name__ == '__main__':
    file_path = r'c:\Users\Lenovo\Desktop\WEALTH_AI_BACKEND\Strategies\Rotation_ETF\services\backtester.py'
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count before
    before_count = content.count('print(')
    
    # Replace
    new_content = replace_print_with_logger(content)
    
    # Count after
    after_count = new_content.count('print(')
    logger_count = new_content.count('self.logger')
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Replacement complete!")
    print(f"   Print statements before: {before_count}")
    print(f"   Print statements after: {after_count}")
    print(f"   Logger calls: {logger_count}")
    print(f"   Replaced: {before_count - after_count} print statements")
