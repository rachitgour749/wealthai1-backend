
import ast
import os

file_path = r"d:\WEALTHAI_V2\wealthai-backend-v2\Strategies\Rotation_ETF\services\backtester.py"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "fillna":
                print(f"Found fillna at line {node.lineno}")
                # Check keywords
                for keyword in node.keywords:
                    if keyword.arg == "method":
                        print(f"  !!! Found method='{keyword.value.s if isinstance(keyword.value, ast.Constant) else '?'}'")

except Exception as e:
    print(f"Error parsing file: {e}")
