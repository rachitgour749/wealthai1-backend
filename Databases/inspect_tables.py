"""
Script to inspect the plan_master, plan_prod_mapping, and prod_master tables
and understand their relationships
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neon_db_connection import create_connection, get_engine
from sqlalchemy import text, inspect

def inspect_table_schema(table_name):
    """Inspect a table's schema"""
    engine = get_engine()
    inspector = inspect(engine)
    
    print(f"\n{'='*70}")
    print(f"Table: {table_name}")
    print(f"{'='*70}")
    
    # Get columns
    columns = inspector.get_columns(table_name)
    print("\nColumns:")
    print("-" * 70)
    for col in columns:
        nullable = "NULL" if col['nullable'] else "NOT NULL"
        default = f" DEFAULT {col['default']}" if col['default'] is not None else ""
        print(f"  {col['name']:30} {str(col['type']):25} {nullable}{default}")
    
    # Get primary keys
    pk_constraint = inspector.get_pk_constraint(table_name)
    if pk_constraint and pk_constraint['constrained_columns']:
        print(f"\nPrimary Key: {', '.join(pk_constraint['constrained_columns'])}")
    
    # Get foreign keys
    fk_constraints = inspector.get_foreign_keys(table_name)
    if fk_constraints:
        print("\nForeign Keys:")
        for fk in fk_constraints:
            print(f"  {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
    
    # Get indexes
    indexes = inspector.get_indexes(table_name)
    if indexes:
        print("\nIndexes:")
        for idx in indexes:
            unique = "UNIQUE" if idx['unique'] else ""
            print(f"  {idx['name']} ({', '.join(idx['column_names'])}) {unique}")
    
    # Get sample data (first 5 rows)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 5"))
            rows = result.fetchall()
            if rows:
                print(f"\nSample Data (first 5 rows):")
                print("-" * 70)
                # Get column names
                col_names = [col['name'] for col in columns]
                # Print all columns
                print(f"  {' | '.join(f'{name[:15]}' for name in col_names)}")
                print("-" * 70)
                for row in rows:
                    print(f"  {' | '.join(str(val)[:15] if val is not None else 'NULL' for val in row)}")
    except Exception as e:
        print(f"\nCould not fetch sample data: {e}")

def analyze_relationships():
    """Analyze relationships between tables"""
    engine = get_engine()
    inspector = inspect(engine)
    
    print(f"\n{'='*70}")
    print("RELATIONSHIP ANALYSIS")
    print(f"{'='*70}")
    
    # Check plan_prod_mapping foreign keys
    fk_constraints = inspector.get_foreign_keys('plan_prod_mapping')
    
    print("\nplan_prod_mapping table relationships:")
    print("-" * 70)
    
    plan_master_refs = []
    prod_master_refs = []
    
    for fk in fk_constraints:
        constrained_cols = ', '.join(fk['constrained_columns'])
        referred_table = fk['referred_table']
        referred_cols = ', '.join(fk['referred_columns'])
        
        print(f"  {constrained_cols} -> {referred_table}.{referred_cols}")
        
        if referred_table == 'plan_master':
            plan_master_refs.append((constrained_cols, referred_cols))
        elif referred_table == 'prod_master':
            prod_master_refs.append((constrained_cols, referred_cols))
    
    print(f"\nRelationship Summary:")
    print(f"  - plan_prod_mapping references plan_master: {len(plan_master_refs)} foreign key(s)")
    print(f"  - plan_prod_mapping references prod_master: {len(prod_master_refs)} foreign key(s)")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("INSPECTING PLAN MANAGEMENT TABLES")
    print("="*70)
    
    # Initialize connection
    if not create_connection():
        print("Failed to connect to database!")
        sys.exit(1)
    
    # Inspect each table
    tables = ['plan_master', 'prod_master', 'plan_prod_mapping']
    
    for table in tables:
        try:
            inspect_table_schema(table)
        except Exception as e:
            print(f"\nError inspecting {table}: {e}")
    
    # Analyze relationships
    try:
        analyze_relationships()
    except Exception as e:
        print(f"\nError analyzing relationships: {e}")
    
    print("\n" + "="*70)

