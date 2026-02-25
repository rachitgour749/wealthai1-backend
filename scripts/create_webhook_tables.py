"""
Script to create webhook tables and seed initial webhook_config.
Run: python scripts/create_webhook_tables.py
"""
import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Databases.app_data_db_connection import create_connection, get_engine, Base
from Databases import webhook_models  # noqa: registers models
from Databases.webhook_models import WebhookConfig, WebhookKey
from sqlalchemy.orm import sessionmaker

print("=" * 60)
print("Webhook Table Setup")
print("=" * 60)

# Step 1: Connect to DB
print("\n[1] Connecting to database...")
if not create_connection():
    print("  FAILED - Cannot connect to database!")
    sys.exit(1)
print("  Connected OK")

engine = get_engine()
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# Step 2: Create tables
print("\n[2] Creating webhook_config and webhook_keys tables...")
Base.metadata.create_all(bind=engine, tables=[
    webhook_models.WebhookConfig.__table__,
    webhook_models.WebhookKey.__table__,
])
print("  Tables created (or already exist)")

# Step 3: Seed WebhookConfig if none exists
print("\n[3] Checking webhook_config seed data...")
existing = db.query(WebhookConfig).first()
if not existing:
    print("  No config found. Inserting default config...")
    seed = WebhookConfig(
        master_secret="wealthai-ra-master-secret-2026",
        allowed_ips="52.89.214.238,34.212.31.25,54.218.243.31,52.32.178.7,127.0.0.1",
        is_ip_check_enabled=False   # Start with False for easy testing
    )
    db.add(seed)
    db.commit()
    db.refresh(seed)
    print(f"  Seeded config with id={seed.id}")
else:
    print(f"  Config already exists (id={existing.id})")

# Step 4: Print current state of both tables
print("\n" + "=" * 60)
print("CURRENT STATE")
print("=" * 60)

print("\n[webhook_config table]")
configs = db.query(WebhookConfig).all()
if configs:
    for c in configs:
        print(f"  id              : {c.id}")
        print(f"  master_secret   : {c.master_secret}")
        print(f"  allowed_ips     : {c.allowed_ips}")
        print(f"  ip_check_enabled: {c.is_ip_check_enabled}")
        print(f"  created_at      : {c.created_at}")
else:
    print("  (no rows)")

print("\n[webhook_keys table]")
keys = db.query(WebhookKey).all()
if keys:
    for k in keys:
        print(f"  id            : {k.id}")
        print(f"  user_email    : {k.user_email}")
        print(f"  run_id        : {k.run_id}")
        print(f"  strategy_name : {k.strategy_name}")
        print(f"  webhook_type  : {k.webhook_type}")
        print(f"  is_active     : {k.is_active}")
        print(f"  webhook_key   : {k.webhook_key}")
        print(f"  created_at    : {k.created_at}")
        print()
else:
    print("  (no rows yet - will be populated when users create webhook strategies)")

db.close()
print("\n" + "=" * 60)
print("Setup Complete!")
print("=" * 60)
