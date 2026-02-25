"""Database models for Webhook Security"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from Databases.app_data_db_connection import Base


class WebhookRA(Base):
    """
    Global webhook security configuration for Research Analysts (RA mode).
    Stores the master secret and IP whitelist used for RA (bulk) mode.
    There should only be ONE active row in this table.
    """
    __tablename__ = "webhook_ra"

    id = Column(Integer, primary_key=True, autoincrement=True)
    master_secret = Column(String(255), nullable=False)          # RA Master Secret
    allowed_ips = Column(Text, nullable=True)                    # Comma-separated IP whitelist, NULL = allow all
    is_ip_check_enabled = Column(Boolean, default=True)          # Toggle IP check on/off
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    def get_allowed_ips(self):
        """Parse allowed_ips string into a list"""
        if not self.allowed_ips:
            return []
        return [ip.strip() for ip in self.allowed_ips.split(",") if ip.strip()]

    def __repr__(self):
        return f"<WebhookRA(id={self.id}, ip_check={self.is_ip_check_enabled})>"


class WebhookIndividual(Base):
    """
    Per-user, per-strategy webhook secret keys (Individual User Mode).
    Each webhook strategy gets a unique key stored here.
    """
    __tablename__ = "webhook_individual"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String(255), nullable=False, index=True)  # Owner email
    run_id = Column(String(100), nullable=False, unique=True, index=True)  # Strategy Run ID
    strategy_name = Column(String(255), nullable=True)             # Strategy display name
    webhook_key = Column(String(255), nullable=False)              # Unique secret key
    webhook_type = Column(String(50), default='individual')        # 'ra' or 'individual'
    is_active = Column(Boolean, default=True)                      # Enable/disable key
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<WebhookIndividual(user={self.user_email}, run_id={self.run_id}, active={self.is_active})>"


class WebhookExecutionLog(Base):
    """
    Detailed log of every trade attempt (success, skip, error) for reporting.
    """
    __tablename__ = "webhook_execution_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=func.now(), index=True)
    user_email = Column(String(255), nullable=False, index=True)
    strategy_name = Column(String(255), nullable=False)
    symbol = Column(String(100), nullable=False)
    side = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)  # 'executed', 'failed', 'skipped', 'error', 'unauthorized'
    message = Column(Text, nullable=True)
    ra_email = Column(String(255), nullable=True, index=True)  # To group for RA reports
    run_id = Column(String(100), nullable=True)

    def __repr__(self):
        return f"<WebhookExecutionLog(user={self.user_email}, status={self.status}, symbol={self.symbol})>"
