"""Database models for Webhook Security"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from Databases.app_data_db_connection import Base


class RAConfig(Base):
    """
    Configuration for Research Analysts (RA).
    Stores secret keys per strategy type for each RA.
    """
    __tablename__ = "ra_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ra_email = Column(String(255), nullable=True, index=True) # New column
    ra_code = Column(String(100), index=True, nullable=False)
    strategy_type = Column(String(100), nullable=False)
    secret_key = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    is_ip_check_enabled = Column(Boolean, default=True) # Legacy field
    allowed_ips = Column(Text, nullable=True)           # Legacy field
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<RAConfig(ra_code={self.ra_code}, strategy={self.strategy_type})>"

    @property
    def master_secret(self):
        """Legacy alias for secret_key"""
        return self.secret_key

# Backward compatibility alias
WebhookRA = RAConfig




class WebhookConf(Base):
    """
    Detailed configuration for webhooks.
    """
    __tablename__ = "webhook_conf"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    strategy_type = Column(String(255), nullable=True)
    run_id = Column(String(100), nullable=True, unique=True, index=True)
    client_info = Column(Text, nullable=True)  # Store as JSON string
    status = Column(String(50), default='active')
    category = Column(String(50), nullable=True) # FNO or EQUITY
    source = Column(String(50), nullable=True)   # INDIVIDUAL or RA
    ra_code = Column(String(255), nullable=True) # Renamed from reference
    name = Column(String(255), nullable=True)    # Strategy name from caller
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<WebhookConf(user_id={self.user_id}, run_id={self.run_id}, source={self.source})>"


class WebhookKey(Base):
    """
    User/RA secret keys for webhook authentication.
    """
    __tablename__ = "webhook_key"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    secret_key = Column(String(255), nullable=False, unique=True)
    source = Column(String(50), nullable=False) # RA or INDIVIDUAL
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<WebhookKey(user_id={self.user_id}, source={self.source})>"


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
