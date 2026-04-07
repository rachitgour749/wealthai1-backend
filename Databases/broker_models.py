from sqlalchemy import Column, Integer, String, DateTime
from Databases.app_data_db_connection import Base
from datetime import datetime

class BrokerSession(Base):
    __tablename__ = "broker_sessions"
    
    user_email = Column(String, primary_key=True, index=True)
    broker_name = Column(String, index=True)
    client_id = Column(String, index=True) # Broker Client ID
    api_key = Column(String)  # Broker API Key
    access_token = Column(String)
    token_expire = Column(DateTime)
    static_ip = Column(String, nullable=True)  # Whitelisted IP for algo trading
    static_ip_username = Column(String, nullable=True)
    static_ip_password = Column(String, nullable=True)
    static_ip_port = Column(String, nullable=True)
    broker_credentials = Column(String, nullable=True) # JSON store of user credentials
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<BrokerSession(email={self.user_email}, broker={self.broker_name})>"
