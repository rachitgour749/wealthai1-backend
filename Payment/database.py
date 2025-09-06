from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

from Payment.config import PaymentConfig
from Payment.models import PaymentStatus, PaymentMethod

Base = declarative_base()

class PaymentOrder(Base):
    """Database model for payment orders"""
    __tablename__ = "payment_orders"
    
    id = Column(String, primary_key=True)
    razorpay_order_id = Column(String, unique=True, nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String, default="INR")
    receipt = Column(String)
    status = Column(String, default="pending")
    customer_name = Column(String)
    customer_email = Column(String)
    customer_contact = Column(String)
    notes = Column(Text)  # JSON string
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class PaymentTransaction(Base):
    """Database model for payment transactions"""
    __tablename__ = "payment_transactions"
    
    id = Column(String, primary_key=True)
    order_id = Column(String, nullable=False)
    razorpay_payment_id = Column(String, unique=True, nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String, default="INR")
    status = Column(String, default="pending")
    method = Column(String)
    bank = Column(String)
    wallet = Column(String)
    card_id = Column(String)
    vpa = Column(String)  # UPI VPA
    email = Column(String)
    contact = Column(String)
    fee = Column(Integer)
    tax = Column(Integer)
    error_code = Column(String)
    error_description = Column(String)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class PaymentRefund(Base):
    """Database model for payment refunds"""
    __tablename__ = "payment_refunds"
    
    id = Column(String, primary_key=True)
    payment_id = Column(String, nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String, default="INR")
    status = Column(String, default="pending")
    speed = Column(String, default="normal")
    notes = Column(Text)  # JSON string
    receipt = Column(String)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class Customer(Base):
    """Database model for customers"""
    __tablename__ = "customers"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    contact = Column(String)
    address = Column(Text)
    city = Column(String)
    state = Column(String)
    zipcode = Column(String)
    country = Column(String, default="India")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class PaymentPlan(Base):
    """Database model for payment plans"""
    __tablename__ = "payment_plans"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    amount = Column(Integer, nullable=False)
    currency = Column(String, default="INR")
    interval = Column(String, default="month")
    interval_count = Column(Integer, default=1)
    trial_period_days = Column(Integer)
    notes = Column(Text)  # JSON string
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class DatabaseManager:
    """Database manager for payment operations"""
    
    def __init__(self):
        self.engine = create_engine(PaymentConfig.DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close database session"""
        session.close()
    
    def create_order(self, order_data: Dict[str, Any]) -> PaymentOrder:
        """Create a new payment order"""
        session = self.get_session()
        try:
            order = PaymentOrder(**order_data)
            session.add(order)
            session.commit()
            session.refresh(order)
            return order
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_order_by_id(self, order_id: str) -> Optional[PaymentOrder]:
        """Get order by ID"""
        session = self.get_session()
        try:
            return session.query(PaymentOrder).filter(PaymentOrder.id == order_id).first()
        finally:
            self.close_session(session)
    
    def get_order_by_razorpay_id(self, razorpay_order_id: str) -> Optional[PaymentOrder]:
        """Get order by Razorpay order ID"""
        session = self.get_session()
        try:
            return session.query(PaymentOrder).filter(
                PaymentOrder.razorpay_order_id == razorpay_order_id
            ).first()
        finally:
            self.close_session(session)
    
    def update_order_status(self, order_id: str, status: str) -> bool:
        """Update order status"""
        session = self.get_session()
        try:
            order = session.query(PaymentOrder).filter(PaymentOrder.id == order_id).first()
            if order:
                order.status = status
                order.updated_at = datetime.utcnow()
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def create_transaction(self, transaction_data: Dict[str, Any]) -> PaymentTransaction:
        """Create a new payment transaction"""
        session = self.get_session()
        try:
            transaction = PaymentTransaction(**transaction_data)
            session.add(transaction)
            session.commit()
            session.refresh(transaction)
            return transaction
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_transaction_by_payment_id(self, payment_id: str) -> Optional[PaymentTransaction]:
        """Get transaction by Razorpay payment ID"""
        session = self.get_session()
        try:
            return session.query(PaymentTransaction).filter(
                PaymentTransaction.razorpay_payment_id == payment_id
            ).first()
        finally:
            self.close_session(session)
    
    def update_transaction_status(self, payment_id: str, status: str) -> bool:
        """Update transaction status"""
        session = self.get_session()
        try:
            transaction = session.query(PaymentTransaction).filter(
                PaymentTransaction.razorpay_payment_id == payment_id
            ).first()
            if transaction:
                transaction.status = status
                transaction.updated_at = datetime.utcnow()
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def create_refund(self, refund_data: Dict[str, Any]) -> PaymentRefund:
        """Create a new refund record"""
        session = self.get_session()
        try:
            refund = PaymentRefund(**refund_data)
            session.add(refund)
            session.commit()
            session.refresh(refund)
            return refund
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_payment_history(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get payment history"""
        session = self.get_session()
        try:
            transactions = session.query(PaymentTransaction).order_by(
                PaymentTransaction.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            history = []
            for transaction in transactions:
                history.append({
                    "order_id": transaction.order_id,
                    "payment_id": transaction.razorpay_payment_id,
                    "amount": transaction.amount,
                    "currency": transaction.currency,
                    "status": transaction.status,
                    "method": transaction.method,
                    "created_at": transaction.created_at.isoformat(),
                    "updated_at": transaction.updated_at.isoformat(),
                    "customer_email": transaction.email,
                    "customer_name": transaction.contact
                })
            
            return history
        finally:
            self.close_session(session)
    
    def get_payment_analytics(self, period: str = "monthly") -> Dict[str, Any]:
        """Get payment analytics"""
        session = self.get_session()
        try:
            # Get total transactions
            total_transactions = session.query(PaymentTransaction).count()
            
            # Get successful transactions
            successful_transactions = session.query(PaymentTransaction).filter(
                PaymentTransaction.status == "captured"
            ).count()
            
            # Get failed transactions
            failed_transactions = session.query(PaymentTransaction).filter(
                PaymentTransaction.status == "failed"
            ).count()
            
            # Get pending transactions
            pending_transactions = session.query(PaymentTransaction).filter(
                PaymentTransaction.status == "pending"
            ).count()
            
            # Get total amount
            total_amount_result = session.query(func.sum(PaymentTransaction.amount)).filter(
                PaymentTransaction.status == "captured"
            ).scalar()
            total_amount = total_amount_result or 0
            
            # Calculate average
            average_amount = total_amount / successful_transactions if successful_transactions > 0 else 0
            
            return {
                "total_transactions": total_transactions,
                "total_amount": total_amount,
                "successful_transactions": successful_transactions,
                "failed_transactions": failed_transactions,
                "pending_transactions": pending_transactions,
                "average_transaction_amount": average_amount,
                "currency": "INR",
                "period": period
            }
        finally:
            self.close_session(session)

    def init_database(self):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(self.engine)
            print("Payment database tables initialized successfully")
        except Exception as e:
            print(f"Error initializing payment database: {e}")

# Global database manager instance
db_manager = DatabaseManager()
