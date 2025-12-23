import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
import logging
from datetime import datetime

class EmailService:
    def __init__(self):
        # Hardcode SMTP credentials (who sends the emails)
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587
        self.email_user = 'mohitswealthwisers@gmail.com'  # SMTP sender
        self.email_password = 'pngzpehdbkixxzob'  # SMTP password
        self.team_email = 'mohitswealthwisers@gmail.com'  # Team notification recipient
        self.logger = logging.getLogger(__name__)
        
        # Test email configuration
        if self.email_user and self.email_password:
            self.logger.info("✅ Email credentials configured successfully")
        else:
            self.logger.error("❌ Email credentials not configured")
    
    def send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email using SMTP"""
        try:
            if not all([self.email_user, self.email_password]):
                self.logger.error("Email credentials not configured")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_user, to_email, text)
            server.quit()
            
            self.logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email to {to_email}: {e}")
            return False
    
    
    
    def send_strategy_notifications(self, strategy_data: Dict[str, Any]) -> Dict[str, bool]:
        """Send email notifications to user and team"""
        results = {
            'user_email_sent': False,
            'team_email_sent': False
        }
        
        # Send user confirmation email
        user_subject = "🎯 Your Custom Strategy Has Been Created - WealthAI"
        user_body = self.create_user_email_body(strategy_data)
        
        results['user_email_sent'] = self.send_email(
            strategy_data.get('user_email'),
            user_subject,
            user_body
        )
        
        # Send team notification email
        team_subject = f"🚨 New Custom Strategy Request - ID: {strategy_data.get('id')} - Rating: {strategy_data.get('analysis', {}).get('strategy_rating', 'N/A')}/4"
        team_body = self.create_team_email_body(strategy_data)
        
        results['team_email_sent'] = self.send_email(
            self.team_email,
            team_subject,
            team_body
        )
        
        return results
