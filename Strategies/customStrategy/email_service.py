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
    
    def create_user_email_body(self, strategy_data: Dict[str, Any]) -> str:
        """Create HTML email body for user confirmation"""
        analysis = strategy_data.get('analysis', {})
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .strategy-details {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
                .rating {{ font-weight: bold; color: #4CAF50; }}
                .footer {{ text-align: center; padding: 20px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Custom Strategy Created Successfully!</h1>
                </div>
                
                <div class="content">
                    <p>Dear Valued Client,</p>
                    
                    <p>Thank you for creating a custom trading strategy with WealthAI. Your strategy has been successfully analyzed and saved to our system.</p>
                    
                    <div class="strategy-details">
                        <h3>Strategy Details</h3>
                        <p><strong>Strategy ID:</strong> {strategy_data.get('id', 'N/A')}</p>
                        <p><strong>Created:</strong> {strategy_data.get('created_at', 'N/A')}</p>
                        <p><strong>Complexity Rating:</strong> <span class="rating">{analysis.get('strategy_rating', 'N/A')}/4</span></p>
                    </div>
                    
                    <div class="strategy-details">
                        <h3>Your Strategy Description</h3>
                        <p>{strategy_data.get('strategy_description', 'N/A')}</p>
                    </div>
                    
                    <div class="strategy-details">
                        <h3>AI Analysis Summary</h3>
                        <p><strong>Trading Instruments:</strong> {analysis.get('trading_instruments', 'N/A')}</p>
                        <p><strong>Time Frame:</strong> {analysis.get('time_frame', 'N/A')}</p>
                        <p><strong>Position Sizing:</strong> {analysis.get('position_sizing', 'N/A')}</p>
                        <p><strong>Risk Management:</strong> {analysis.get('risk_management', 'N/A')}</p>
                    </div>
                    
                    <p>Our team will review your strategy and contact you within 24-48 hours to discuss implementation details.</p>
                    
                    <p>If you have any questions, please don't hesitate to contact us.</p>
                </div>
                
                <div class="footer">
                    <p>Best regards,<br>The WealthAI Team</p>
                    <p>📧 support@wealthai.com | 📞 +91-XXXX-XXXX</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_body
    
    def create_team_email_body(self, strategy_data: Dict[str, Any]) -> str:
        """Create HTML email body for team notification"""
        analysis = strategy_data.get('analysis', {})
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2196F3; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .strategy-details {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }}
                .urgent {{ background-color: #ffebee; border-left-color: #f44336; }}
                .footer {{ text-align: center; padding: 20px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 New Custom Strategy Request</h1>
                </div>
                
                <div class="content">
                    <p>Team,</p>
                    
                    <p>A new custom trading strategy has been submitted and requires your attention.</p>
                    
                    <div class="strategy-details {'urgent' if analysis.get('strategy_rating', 0) >= 3 else ''}">
                        <h3>Strategy Information</h3>
                        <p><strong>Strategy ID:</strong> {strategy_data.get('id', 'N/A')}</p>
                        <p><strong>User Email:</strong> {strategy_data.get('user_email', 'N/A')}</p>
                        <p><strong>User Phone:</strong> {strategy_data.get('user_phone', 'N/A')}</p>
                        <p><strong>Created:</strong> {strategy_data.get('created_at', 'N/A')}</p>
                        <p><strong>Complexity Rating:</strong> {analysis.get('strategy_rating', 'N/A')}/4</p>
                    </div>
                    
                    <div class="strategy-details">
                        <h3>Strategy Description</h3>
                        <p>{strategy_data.get('strategy_description', 'N/A')}</p>
                    </div>
                    
                    <div class="strategy-details">
                        <h3>AI Analysis</h3>
                        <p><strong>Trading Instruments:</strong> {analysis.get('trading_instruments', 'N/A')}</p>
                        <p><strong>Time Frame:</strong> {analysis.get('time_frame', 'N/A')}</p>
                        <p><strong>Trading Rules:</strong></p>
                        <ul>
                            {''.join([f'<li>{rule}</li>' for rule in analysis.get('trading_rules', [])])}
                        </ul>
                        <p><strong>Position Sizing:</strong> {analysis.get('position_sizing', 'N/A')}</p>
                        <p><strong>Risk Management:</strong> {analysis.get('risk_management', 'N/A')}</p>
                        <p><strong>Strategy Logic:</strong> {analysis.get('strategy_logic', 'N/A')}</p>
                    </div>
                    
                    <p><strong>Action Required:</strong> Please review this strategy and contact the client within 24-48 hours.</p>
                </div>
                
                <div class="footer">
                    <p>WealthAI Strategy Management System</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_body
    
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
