import os
import json
import sqlite3
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class ChatAICore:
    def __init__(self, db_path: str = "chat_ai_data.db"):
        """Initialize ChatAI core with database connection"""
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT UNIQUE,
                    user_prompt TEXT,
                    ai_response TEXT,
                    system_prompt TEXT,
                    model_used TEXT,
                    provider TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    rating REAL,
                    feedback_comment TEXT
                )
            ''')
            
            # Create ratings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT UNIQUE,
                    conversation_id TEXT,
                    user_rating INTEGER,
                    feedback_comment TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ ChatAI database initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing ChatAI database: {e}")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI response using the existing backend logic"""
        try:
            # Default system prompt if none provided
            if not system_prompt:
                system_prompt = "You are a ChatGPT-style financial expert. FORMAT: Start with 📚 DEFINITION (30 words max), then 💡 KEY POINTS (1 line each), add 🎯 EXAMPLE (1-2 lines), end with ✅ PRACTICAL TAKEAWAY (1 line). Use emojis, bullet points, and clear sections. Keep it concise but engaging like ChatGPT."
            
            # Generate response using the enhanced sample response generator
            response = self._generate_sample_response(prompt)
            
            # Store conversation in database
            conversation_id = str(uuid.uuid4())
            self._store_conversation(
                conversation_id=conversation_id,
                user_prompt=prompt,
                ai_response=response,
                system_prompt=system_prompt,
                model_used="mf-assistant:latest",
                provider="langchain_modal"
            )
            
            return {
                "response": response,
                "model_used": "mf-assistant:latest",
                "provider": "langchain_modal",
                "conversation_id": conversation_id
            }
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return {
                "response": "Sorry, I couldn't process your request. Please try again.",
                "model_used": "error",
                "provider": "error",
                "conversation_id": None
            }
    
    def _generate_sample_response(self, prompt: str) -> str:
        """Generate a sample response (replace with your actual AI integration)"""
        # This is a placeholder - replace with your actual AI backend integration
        sample_responses = {
            "mutual fund": "📚 DEFINITION: A mutual fund is a type of investment vehicle that pools money from many investors to purchase a portfolio of stocks, bonds, or other securities.\n\n💡 KEY POINTS:\n• Mutual funds offer diversification and professional management\n• Investors can choose from various fund types based on goals and risk tolerance\n\n🎯 EXAMPLE: Vanguard 500 Index Fund tracks the S&P 500 index and offers broad US stock market exposure.\n\n✅ PRACTICAL TAKEAWAY: Investing in mutual funds provides access to diversified portfolios without individual security selection.",
            "etf": "📚 DEFINITION: An ETF (Exchange-Traded Fund) is a type of investment fund that trades on stock exchanges like individual stocks.\n\n💡 KEY POINTS:\n• ETFs offer diversification with lower expense ratios than mutual funds\n• They provide real-time pricing and intraday trading flexibility\n\n🎯 EXAMPLE: SPY (SPDR S&P 500 ETF) is one of the most popular ETFs tracking the S&P 500 index.\n\n✅ PRACTICAL TAKEAWAY: ETFs combine the benefits of mutual funds with the trading flexibility of stocks.",
            "stock": "📚 DEFINITION: A stock represents ownership in a company, giving shareholders a claim on the company's assets and earnings.\n\n💡 KEY POINTS:\n• Stocks offer potential for capital appreciation and dividend income\n• They carry higher risk but potentially higher returns than bonds\n\n🎯 EXAMPLE: Apple Inc. (AAPL) stock represents ownership in the technology company.\n\n✅ PRACTICAL TAKEAWAY: Stocks are fundamental building blocks for long-term wealth building.",
            "bond": "📚 DEFINITION: A bond is a fixed-income security that represents a loan made by an investor to a borrower, typically corporate or governmental.\n\n💡 KEY POINTS:\n• Bonds provide regular interest payments and return of principal at maturity\n• They are generally considered lower risk than stocks but offer lower potential returns\n\n🎯 EXAMPLE: US Treasury bonds are government-issued bonds considered among the safest investments.\n\n✅ PRACTICAL TAKEAWAY: Bonds provide income and stability to investment portfolios.",
            "dividend": "📚 DEFINITION: A dividend is a distribution of profits by a corporation to its shareholders, usually paid in cash or additional shares.\n\n💡 KEY POINTS:\n• Dividends provide regular income to shareholders\n• Companies with stable earnings typically pay consistent dividends\n\n🎯 EXAMPLE: Coca-Cola (KO) has paid quarterly dividends for over 50 years, making it a dividend aristocrat.\n\n✅ PRACTICAL TAKEAWAY: Dividend-paying stocks can provide income and potential for growth.",
            "portfolio": "📚 DEFINITION: A portfolio is a collection of financial investments like stocks, bonds, mutual funds, and other assets.\n\n💡 KEY POINTS:\n• Diversification across different asset classes reduces risk\n• Regular rebalancing helps maintain target allocation\n\n🎯 EXAMPLE: A balanced portfolio might include 60% stocks, 30% bonds, and 10% cash.\n\n✅ PRACTICAL TAKEAWAY: Building a diversified portfolio is key to long-term investment success.",
            "diversification": "📚 DEFINITION: Diversification is an investment strategy that spreads money across different assets to reduce risk.\n\n💡 KEY POINTS:\n• Don't put all your eggs in one basket - spread investments across sectors\n• Diversification can reduce portfolio volatility without sacrificing returns\n\n🎯 EXAMPLE: Instead of investing only in tech stocks, diversify across healthcare, finance, and consumer goods.\n\n✅ PRACTICAL TAKEAWAY: Diversification is essential for managing investment risk.",
            "risk": "📚 DEFINITION: Investment risk is the possibility of losing money or not achieving expected returns on an investment.\n\n💡 KEY POINTS:\n• Higher potential returns typically come with higher risk\n• Understanding your risk tolerance helps in investment decisions\n\n🎯 EXAMPLE: Stocks are riskier than bonds but historically offer higher returns over time.\n\n✅ PRACTICAL TAKEAWAY: Balance risk and return based on your financial goals and timeline.",
            "retirement": "📚 DEFINITION: Retirement planning involves saving and investing to ensure financial security after leaving the workforce.\n\n💡 KEY POINTS:\n• Start saving early to benefit from compound interest\n• Consider tax-advantaged accounts like 401(k)s and IRAs\n\n🎯 EXAMPLE: Contributing $500 monthly to a 401(k) starting at age 25 could grow to over $1 million by age 65.\n\n✅ PRACTICAL TAKEAWAY: Early and consistent retirement saving is crucial for financial security.",
            "compound": "📚 DEFINITION: Compound interest is when you earn interest on both your original investment and accumulated interest.\n\n💡 KEY POINTS:\n• Time is your greatest ally in compound growth\n• Regular contributions accelerate wealth building\n\n🎯 EXAMPLE: $10,000 invested at 7% annual return grows to $19,672 in 10 years through compounding.\n\n✅ PRACTICAL TAKEAWAY: Start investing early to maximize the power of compound interest.",
            "index": "📚 DEFINITION: An index is a statistical measure that tracks the performance of a group of securities representing a market segment.\n\n💡 KEY POINTS:\n• Index funds provide broad market exposure at low cost\n• They typically outperform most actively managed funds over time\n\n🎯 EXAMPLE: The S&P 500 index tracks 500 large US companies and is a benchmark for market performance.\n\n✅ PRACTICAL TAKEAWAY: Index investing is a simple, effective strategy for most investors.",
            "expense ratio": "📚 DEFINITION: An expense ratio is the annual fee charged by mutual funds and ETFs, expressed as a percentage of assets.\n\n💡 KEY POINTS:\n• Lower expense ratios mean more money stays in your pocket\n• Index funds typically have lower expense ratios than active funds\n\n🎯 EXAMPLE: A 1% expense ratio on a $10,000 investment costs $100 annually in fees.\n\n✅ PRACTICAL TAKEAWAY: Choose low-cost funds to maximize your investment returns.",
            "asset allocation": "📚 DEFINITION: Asset allocation is the distribution of investments across different asset classes like stocks, bonds, and cash.\n\n💡 KEY POINTS:\n• Allocation should match your risk tolerance and time horizon\n• Younger investors can typically take more risk with higher stock allocations\n\n🎯 EXAMPLE: A 30-year-old might allocate 80% to stocks and 20% to bonds, while a 60-year-old might reverse this.\n\n✅ PRACTICAL TAKEAWAY: Proper asset allocation is fundamental to investment success.",
            "market cap": "📚 DEFINITION: Market capitalization is the total value of a company's outstanding shares, calculated by share price × number of shares.\n\n💡 KEY POINTS:\n• Large-cap companies are generally more stable but may grow slower\n• Small-cap companies offer higher growth potential but more volatility\n\n🎯 EXAMPLE: Apple's market cap of $3 trillion makes it a large-cap stock, while smaller companies under $2 billion are small-cap.\n\n✅ PRACTICAL TAKEAWAY: Market cap helps categorize companies by size and investment characteristics.",
            "bull market": "📚 DEFINITION: A bull market is a period of rising stock prices, typically accompanied by investor optimism and economic growth.\n\n💡 KEY POINTS:\n• Bull markets can last for years and provide significant returns\n• Stay invested during bull markets to capture growth\n\n🎯 EXAMPLE: The 2009-2020 bull market saw the S&P 500 rise over 400% from its financial crisis low.\n\n✅ PRACTICAL TAKEAWAY: Bull markets create wealth-building opportunities for patient investors.",
            "bear market": "📚 DEFINITION: A bear market is a period of falling stock prices, typically defined as a 20% or greater decline from recent highs.\n\n💡 KEY POINTS:\n• Bear markets are normal parts of the investment cycle\n• They can present buying opportunities for long-term investors\n\n🎯 EXAMPLE: The 2020 COVID-19 bear market was the shortest in history, lasting just 33 days.\n\n✅ PRACTICAL TAKEAWAY: Bear markets are temporary and often create opportunities to buy quality investments at lower prices."
        }
        
        # Check if prompt contains any keywords
        prompt_lower = prompt.lower()
        
        # Sort keywords by length (longest first) to match multi-word terms first
        sorted_keywords = sorted(sample_responses.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            if keyword in prompt_lower:
                return sample_responses[keyword]
        
        # If no specific keyword found, provide a helpful response based on the question
        if any(word in prompt_lower for word in ["what", "how", "why", "when", "where", "which", "who"]):
            return "📚 DEFINITION: I'd be happy to help with your specific financial question.\n\n💡 KEY POINTS:\n• Please provide more details about your investment topic\n• I can explain concepts, strategies, or specific financial products\n\n🎯 EXAMPLE: Try asking about specific topics like 'mutual funds', 'retirement planning', or 'risk management'.\n\n✅ PRACTICAL TAKEAWAY: The more specific your question, the better I can assist with your financial education."
        
        # For general statements or greetings, provide a brief, helpful response
        return "📚 DEFINITION: I'm your AI financial assistant, ready to help with investment questions.\n\n💡 KEY POINTS:\n• Ask me about any financial topic or investment concept\n• I provide clear, structured explanations with practical examples\n\n🎯 EXAMPLE: Try asking 'What is a mutual fund?' or 'How does compound interest work?'\n\n✅ PRACTICAL TAKEAWAY: I'm here to help you understand investing and make informed financial decisions."
    
    def _store_conversation(self, conversation_id: str, user_prompt: str, ai_response: str, 
                           system_prompt: str, model_used: str, provider: str):
        """Store conversation in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations 
                (conversation_id, user_prompt, ai_response, system_prompt, model_used, provider)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conversation_id, user_prompt, ai_response, system_prompt, model_used, provider))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error storing conversation: {e}")
    
    def store_rating(self, trace_id: str, user_rating: int, feedback_comment: str = "") -> bool:
        """Store user rating and feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store rating
            cursor.execute('''
                INSERT OR REPLACE INTO ratings 
                (trace_id, user_rating, feedback_comment)
                VALUES (?, ?, ?)
            ''', (trace_id, user_rating, feedback_comment))
            
            # Update conversation with rating
            cursor.execute('''
                UPDATE conversations 
                SET rating = ?, feedback_comment = ?
                WHERE conversation_id = ?
            ''', (user_rating, feedback_comment, trace_id))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Rating stored successfully: {trace_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing rating: {e}")
            return False
    
    def get_conversation_history(self, limit: int = 10) -> list:
        """Get recent conversation history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT conversation_id, user_prompt, ai_response, timestamp, rating
                FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    "conversation_id": row[0],
                    "user_prompt": row[1],
                    "ai_response": row[2],
                    "timestamp": row[3],
                    "rating": row[4]
                })
            
            conn.close()
            return conversations
            
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            return []
    
    def get_rating_stats(self) -> Dict[str, Any]:
        """Get rating statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get average rating
            cursor.execute('SELECT AVG(rating) FROM conversations WHERE rating IS NOT NULL')
            avg_rating = cursor.fetchone()[0] or 0
            
            # Get total conversations
            cursor.execute('SELECT COUNT(*) FROM conversations')
            total_conversations = cursor.fetchone()[0]
            
            # Get rated conversations
            cursor.execute('SELECT COUNT(*) FROM conversations WHERE rating IS NOT NULL')
            rated_conversations = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "average_rating": round(avg_rating, 2),
                "total_conversations": total_conversations,
                "rated_conversations": rated_conversations,
                "rating_percentage": round((rated_conversations / total_conversations * 100) if total_conversations > 0 else 0, 2)
            }
            
        except Exception as e:
            print(f"❌ Error getting rating stats: {e}")
            return {
                "average_rating": 0,
                "total_conversations": 0,
                "rated_conversations": 0,
                "rating_percentage": 0
            }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Close any open database connections if needed
            print("✅ ChatAI cleanup completed")
        except Exception as e:
            print(f"❌ Error during ChatAI cleanup: {e}")
