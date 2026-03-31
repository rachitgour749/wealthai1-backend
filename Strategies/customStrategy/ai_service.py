import os
import json
import requests
from typing import Dict, Any, Optional
import logging

# Import Google Generative AI library
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.getLogger(__name__).warning("google-generativeai not installed. Please run: pip install google-generativeai")

class AIService:
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        if not self.gemini_api_key:
            self.logger = logging.getLogger(__name__)
            self.logger.error("GEMINI_API_KEY environment variable not set")
            return
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Gemini API Key loaded: {self.gemini_api_key[:10]}...")
        
        # Configure Google Generative AI
        if GENAI_AVAILABLE:
            genai.configure(api_key=self.gemini_api_key)
            # Try different models in order of preference
            self.models_to_try = [
                'gemini-2.0-flash',    # Fast and efficient
                # 'gemini-1.0-pro',      # Reliable fallback
                # 'gemini-pro'           # Classic fallback
            ]

            self.logger.info("Google Generative AI library initialized")
        else:
            self.logger.error("Google Generative AI library not available")
    
    def load_master_prompt(self) -> str:
        """Load the master prompt from Strategy_Master_Prompt.txt"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_file = os.path.join(current_dir, "Strategy_Master_Prompt.txt")
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                master_prompt = f.read()
            
            return master_prompt
        except Exception as e:
            self.logger.error(f"Error loading master prompt: {e}")
            return ""
    
    def analyze_strategy_with_gemini(self, user_description: str) -> Optional[Dict[str, Any]]:
        """Analyze strategy using Google Generative AI library"""
        try:
            if not GENAI_AVAILABLE:
                self.logger.error("Google Generative AI library not available")
                return None
            
            if not self.gemini_api_key:
                self.logger.error("Gemini API key not found")
                return None
            
            master_prompt = self.load_master_prompt()
            full_prompt = f"{master_prompt}\n{user_description}"
            
            # Configure safety settings to allow financial content
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            # Try different models in order of preference
            for model_name in self.models_to_try:
                try:
                    self.logger.info(f"Trying model: {model_name}")
                    
                    # Create model instance with safety settings
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        safety_settings=safety_settings
                    )
                    
                    # Generate content with JSON response format
                    response = model.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=2000,
                            response_mime_type="application/json"
                        )
                    )
                    
                    self.logger.debug(f"Response received from {model_name}")
                    # Check if response was blocked
                    if not response.candidates:
                        self.logger.warning(f"No candidates returned from {model_name}")
                        continue
                    
                    # Check finish reason
                    finish_reason = response.candidates[0].finish_reason
                    self.logger.info(f"Finish reason for {model_name}: {finish_reason}")
                    
                    if finish_reason != 1:  # 1 = STOP (normal completion)
                        self.logger.warning(f"Model {model_name} blocked with finish_reason: {finish_reason}")
                        continue
                    
                    if response and response.text:
                        content = response.text
                        self.logger.info(f"Received response from {model_name}")
                        
                        self.logger.debug(f"Raw response from {model_name}: {content[:200]}")
                        
                        # Try to parse JSON from the response
                        try:
                            analysis = json.loads(content)
                            self.logger.info(f"Successfully parsed JSON from {model_name}")
                            
                            # Map field names with spaces to underscore format
                            analysis = self._map_field_names(analysis)
                            
                            # Add default strategy levels if missing
                            if 'strategy_levels' not in analysis:
                                analysis['strategy_levels'] = 2  # Default to medium complexity
                                self.logger.warning("strategy_levels missing, using default value: 2")
                            
                            self.logger.debug(f"Parsed JSON from {model_name}: {list(analysis.keys())}")
                            
                            return analysis
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"JSON parsing failed for {model_name}: {e}")
                            
                            # Fallback: try to extract JSON from text
                            json_start = content.find('{')
                            json_end = content.rfind('}') + 1
                            
                            if json_start != -1 and json_end > json_start:
                                json_str = content[json_start:json_end]
                                try:
                                    analysis = json.loads(json_str)
                                    self.logger.info(f"Successfully extracted JSON from {model_name}")
                                    
                                    # Map field names with spaces to underscore format
                                    analysis = self._map_field_names(analysis)
                                    
                                    # # Add default rating if missing
                                    # if 'strategy_rating' not in analysis:
                                    #     analysis['strategy_rating'] = 2  # Default to medium complexity
                                    #     self.logger.warning("strategy_rating missing, using default value: 2")
                                    
                                    self.logger.debug(f"Fallback parsed JSON from {model_name}: {list(analysis.keys())}")
                                    
                                    return analysis
                                except json.JSONDecodeError:
                                    self.logger.warning(f"JSON extraction failed for {model_name}")
                                    continue
                            else:
                                self.logger.warning(f"No JSON found in response from {model_name}")
                                continue
                    else:
                        self.logger.warning(f"No response from {model_name}")
                        continue
                        
                except Exception as e:
                    self.logger.warning(f"Exception with {model_name}: {e}")
                    continue
            
            self.logger.error("All Gemini models failed")
            return None
                
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {e}")
            return None
    
    def analyze_strategy(self, user_description: str) -> Optional[Dict[str, Any]]:
        """Analyze strategy using Google Generative AI library"""
        analysis = self.analyze_strategy_with_gemini(user_description)
        
        if analysis:
            self.logger.info("Strategy analyzed successfully with Google Generative AI")
            return analysis
        
        self.logger.error("Google Generative AI analysis failed")
        return None
    
    def _map_field_names(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Map field names with spaces to underscore format"""
        
        # Handle list response first
        if isinstance(analysis, list):
            if len(analysis) > 0:
                analysis = analysis[0]
            else:
                return {}
        
        # Case-insensitive field mapping
        field_mapping = {
            "strategy levels": "strategy_levels",
            "trading instrument(s)/symbol(s)": "trading_instruments", 
            "time frame": "time_frame",
            "trading rule(s)": "trading_rules",
            "position sizing": "position_sizing",
            "risk management": "risk_management",
            "precise strategy logic/workflow": "strategy_logic"
        }
        
        mapped_analysis = {}
        for key, value in analysis.items():
            key_lower = key.lower()
            if key_lower in field_mapping:
                mapped_key = field_mapping[key_lower]
                # Convert strategy_levels to integer
                if mapped_key == 'strategy_levels':
                    try:
                        mapped_analysis[mapped_key] = int(value)
                    except (ValueError, TypeError):
                        mapped_analysis[mapped_key] = value  # Keep original if conversion fails
                else:
                    mapped_analysis[mapped_key] = value
            else:
                mapped_analysis[key] = value
        
        return mapped_analysis
    
    def validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate the AI analysis response"""
        self.logger.info(f"Validating analysis: {analysis}")
        self.logger.info(f"Analysis type: {type(analysis)}")
        
        required_fields = [
            'strategy_levels',
            'time_frame',
            'trading_rules',
            'position_sizing',
            'risk_management',
            'strategy_logic'
        ]
        
        for field in required_fields:
            if field not in analysis:
                self.logger.error(f"Missing required field: {field}")
                self.logger.error(f"Available fields: {list(analysis.keys())}")
                return False
            else:
                self.logger.info(f"Field {field}: {analysis[field]} (type: {type(analysis[field])})")
        
        # Validate strategy levels
        if not isinstance(analysis['strategy_levels'], int) or analysis['strategy_levels'] < 1 or analysis['strategy_levels'] > 4:
            self.logger.error(f"Invalid strategy levels: {analysis['strategy_levels']} (type: {type(analysis['strategy_levels'])})")
            return False
        
        # Validate trading rules is a list
        if not isinstance(analysis['trading_rules'], list):
            self.logger.error(f"Trading rules should be a list, got: {type(analysis['trading_rules'])}")
            return False
        
        self.logger.info("Validation passed successfully")
        return True