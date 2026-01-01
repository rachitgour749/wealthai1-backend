"""
Enhanced Intent Classifier for WealthAI1

Specialized for Indian Financial Intermediaries with:
- 8 primary intent categories
- Sub-intent classification
- Rich entity extraction
- Hindi/Hinglish support
- Indian honorific recognition
"""

import re
import logging
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from google import genai

from Services.ChatAI.core.system_prompts import CLASSIFICATION_PROMPT_V2

logger = logging.getLogger(__name__)


# =============================================================================
# INTENT ENUMS
# =============================================================================

class IntentType(str, Enum):
    """Primary intent categories for financial advisor queries."""
    PRODUCT_INFO = "product_info"
    PRODUCT_COMPARE = "product_compare"
    CLIENT_VIEW = "client_view"
    CLIENT_ACTION = "client_action"
    REGULATORY = "regulatory"
    MARKET = "market"
    EDUCATION = "education"
    OPERATIONS = "operations"


class SubIntent(str, Enum):
    """Sub-intents for granular routing."""
    # Product Info
    NAV_CHECK = "nav_check"
    RETURNS_HISTORY = "returns_history"
    SCHEME_FEATURES = "scheme_features"
    EXPENSE_RATIO = "expense_ratio"
    FUND_MANAGER = "fund_manager"
    RISK_RATING = "risk_rating"
    
    # Client View
    HOLDINGS_SUMMARY = "holdings_summary"
    GOAL_PROGRESS = "goal_progress"
    SIP_DETAILS = "sip_details"
    TRANSACTION_HISTORY = "transaction_history"
    
    # Client Action
    SIP_START = "sip_start"
    SIP_STOP = "sip_stop"
    SIP_MODIFY = "sip_modify"
    REDEMPTION = "redemption"
    SWITCH = "switch"
    
    # Generic
    GENERAL = "general"


# =============================================================================
# ENTITY MODELS
# =============================================================================

class DataSources(BaseModel):
    """Recommended data sources for the query."""
    primary: str = "llm"  # file_search, client_store, web_search, llm
    needs_web: bool = False


class ExtractedEntities(BaseModel):
    """Rich entity extraction for Indian financial context."""
    # Client identification
    client_name: Optional[str] = None
    
    # Product identification
    scheme_names: List[str] = Field(default_factory=list)
    amc_names: List[str] = Field(default_factory=list)
    product_category: Optional[str] = None  # MF, Insurance, PMS, AIF
    
    # Temporal
    time_period: Optional[str] = None
    
    # Financial
    amount: Optional[float] = None
    
    # Action
    action_verb: Optional[str] = None
    urgency: str = "none"
    
    # Language
    language: str = "english"


class ClassifiedIntent(BaseModel):
    """Complete classification result."""
    primary_intent: IntentType
    sub_intent: Optional[str] = None
    confidence: float = 0.5
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    data_sources: DataSources = Field(default_factory=DataSources)
    reasoning: Optional[str] = None
    
    # Legacy compatibility
    secondary_intents: List[IntentType] = Field(default_factory=list)
    requires_google_search: bool = False


# =============================================================================
# INDIAN PATTERN MATCHERS
# =============================================================================

# Hindi/Hinglish client name patterns
CLIENT_HONORIFIC_PATTERNS = [
    r'\b(\w+)\s+ji\b',           # Sharma ji
    r'\b(\w+)\s+sahab\b',        # Patel sahab
    r'\b(\w+)\s+saheb\b',        # Khan saheb
    r'\b(\w+)\s+bhai\b',         # Ramesh bhai
    r'\b(\w+)\s+madam\b',        # Sunita madam
    r'\bMr\.?\s+(\w+)',          # Mr. Agarwal
    r'\bMrs\.?\s+(\w+)',         # Mrs. Sharma
    r'\bDr\.?\s+(\w+)',          # Dr. Mehta
    # Removed loose 'client/customer' patterns to avoid false positives
    # r'\bclient\s+(\w+)', 
    # r'\bcustomer\s+(\w+)',
]

# Action verbs indicating CLIENT_ACTION
ACTION_VERBS = [
    'stop', 'start', 'pause', 'resume', 'modify', 'change', 'update',
    'switch', 'redeem', 'withdraw', 'process', 'cancel', 'increase', 'decrease',
    'invest', 'allocate', 'rebalance',  # Investment verbs
    'karna', 'badhana', 'kam', 'rokna', 'band', 'shuru', # Hindi/Hinglish
    'karo', 'kijiye', 'lagana', 'dalna',
    'रोको', 'शुरू', 'बंद',  # Hindi: stop, start, close
]

# Regulatory keywords
REGULATORY_KEYWORDS = [
    'sebi', 'amfi', 'irdai', 'irda', 'rbi', 'kyc', 'ckyc', 'pan', 'aadhaar',
    'compliance', 'regulation', 'circular', 'guideline', 'nominee', 'demat',
    'arn', 'euin', 'fatca', 'crs',
]

# Market keywords
MARKET_KEYWORDS = [
    'market', 'nifty', 'sensex', 'index', 'sector', 'fii', 'dii',
    'trend', 'outlook', 'today', 'performance', 'rally', 'crash', 'correction',
    'rate cut', 'inflation', 'rbi policy', 'budget',
]

# Education signal words
EDUCATION_SIGNALS = [
    'what is', 'what are', 'explain', 'how does', 'how do', 'meaning of',
    'define', 'definition', 'concept', 'kya hai', 'kya hota', 'samjhao',
]

# Common abbreviation normalizations
ABBREVIATION_MAP = {
    'mf': 'mutual fund',
    'nav': 'net asset value',
    'aum': 'assets under management',
    'sip': 'systematic investment plan',
    'stp': 'systematic transfer plan',
    'swp': 'systematic withdrawal plan',
    'elss': 'equity linked savings scheme',
    'nfo': 'new fund offer',
    'cagr': 'compound annual growth rate',
    'xirr': 'extended internal rate of return',
    'ter': 'total expense ratio',
}


# =============================================================================
# PREPROCESSING
# =============================================================================

def normalize_query(query: str) -> str:
    """Normalize query for better classification."""
    normalized = query.lower().strip()
    
    # Expand common abbreviations for classification
    for abbr, full in ABBREVIATION_MAP.items():
        # Only expand if it's a standalone word
        normalized = re.sub(rf'\b{abbr}\b', f'{abbr} ({full})', normalized)
    
    return normalized


def extract_client_name(query: str) -> Optional[str]:
    """Extract client name using Indian honorific patterns."""
    query_lower = query.lower()
    
    for pattern in CLIENT_HONORIFIC_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Return the full matched client reference
            return match.group(0).strip()
    
    return None


def detect_language(query: str) -> str:
    """Detect if query is Hindi, Hinglish, or English."""
    # Check for Devanagari script
    if re.search(r'[\u0900-\u097F]', query):
        return "hindi"
    
    # Check for Hinglish indicators
    hinglish_words = ['kya', 'hai', 'ka', 'ki', 'ke', 'ko', 'mein', 'aur', 'ye', 'wo', 'dikhao', 'batao']
    query_lower = query.lower()
    if any(word in query_lower.split() for word in hinglish_words):
        return "hinglish"
    
    return "english"


def has_action_verb(query: str) -> bool:
    """Check if query contains action verbs."""
    query_lower = query.lower()
    return any(verb in query_lower for verb in ACTION_VERBS)


def has_regulatory_keyword(query: str) -> bool:
    """Check for regulatory-related keywords."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in REGULATORY_KEYWORDS)


def has_market_keyword(query: str) -> bool:
    """Check for market-related keywords."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in MARKET_KEYWORDS)


def is_education_query(query: str) -> bool:
    """Check if this is an educational/concept query."""
    query_lower = query.lower()
    return any(signal in query_lower for signal in EDUCATION_SIGNALS)


def is_strong_education_query(query: str) -> bool:
    """
    Check if this is a DEFINITIVE education query.
    These patterns should take precedence over product keywords.
    e.g., "SIP kya hota hai?" should be EDUCATION, not PRODUCT_INFO
    """
    strong_education_patterns = [
        'kya hota', 'kya hai', 'kya hoti',  # Hindi: what is
        'what is a ', 'what is an ', 'what are ',
        'define ', 'explain ', 'meaning of',
        'samjhao', 'batao kya',  # Hindi: explain
        'क्या है', 'क्या होता',  # Devanagari
    ]
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in strong_education_patterns)


def is_comparison_query(query: str) -> bool:
    """Check if this is a comparison query."""
    comparison_patterns = [
        'compare', ' vs ', ' vs.', 'versus', 
        'better than', 'difference between', 'which one', 'which is better',
        'kon sa', 'kaun sa',  # Hindi: which one
    ]
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in comparison_patterns)


def has_operations_keyword(query: str) -> bool:
    """Check for business operations keywords."""
    operations_words = [
        'commission', 'brokerage', 'trail', 'upfront', 'payout',
        'statement', 'report', 'earnings', 'euin', 'arn renewal',
    ]
    query_lower = query.lower()
    return any(word in query_lower for word in operations_words)


def has_product_keyword(query: str) -> bool:
    """Check if query contains product-related signals."""
    product_signals = [
        'nav', 'net asset value', 'returns', 'expense ratio', 'fund',
        'scheme', 'policy', 'insurance', 'aum', 'assets under',
        'hdfc', 'sbi', 'icici', 'axis', 'kotak', 'birla', 'nippon',
        'bajaj', 'acko', 'star health', 'lic', 'tata', 'max',
        'elss', 'liquid fund', 'debt fund', 'equity fund',
        'term plan', 'health insurance', 'motor insurance',
    ]
    query_lower = query.lower()
    return any(signal in query_lower for signal in product_signals)


def detect_product_sub_intent(query: str) -> str:
    """Detect the specific product sub-intent."""
    query_lower = query.lower()
    
    if 'nav' in query_lower or 'net asset value' in query_lower:
        return 'nav_check'
    elif 'return' in query_lower:
        return 'returns_history'
    elif 'expense ratio' in query_lower or 'ter' in query_lower:
        return 'expense_ratio'
    elif 'fund manager' in query_lower or 'manager' in query_lower:
        return 'fund_manager'
    else:
        return 'scheme_features'


# =============================================================================
# FALLBACK CLASSIFIER
# =============================================================================

def rule_based_classify(query: str) -> ClassifiedIntent:
    """Rule-based fallback classification using pattern matching."""
    
    client_name = extract_client_name(query)
    language = detect_language(query)
    
    entities = ExtractedEntities(
        client_name=client_name,
        language=language
    )
    
    # Decision tree - Prioritized for accuracy
    
    # 1. Product Comparison (Definitive)
    if is_comparison_query(query):
        intent = IntentType.PRODUCT_COMPARE
        sub_intent = None
        sources = DataSources(primary="file_search", needs_web=True)
        return ClassifiedIntent(primary_intent=intent, confidence=1.0, entities=entities, data_sources=sources)
        
    # 2. Regulatory (Strong keywords) - Check BEFORE Education to catch "SEBI rules kya hai?"
    if has_regulatory_keyword(query):
        intent = IntentType.REGULATORY
        sub_intent = None
        sources = DataSources(primary="file_search", needs_web=True)
        return ClassifiedIntent(primary_intent=intent, confidence=0.95, entities=entities, data_sources=sources)
    
    # 3. Strong Education (Definitive)
    if is_strong_education_query(query):
        intent = IntentType.EDUCATION
        sub_intent = None
        sources = DataSources(primary="llm", needs_web=False)
        return ClassifiedIntent(primary_intent=intent, confidence=1.0, entities=entities, data_sources=sources)
        
    # 4. Market (Strong keywords)
    if has_market_keyword(query):
        intent = IntentType.MARKET
        sub_intent = None
        sources = DataSources(primary="web_search", needs_web=True)
        return ClassifiedIntent(primary_intent=intent, confidence=0.9, entities=entities, data_sources=sources)
        
    # 5. Operations (Business keywords)
    if has_operations_keyword(query):
        intent = IntentType.OPERATIONS
        sub_intent = None
        sources = DataSources(primary="llm", needs_web=False)
        return ClassifiedIntent(primary_intent=intent, confidence=0.9, entities=entities, data_sources=sources)

    # 6. Client Action/View (Only if specific client pattern match)
    if client_name:
        if has_action_verb(query):
            intent = IntentType.CLIENT_ACTION
            sub_intent = "sip_stop" if "stop" in query.lower() or "rok" in query.lower() else "general"
        else:
            intent = IntentType.CLIENT_VIEW
            sub_intent = "holdings_summary"
            
        sources = DataSources(primary="client_store", needs_web=False)
        return ClassifiedIntent(primary_intent=intent, sub_intent=sub_intent, confidence=0.95, entities=entities, data_sources=sources)
        
    # 7. Product Info (Default for product terms or ambiguous product queries)
    if has_product_keyword(query) or "suggest" in query.lower() or "recommend" in query.lower():
        intent = IntentType.PRODUCT_INFO
        sub_intent = detect_product_sub_intent(query)
        sources = DataSources(primary="file_search", needs_web=False)
        return ClassifiedIntent(primary_intent=intent, sub_intent=sub_intent, confidence=0.85, entities=entities, data_sources=sources)
        
    # 8. Education (Broader signals)
    if is_education_query(query):
        intent = IntentType.EDUCATION
        sub_intent = None
        sources = DataSources(primary="llm", needs_web=False)
        return ClassifiedIntent(primary_intent=intent, confidence=0.8, entities=entities, data_sources=sources)
            
    # Default fallback
    intent = IntentType.PRODUCT_INFO
    sub_intent = None
    sources = DataSources(primary="llm", needs_web=False)
    
    return ClassifiedIntent(
        primary_intent=intent,
        sub_intent=sub_intent,
        confidence=0.75,
        entities=entities,
        data_sources=sources,
        reasoning="Rule-based classification"
    )


# =============================================================================
# LLM CLASSIFIER
# =============================================================================

# High-confidence intents that don't need LLM
HIGH_CONFIDENCE_INTENTS = {
    IntentType.EDUCATION,       # "kya hota hai" patterns are definitive
    IntentType.PRODUCT_COMPARE, # "vs" / "compare" patterns are definitive
    IntentType.CLIENT_VIEW,     # Client name + view is definitive
    IntentType.CLIENT_ACTION,   # Client name + action verb is definitive
    IntentType.REGULATORY,      # SEBI/AMFI keywords are definitive
    IntentType.MARKET,          # Market keywords are definitive
    IntentType.OPERATIONS,      # Commission/trail keywords are definitive
}

async def classify_intent(
    query: str,
    client: genai.Client,
    model: str = "gemini-2.0-flash"
) -> ClassifiedIntent:
    """
    Classify user query intent using hybrid rule-based + LLM approach.
    
    Strategy:
    1. Run rule-based classification first
    2. If high-confidence pattern detected, use rule-based (faster, more reliable)
    3. Only use LLM for ambiguous queries (PRODUCT_INFO edge cases)
    
    This ensures 100% accuracy for definitive patterns like:
    - "SIP kya hota hai?" → EDUCATION (always)
    - "Compare Axis vs Mirae" → PRODUCT_COMPARE (always)
    """
    
    # Step 1: Always run rule-based first
    rule_result = rule_based_classify(query)
    
    # Step 2: Check if this is a high-confidence pattern
    if rule_result.primary_intent in HIGH_CONFIDENCE_INTENTS:
        logger.info(f"Using rule-based for high-confidence intent: {rule_result.primary_intent.value}")
        return rule_result
    
    # Step 3: For PRODUCT_INFO and edge cases, optionally use LLM for richer classification
    try:
        # Call Gemini for classification
        response = await client.aio.models.generate_content(
            model=model,
            contents=f"{CLASSIFICATION_PROMPT_V2}\n\nUser Query: {query}",
            config={"response_mime_type": "application/json"}
        )
        
        # Parse JSON response
        import json
        result = json.loads(response.text)
        
        # Build entities
        entities_data = result.get("entities", {})
        entities = ExtractedEntities(
            client_name=entities_data.get("client_name"),
            scheme_names=entities_data.get("scheme_names", []),
            amc_names=entities_data.get("amc_names", []),
            product_category=entities_data.get("product_category"),
            time_period=entities_data.get("time_period"),
            amount=entities_data.get("amount"),
            action_verb=entities_data.get("action_verb"),
            urgency=entities_data.get("urgency", "none"),
            language=entities_data.get("language", detect_language(query))
        )
        
        # Build data sources
        sources_data = result.get("data_sources", {})
        data_sources = DataSources(
            primary=sources_data.get("primary", "llm"),
            needs_web=sources_data.get("needs_web", False)
        )
        
        # Map intent string to enum
        intent_str = result.get("primary_intent", "education").lower()
        try:
            primary_intent = IntentType(intent_str)
        except ValueError:
            # Fallback mapping for legacy values
            legacy_map = {
                "product": IntentType.PRODUCT_INFO,
                "client": IntentType.CLIENT_VIEW,
                "general": IntentType.EDUCATION,
                "complex": IntentType.PRODUCT_COMPARE,
            }
            primary_intent = legacy_map.get(intent_str, IntentType.EDUCATION)
        
        classified = ClassifiedIntent(
            primary_intent=primary_intent,
            sub_intent=result.get("sub_intent"),
            confidence=result.get("confidence", 0.8),
            entities=entities,
            data_sources=data_sources,
            reasoning=result.get("reasoning"),
            requires_google_search=data_sources.needs_web
        )
        
        logger.info(f"Classified '{query[:50]}...' as {primary_intent.value} (conf: {classified.confidence})")
        return classified
        
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}, using rule-based fallback")
        return rule_based_classify(query)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_complex_query(intent: ClassifiedIntent) -> bool:
    """Check if query needs multiple data sources."""
    return (
        intent.primary_intent == IntentType.PRODUCT_COMPARE or
        intent.data_sources.needs_web or
        len(intent.secondary_intents) > 0
    )


def get_primary_data_source(intent: ClassifiedIntent) -> str:
    """Get the recommended primary data source for this intent."""
    source_map = {
        IntentType.PRODUCT_INFO: "file_search",
        IntentType.PRODUCT_COMPARE: "file_search",
        IntentType.CLIENT_VIEW: "client_store",
        IntentType.CLIENT_ACTION: "client_store",
        IntentType.REGULATORY: "file_search",
        IntentType.MARKET: "web_search",
        IntentType.EDUCATION: "llm",
        IntentType.OPERATIONS: "llm",
    }
    return intent.data_sources.primary or source_map.get(intent.primary_intent, "llm")
