# app/services/llm_client.py
"""LLM client for Gemini API interactions"""
import logging
import json
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from ChatAI1.chatai1_config import settings
from ChatAI1.chatai1_schemas import (
    DomainRelevance, Category, ThirdLevelIntent, Audience, ZohoCRMDataStatus
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with Gemini API"""

    def __init__(self):
        """Initialize Gemini client"""
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.router_model = genai.GenerativeModel(settings.ROUTER_MODEL_NAME)
        self.answer_model = genai.GenerativeModel(settings.ANSWER_MODEL_NAME)
        logger.info(
            f"Initialized LLMClient with router={settings.ROUTER_MODEL_NAME}, "
            f"answer={settings.ANSWER_MODEL_NAME}"
        )

    def _validate_and_normalize_router_output(self, router_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize router JSON output to ensure all enum values are valid.
        
        Args:
            router_json: Raw router JSON from model
            
        Returns:
            Normalized router JSON with valid enum values
        """
        normalized = {}
        
        # Validate domain_relevance (REQUIRED)
        domain_val = router_json.get("domain_relevance", "out_of_scope")
        try:
            normalized["domain_relevance"] = DomainRelevance(domain_val).value
        except (ValueError, TypeError):
            logger.warning(f"Invalid domain_relevance value: {domain_val}, defaulting to 'out_of_scope'")
            normalized["domain_relevance"] = DomainRelevance.OUT_OF_SCOPE.value
        
        # Validate primary_category (Optional)
        primary_cat = router_json.get("primary_category")
        if primary_cat:
            try:
                normalized["primary_category"] = Category(primary_cat).value
            except (ValueError, TypeError):
                logger.warning(f"Invalid primary_category value: {primary_cat}, setting to None")
                normalized["primary_category"] = None
        else:
            normalized["primary_category"] = None
        
        # Validate additional_categories (Optional list)
        additional_cats = router_json.get("additional_categories", [])
        if isinstance(additional_cats, list):
            validated_cats = []
            for cat in additional_cats:
                try:
                    validated_cats.append(Category(cat).value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid additional_category value: {cat}, skipping")
            normalized["additional_categories"] = validated_cats
        else:
            normalized["additional_categories"] = []
        
        # Validate is_multi_category (boolean)
        normalized["is_multi_category"] = bool(router_json.get("is_multi_category", False))
        
        # Validate third_level_intent (Optional)
        third_intent = router_json.get("third_level_intent")
        if third_intent:
            try:
                normalized["third_level_intent"] = ThirdLevelIntent(third_intent).value
            except (ValueError, TypeError):
                logger.warning(f"Invalid third_level_intent value: {third_intent}, setting to None")
                normalized["third_level_intent"] = None
        else:
            normalized["third_level_intent"] = None
        
        # Validate audience (Optional)
        audience_val = router_json.get("audience")
        if audience_val:
            try:
                normalized["audience"] = Audience(audience_val).value
            except (ValueError, TypeError):
                logger.warning(f"Invalid audience value: {audience_val}, setting to None")
                normalized["audience"] = None
        else:
            normalized["audience"] = None
        
        # Validate use_zoho_crm_data (boolean)
        normalized["use_zoho_crm_data"] = bool(router_json.get("use_zoho_crm_data", False))
        
        # Validate use_common_kb (boolean)
        normalized["use_common_kb"] = bool(router_json.get("use_common_kb", False))
        
        # Validate zoho_crm_data_status (Optional, has default)
        zoho_status = router_json.get("zoho_crm_data_status", "not_required")
        try:
            normalized["zoho_crm_data_status"] = ZohoCRMDataStatus(zoho_status).value
        except (ValueError, TypeError):
            logger.warning(f"Invalid zoho_crm_data_status value: {zoho_status}, defaulting to 'not_required'")
            normalized["zoho_crm_data_status"] = ZohoCRMDataStatus.NOT_REQUIRED.value
        
        return normalized

    def call_router_model(self, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call router model to classify intent and determine routing

        Args:
            system_prompt: Router system prompt with instructions
            payload: Dictionary with user_query, conversation_snippet, metadata

        Returns:
            Parsed router JSON output
        """
        try:
            # JSON schema example to guide the model with all valid enum values
            json_schema_example = {
                "domain_relevance": "finance_intermediary_domain",  # REQUIRED: "finance_intermediary_domain" or "out_of_scope"
                "primary_category": "Mutual Funds",  # Optional: "Mutual Funds", "Insurance", or "Stock Markets"
                "additional_categories": [],  # Optional: list of "Mutual Funds", "Insurance", or "Stock Markets"
                "is_multi_category": False,  # Optional: boolean
                "third_level_intent": None,  # Optional: one of: "educational_explanation", "regulation_or_compliance", "product_selection_or_comparison", "client_case_planning_or_suitability", "portfolio_or_policy_review", "operations_or_transaction_support", "sales_or_marketing_communication", "tools_or_workflow_or_automation", "other_in_domain"
                "audience": None,  # Optional: "intermediary" or "end_client"
                "use_zoho_crm_data": False,  # Optional: boolean
                "use_common_kb": False,  # Optional: boolean
                "zoho_crm_data_status": "not_required"  # Optional: "not_required", "available", "missing", "unknown"
            }
            
            # Construct full prompt with system instructions, schema example, and payload
            full_prompt = f"""{system_prompt}

REQUIRED JSON OUTPUT FORMAT:
{json.dumps(json_schema_example, indent=2)}

VALID ENUM VALUES:
- domain_relevance (REQUIRED): "finance_intermediary_domain" or "out_of_scope"
- primary_category: "Mutual Funds", "Insurance", or "Stock Markets" (or null)
- third_level_intent: "educational_explanation", "regulation_or_compliance", "product_selection_or_comparison", "client_case_planning_or_suitability", "portfolio_or_policy_review", "operations_or_transaction_support", "sales_or_marketing_communication", "tools_or_workflow_or_automation", "other_in_domain" (or null)
- audience: "intermediary" or "end_client" (or null)
- zoho_crm_data_status: "not_required", "available", "missing", "unknown"

IMPORTANT: Use EXACT enum values as shown above. Do not use custom strings or variations.

INPUT:
{json.dumps(payload, indent=2)}

OUTPUT (JSON only, must include "domain_relevance" field with exact enum value):"""

            logger.info(f"Calling router model with query length: {len(payload.get('user_query', ''))}")

            # Configure generation for JSON output
            generation_config = genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )

            # Generate response
            response = self.router_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # Parse JSON response
            router_json = json.loads(response.text)
            
            # Validate and normalize all enum fields
            router_json = self._validate_and_normalize_router_output(router_json)
            
            logger.info(f"Router output: domain={router_json.get('domain_relevance')}, "
                        f"category={router_json.get('primary_category')}, "
                        f"intent={router_json.get('third_level_intent')}")

            return router_json

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse router JSON: {e}, response: {response.text[:200]}")
            # Fallback to out_of_scope if JSON parsing fails
            return {
                "domain_relevance": "out_of_scope",
                "primary_category": None,
                "additional_categories": [],
                "is_multi_category": False,
                "third_level_intent": None,
                "audience": None,
                "use_zoho_crm_data": False,
                "use_common_kb": False,
                "zoho_crm_data_status": "not_required"
            }
        except Exception as e:
            logger.error(f"Error calling router model: {e}")
            raise

    def call_answer_model(
            self,
            system_prompts: List[str],
            router_json: Dict[str, Any],
            user_context: str,
            kb_context: str,
            user_query: str,
            conversation_snippet: str
    ) -> str:
        """
        Call answer model to generate final response

        Args:
            system_prompts: List of system prompts (base + domain-specific)
            router_json: Router output JSON
            user_context: User-specific context from RAG
            kb_context: Common knowledgebase context from RAG
            user_query: User's query
            conversation_snippet: Recent conversation context

        Returns:
            Generated answer text
        """
        try:
            # Combine all system prompts
            combined_system_prompt = "\n\n".join(system_prompts)

            # Replace placeholders in base system prompt
            combined_system_prompt = combined_system_prompt.replace(
                "{{ROUTER_OUTPUT_JSON}}", json.dumps(router_json, indent=2)
            )
            combined_system_prompt = combined_system_prompt.replace(
                "{{USER_QUERY}}", user_query
            )
            combined_system_prompt = combined_system_prompt.replace(
                "{{CONVERSATION_SNIPPET}}", conversation_snippet or "(No recent conversation)"
            )
            combined_system_prompt = combined_system_prompt.replace(
                "{{USER_SPECIFIC_CONTEXT}}", user_context or "(No user-specific context available)"
            )
            combined_system_prompt = combined_system_prompt.replace(
                "{{COMMON_KB_CONTEXT}}", kb_context or "(No common KB context available)"
            )

            # Construct messages for the answer model
            # Gemini expects system instruction separate from content
            full_prompt = f"{combined_system_prompt}\n\nNow, answer the user's query:\n{user_query}"

            logger.info(f"Calling answer model with prompt length: {len(full_prompt)}")

            # Configure generation
            generation_config = genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048
            )

            # Generate response
            response = self.answer_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            answer_text = response.text
            logger.info(f"Answer model generated response of length: {len(answer_text)}")

            return answer_text

        except Exception as e:
            logger.error(f"Error calling answer model: {e}")
            raise