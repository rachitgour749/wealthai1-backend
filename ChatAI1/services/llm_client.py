# app/services/llm_client.py
"""LLM client for Gemini API interactions"""
import logging
import json
from typing import Dict, List, Any
import google.generativeai as genai
from ChatAI1.chatai1_config import settings

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
            # Construct full prompt with system instructions and payload
            full_prompt = f"{system_prompt}\n\nINPUT:\n{json.dumps(payload, indent=2)}\n\nOUTPUT (JSON only):"

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
            logger.info(f"Router output: domain={router_json.get('domain_relevance')}, "
                        f"category={router_json.get('primary_category')}")

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