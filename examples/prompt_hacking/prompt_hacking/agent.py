#!/usr/bin/env python3
"""
Prompt Hacking Detection Agent
"""

import os
import json
import requests
import re
from typing import Dict, Any
from dotenv import load_dotenv

from northau.autotuning import Config

# Load environment variables
load_dotenv()


class PromptHackingAgent:
    """Agent for detecting prompt hacking attempts using LLM API calls."""
    
    def __init__(self, config: Config):
        self.config = config
        # Handle both dict and string system prompts for backward compatibility
        self.system_prompt = config.system_prompts.get('default')
        self.llm_config = config.llm_config
        self.last_token_usage = {}
        
        # LLM configuration from environment
        self.model = self.llm_config['model']
        self.base_url = self.llm_config['base_url']
        self.api_key = self.llm_config['api_key']
        
    def run(self, message: str) -> str:
        """Process message for experiment runner - returns JSON string."""
        result = self.classify_prompt(message)
        return json.dumps(result)
    
    def classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """Classify if a prompt contains hacking attempts."""
        
        # Build the classification prompt
        classification_prompt = f"""

USER PROMPT TO ANALYZE:
```
{prompt}
```

Analyze this prompt and respond with ONLY a JSON object:
{{
    "is_prompt_hack": true/false,
    "confidence": 0.85,
    "reasoning": "brief explanation"
}}
"""
        
        try:
            # Make API call to LLM
            response = self._call_llm(self.system_prompt, classification_prompt)
            
            # Parse response
            result = self._parse_response(response)
            
            # Mock token usage for demonstration
            self.last_token_usage = {
                "prompt_tokens": len(classification_prompt.split()) * 2,
                "completion_tokens": len(str(response).split()) * 2,
                "total_tokens": len(classification_prompt.split()) * 2 + len(str(response).split()) * 2
            }
            
            return result
            
        except Exception as e:
            print(f"Error in classification: {e}")
            # Return safe default
            return {
                "is_prompt_hack": False,
                "confidence": 0.0,
                "reasoning": f"Error during classification: {str(e)}"
            }
    
    def _call_llm(self, system_prompt: str, prompt: str) -> str:
        """Make API call to LLM service."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt}, 
                {"role": "user", "content": prompt}
            ],
            "temperature": self.llm_config.get("temperature", 0.3),
            "max_tokens": self.llm_config.get("max_tokens", 300)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            return f"Network Error: {str(e)}"
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                return {
                    "is_prompt_hack": data.get("is_prompt_hack", False),
                    "confidence": max(0.0, min(1.0, data.get("confidence", 0.0))),
                    "reasoning": data.get("reasoning", "")
                }
            else:
                # Fallback parsing
                response_lower = response.lower()
                is_hack = any(word in response_lower for word in [
                    "injection", "jailbreak", "hack", "malicious", "attack"
                ])
                
                return {
                    "is_prompt_hack": is_hack,
                    "confidence": 0.5,
                    "reasoning": "Parsed from unstructured response"
                }
                
        except Exception as e:
            return {
                "is_prompt_hack": False,
                "confidence": 0.0,
                "reasoning": f"Parse error: {str(e)}"
            }


def create_agent_factory():
    """Factory function to create agents."""
    def factory(config: Config) -> PromptHackingAgent:
        return PromptHackingAgent(config)
    return factory