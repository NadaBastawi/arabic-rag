"""
LLM service for generating text responses.

This module provides the LLMService class for generating text responses
using large language models, with support for:
- Local transformers models (JAIS, Falcon)
- API-based models (Gemini, OpenAI)
"""

from typing import Optional, Union
import logging
import re
import time

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for generating text responses using large language models.
    
    Supports both local transformers models and API-based models.
    
    Attributes:
        model: Model identifier or type.
        api_key: API key for external models.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    LLM_QUOTA_EXCEEDED = "__LLM_QUOTA_EXCEEDED__"
    LLM_API_UNAVAILABLE = "__LLM_API_UNAVAILABLE__"
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_length: int = 512,
        max_new_tokens: int = 256
    ):
        """
        Initialize the LLM service.
        
        Args:
            model: Model name or type. Options:
                - "gemini-2.0-flash", "gemini-pro": Google Gemini API
                - "gpt-4o", "gpt-4o-mini": OpenAI API
                - Local model names (e.g., "tiiuae/falcon-7b-instruct")
            api_key: API key for external models.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            max_length: Max input length for local models.
            max_new_tokens: Max new tokens for local models.
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        
        self._client = None
        self._is_api = model.startswith(("gemini", "gpt", "claude", "openai"))
        self._quota_backoff_until_ts = 0.0
        
        if self._is_api:
            self._init_api_client()
        else:
            self._init_local_model()
    
    def _init_api_client(self) -> None:
        """Initialize API client for external LLM."""
        if not self.api_key:
            logger.warning("No API key provided, using mock client")
            self._client = None
            return
        
        if "gemini" in self.model.lower():
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
                logger.info(f"Initialized Gemini client for {self.model}")
            except ImportError:
                logger.error("google-generativeai not installed")
                self._client = None
        elif "gpt" in self.model.lower() or "openai" in self.model.lower():
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client for {self.model}")
            except ImportError:
                logger.error("openai not installed")
                self._client = None

    @staticmethod
    def _is_quota_or_rate_limit_error(error_text: str) -> bool:
        """Detect provider quota/rate-limit failures."""
        text = (error_text or "").lower()
        indicators = (
            "429",
            "quota exceeded",
            "rate limit",
            "resource exhausted",
            "generate_content_free_tier",
            "too many requests",
        )
        return any(indicator in text for indicator in indicators)

    @staticmethod
    def _extract_retry_delay_seconds(error_text: str) -> int:
        """Extract retry delay from provider error text when available."""
        text = error_text or ""

        match = re.search(r"retry(?:\s+in)?\s+([0-9]+(?:\.[0-9]+)?)\s*s", text, flags=re.IGNORECASE)
        if match:
            return max(1, int(float(match.group(1))))

        match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return max(1, int(match.group(1)))

        return 30

    def _activate_quota_backoff(self, error_text: str) -> None:
        retry_seconds = self._extract_retry_delay_seconds(error_text)
        retry_seconds = min(max(retry_seconds, 15), 900)
        self._quota_backoff_until_ts = time.time() + retry_seconds
        logger.warning(
            "Quota/rate limit detected. Skipping API generation for the next %s seconds.",
            retry_seconds,
        )

    def is_quota_backoff_active(self) -> bool:
        return self._quota_backoff_until_ts > time.time()
    
    def _init_local_model(self) -> None:
        """Initialize local transformers model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading local model: {self.model}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                trust_remote_code=True
            )
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self._device != "cuda" or self._model.device.type != self._device:
                self._model = self._model.to(self._device)
            
            self._model.eval()
            logger.info(f"Loaded local model on {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {str(e)}")
            raise RuntimeError(f"Failed to load LLM model: {str(e)}") from e
    
    def generate(self, prompt: str) -> str:
        """
        Generate a text response from the LLM.
        
        Args:
            prompt: Input prompt.
            
        Returns:
            Generated text response.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        if self._is_api:
            return self._generate_api(prompt)
        else:
            return self._generate_local(prompt)
    
    def _generate_api(self, prompt: str) -> str:
        """Generate using API."""
        if self._client is None:
            return self.LLM_API_UNAVAILABLE
        if self.is_quota_backoff_active():
            return self.LLM_QUOTA_EXCEEDED
        
        try:
            if "gemini" in self.model.lower():
                return self._generate_gemini(prompt)
            elif "gpt" in self.model.lower():
                return self._generate_openai(prompt)
            else:
                return self.LLM_API_UNAVAILABLE
        except Exception as e:
            logger.error(f"API generation error: {str(e)}")
            if self._is_quota_or_rate_limit_error(str(e)):
                self._activate_quota_backoff(str(e))
                return self.LLM_QUOTA_EXCEEDED
            return f"Error generating response: {str(e)}"
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate using Gemini API."""
        import google.generativeai as genai
        
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        model = genai.GenerativeModel(
            self.model,
            generation_config=generation_config
        )
        
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return text
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            max_tokens=self.max_tokens
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty response")
        return content

    def _generate_local(self, prompt: str) -> str:
        """Generate using local model."""
        try:
            import torch

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - self.max_new_tokens
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )

            generated_text = self._tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Local generation error: {str(e)}")
            raise RuntimeError(f"Failed to generate text: {str(e)}") from e



