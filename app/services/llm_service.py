"""
LLM service for generating text responses.

This module provides the LLMService class for generating text responses
using large language models, with support for Arabic and multilingual models.
"""

from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLMService:
    """
    Service for generating text responses using large language models.
    
    This class handles the loading and usage of LLM models, designed to work
    with models like JAIS, Falcon, and other Hugging Face transformers models.
    The service is modular and can be easily extended to support different
    model backends or APIs in the future.
    
    Attributes:
        model_name: Name or path of the Hugging Face model.
        model: Loaded model instance.
        tokenizer: Loaded tokenizer instance.
        device: Device on which the model runs (cuda/cpu).
        max_length: Maximum length for generated text.
    """
    
    def __init__(
        self,
        model_name: str = "tiiuae/falcon-7b-instruct",
        device: Optional[str] = None,
        max_length: int = 512,
        max_new_tokens: int = 256
    ):
        """
        Initialize the LLMService with a specified model.
        
        Args:
            model_name: Name or path of the Hugging Face model.
                       Defaults to "tiiuae/falcon-7b-instruct".
                       For Arabic, consider "inceptionai/JAIS-13B-Chat" or
                       "inceptionai/JAIS-30B-Chat".
            device: Device to run the model on ('cuda', 'cpu', or None for auto).
                    If None, automatically selects the best available device.
            max_length: Maximum total sequence length (input + output).
            max_new_tokens: Maximum number of new tokens to generate.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the tokenizer and model from Hugging Face.
        
        This method is called during initialization to load the model and tokenizer.
        The model is loaded with appropriate settings for inference.
        """
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or self.model.device.type != self.device:
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LLM model '{self.model_name}': {str(e)}"
            ) from e
    
    def generate(self, prompt: str) -> str:
        """
        Generate a text response from the LLM given a prompt.
        
        This method takes a prompt string and returns the model's generated
        response. The generation uses the configured parameters for length
        and sampling.
        
        Args:
            prompt: Input text prompt for the model.
            
        Returns:
            Generated text response as a string.
            
        Raises:
            ValueError: If prompt is empty or invalid.
            RuntimeError: If text generation fails.
            
        Example:
            >>> service = LLMService()
            >>> response = service.generate("ما هو الذكاء الاصطناعي؟")
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - self.max_new_tokens
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text.strip()
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate text: {str(e)}"
            ) from e

