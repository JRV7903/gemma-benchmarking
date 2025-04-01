import logging
import torch
import os
from typing import Dict, List, Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class GemmaModel:
    """
    Wrapper for loading and running Gemma models from Hugging Face.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        precision: str = "fp16",
        max_length: int = 2048,
    ):
        """
        Initialize the Gemma model.

        Args:
            model_name_or_path: HuggingFace model name or path
            device: Device to run the model on ('cuda' or 'cpu')
            precision: Model precision ('fp16', 'fp32', or 'int8')
            max_length: Maximum sequence length
        """
        self.model_name = model_name_or_path
        self.device = device
        self.precision = precision
        self.max_length = max_length
        
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            logger.warning("HF_TOKEN environment variable not set. Authentication may fail for Gemma models.")

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Error loading model {model_name_or_path}: {str(e)}")
            raise

    def _load_model(self):
        """Load the model and tokenizer with appropriate settings."""
        logger.info(f"Loading Gemma model: {self.model_name}")

        dtype = torch.float32
        if self.precision == "fp16" and self.device == "cuda":
            dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
            token=self.hf_token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        logger.info(f"Successfully loaded {self.model_name}")

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_full_text: bool = False,
    ) -> str:
        """
        Generate a response from the model for a given prompt.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 means deterministic)
            top_p: Top-p sampling parameter
            return_full_text: Whether to return the full text or just the response

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=(temperature > 0),
            )

        if return_full_text:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            input_length = inputs.input_ids.shape[1]
            return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            batch_size: Batch size for processing
            max_new_tokens: Maximum number of new tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            List of generated responses
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - max_new_tokens
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=(temperature > 0),
                )

            input_lengths = [len(ids) for ids in inputs.input_ids]
            for j, output in enumerate(outputs):
                gen_text = self.tokenizer.decode(output[input_lengths[j]:], skip_special_tokens=True)
                results.append(gen_text.strip())

        return results

    def extract_answer_from_response(self, response: str) -> Optional[str]:
        """
        Extract a multiple-choice answer (A, B, C, D) from a model response.

        Args:
            response: The generated model response

        Returns:
            The extracted answer (A, B, C, or D) or None if not found
        """
        response = response.strip().upper()

        if response and response[0] in "ABCD":
            return response[0]

        for pattern in ["ANSWER: ", "ANSWER IS ", "ANSWER:"]:
            if pattern in response:
                answer_part = response.split(pattern)[1].strip()
                if answer_part and answer_part[0] in "ABCD":
                    return answer_part[0]

        for letter in "ABCD":
            if letter in response:
                return letter

        return None
