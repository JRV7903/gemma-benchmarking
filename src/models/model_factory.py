import logging
from typing import Dict, Optional, Union

from .gemma_loader import GemmaModel
from .mistral_loader import MistralModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating model instances based on configuration.
    """

    @staticmethod
    def create_model(
        model_config: Dict,
        device: str = "cuda",
        precision: str = "fp16"
    ) -> Union[GemmaModel, MistralModel]:
        """
        Create a model instance based on the provided configuration.

        Args:
            model_config: Model configuration dictionary
            device: Device to run the model on ('cuda' or 'cpu')
            precision: Model precision ('fp16', 'fp32', or 'int8')

        Returns:
            An instance of a model class (GemmaModel or MistralModel)

        Raises:
            ValueError: If the model_type is not supported
        """
        model_type = model_config.get("model_type", "").lower()
        model_name = model_config.get("model_name_or_path")
        max_length = model_config.get("max_length", 2048)

        logger.info(f"Creating model of type '{model_type}': {model_name}")

        if model_type == "gemma":
            return GemmaModel(
                model_name_or_path=model_name,
                device=device,
                precision=precision,
                max_length=max_length,
            )
        elif model_type == "mistral":
            return MistralModel(
                model_name_or_path=model_name,
                device=device,
                precision=precision,
                max_length=max_length,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def get_batch_size_for_model(model_config: Dict) -> int:
        """Get the recommended batch size for a model."""
        return model_config.get("batch_size", 1)
