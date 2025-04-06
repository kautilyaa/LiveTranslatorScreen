import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the manga processing pipeline"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from environment variables or config file
        
        Args:
            config_file: Path to optional JSON/YAML config file
        """
        # Default settings
        self.default_ocr_provider = "tesseract"
        self.default_inpainting_model = "lama"
        self.default_translator = "huggingface"
        self.default_source_language = "ja"
        self.default_target_language = "en"
        
        # API Keys
        self.azure_api_key = os.getenv("AZURE_API_KEY", "")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT", "")
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY", "")
        self.aws_secret_key = os.getenv("AWS_SECRET_KEY", "")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        # Model paths
        self.yolo_model_path = os.getenv("YOLO_MODEL_PATH", "models/weights/yolo_text_detection.pt")
        self.lama_model_path = os.getenv("LAMA_MODEL_PATH", "models/weights/lama.pth")
        self.sd_model_path = os.getenv("SD_MODEL_PATH", "models/weights/stable_diffusion.pth")
        
        # Performance settings
        self.use_gpu = os.getenv("USE_GPU", "False").lower() in ("true", "1", "t")
        self.batch_size = int(os.getenv("BATCH_SIZE", "1"))
        
        # Override from config file if provided
        if config_file:
            self._load_from_file(config_file)
        
        # Log configuration
        logger.info(f"Initialized configuration: OCR={self.default_ocr_provider}, "
                   f"Inpainting={self.default_inpainting_model}, "
                   f"Translator={self.default_translator}, "
                   f"Languages={self.default_source_language}->{self.default_target_language}")
        
        if self.use_gpu:
            logger.info("GPU enabled for model inference")
        else:
            logger.info("Using CPU for model inference")
    
    def _load_from_file(self, config_file: str):
        """Load configuration from file"""
        import json
        import yaml
        
        try:
            if config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                with open(config_file, 'r') as f:
                    import yaml
                    config_data = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_file}")
                return
            
            # Update configuration from file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info(f"Loaded configuration from {config_file}")
        
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "default_ocr_provider": self.default_ocr_provider,
            "default_inpainting_model": self.default_inpainting_model,
            "default_translator": self.default_translator,
            "default_source_language": self.default_source_language,
            "default_target_language": self.default_target_language,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size
        }
