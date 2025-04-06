import os
import logging
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages downloading and caching of large model files from external sources.
    """
    
    def __init__(self, models_dir='models/weights'):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry = {}
        
    def register_model(self, model_id, url=None, md5=None, local_path=None, required=False):
        """
        Register a model with the manager.
        
        Args:
            model_id: Unique identifier for the model
            url: URL to download the model from
            md5: Expected MD5 hash for verification (optional)
            local_path: Local path to the model if already available
            required: Whether the model is required for system operation
            
        Returns:
            Path to the model file (may not exist yet if download is needed)
        """
        model_path = self.models_dir / f"{model_id.replace('/', '_')}"
        
        self.model_registry[model_id] = {
            'url': url,
            'md5': md5,
            'path': model_path,
            'downloaded': model_path.exists(),
            'local_path': local_path,
            'required': required
        }
        
        return model_path
    
    def get_model_path(self, model_id):
        """
        Get the path to a registered model file.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to the model file or None if not found
        """
        if model_id not in self.model_registry:
            return None
            
        if self.model_registry[model_id]['local_path']:
            return self.model_registry[model_id]['local_path']
            
        model_path = self.model_registry[model_id]['path']
        if model_path.exists():
            return str(model_path)
            
        return None
    
    def download_model(self, model_id, force=False):
        """
        Download a model from its registered URL.
        
        Args:
            model_id: Model identifier
            force: Whether to force download even if file exists
            
        Returns:
            Path to the downloaded model file or None if download failed
        """
        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not registered")
            return None
            
        model_info = self.model_registry[model_id]
        model_path = model_info['path']
        
        # Check if model already exists
        if model_path.exists() and not force:
            logger.info(f"Model {model_id} already exists at {model_path}")
            return str(model_path)
            
        # Check if URL provided
        if not model_info['url']:
            logger.error(f"No URL provided for model {model_id}")
            return None
            
        # Download the model
        try:
            logger.info(f"Downloading model {model_id} from {model_info['url']}")
            
            # Stream the file download with progress bar
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            
            temp_path = model_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f, tqdm(
                desc=f"Downloading {model_id}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    progress_bar.update(size)
            
            # Verify MD5 if provided
            if model_info['md5']:
                md5_hash = hashlib.md5()
                with open(temp_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        md5_hash.update(chunk)
                        
                calculated_md5 = md5_hash.hexdigest()
                if calculated_md5 != model_info['md5']:
                    logger.error(f"MD5 mismatch for {model_id}: expected {model_info['md5']}, got {calculated_md5}")
                    temp_path.unlink()
                    return None
            
            # Rename temp file to final filename
            temp_path.rename(model_path)
            self.model_registry[model_id]['downloaded'] = True
            
            logger.info(f"Successfully downloaded {model_id} to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            return None
    
    def download_all_required(self):
        """
        Download all models marked as required.
        
        Returns:
            Dictionary mapping model IDs to download success status
        """
        results = {}
        for model_id, model_info in self.model_registry.items():
            if model_info['required']:
                path = self.download_model(model_id)
                results[model_id] = path is not None
                
        return results
    
    def clear_cache(self, model_id=None):
        """
        Clear cached model files.
        
        Args:
            model_id: Specific model to clear or None for all
            
        Returns:
            List of removed model files
        """
        removed = []
        
        if model_id:
            if model_id not in self.model_registry:
                return removed
                
            model_path = self.model_registry[model_id]['path']
            if model_path.exists():
                model_path.unlink()
                self.model_registry[model_id]['downloaded'] = False
                removed.append(model_id)
        else:
            for model_id, model_info in self.model_registry.items():
                model_path = model_info['path']
                if model_path.exists():
                    model_path.unlink()
                    self.model_registry[model_id]['downloaded'] = False
                    removed.append(model_id)
                    
        return removed