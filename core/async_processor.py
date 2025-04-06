import asyncio
import logging
import concurrent.futures
import functools
import time
from typing import Dict, List, Any, Callable, Tuple, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class AsyncProcessor:
    """
    Asynchronous processor for handling time-consuming manga processing tasks.
    Uses a thread pool to run multiple tasks concurrently.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the async processor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.loop = asyncio.new_event_loop()
        
    def run_in_thread(self, func, *args, **kwargs):
        """
        Run a function in a separate thread.
        
        Args:
            func: Function to run
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Future object
        """
        return self.executor.submit(func, *args, **kwargs)
    
    async def run_async(self, func, *args, **kwargs):
        """
        Run a function asynchronously.
        
        Args:
            func: Function to run
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            functools.partial(func, *args, **kwargs)
        )
    
    def process_batch(self, images: List[np.ndarray], processor_func: Callable, **kwargs) -> List[Any]:
        """
        Process a batch of images with the given function.
        
        Args:
            images: List of images to process
            processor_func: Function to process each image
            **kwargs: Additional arguments to pass to the processor function
            
        Returns:
            List of results
        """
        futures = []
        for image in images:
            future = self.run_in_thread(processor_func, image, **kwargs)
            futures.append(future)
            
        return [future.result() for future in futures]
    
    async def async_process_batch(self, images: List[np.ndarray], processor_func: Callable, **kwargs) -> List[Any]:
        """
        Process a batch of images asynchronously.
        
        Args:
            images: List of images to process
            processor_func: Function to process each image
            **kwargs: Additional arguments to pass to the processor function
            
        Returns:
            List of results
        """
        tasks = []
        for image in images:
            task = self.run_async(processor_func, image, **kwargs)
            tasks.append(task)
            
        return await asyncio.gather(*tasks)
    
    def run_pipeline_stages(self, pipeline_stages: List[Tuple[str, Callable, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run a series of pipeline stages sequentially, but with parallel processing within each stage.
        
        Args:
            pipeline_stages: List of (stage_name, function, kwargs) tuples representing pipeline stages
            
        Returns:
            Dictionary with results from each stage
        """
        results = {}
        timing = {}
        
        for stage_name, func, kwargs in pipeline_stages:
            start_time = time.time()
            result = func(**kwargs)
            elapsed = time.time() - start_time
            
            results[stage_name] = result
            timing[stage_name] = elapsed
            
            logger.info(f"Pipeline stage {stage_name} completed in {elapsed:.2f}s")
            
        return {
            'results': results,
            'timing': timing
        }
    
    async def run_async_pipeline(self, image: Union[np.ndarray, Image.Image], 
                              pipeline_stages: List[Tuple[str, Callable, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run a full processing pipeline asynchronously.
        
        Args:
            image: Input image
            pipeline_stages: List of (stage_name, function, kwargs) tuples representing pipeline stages
            
        Returns:
            Dictionary with results and timing information
        """
        results = {}
        timing = {}
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            np_image = np.array(image)
        else:
            np_image = image
            
        current_input = np_image
        
        for i, (stage_name, func, kwargs) in enumerate(pipeline_stages):
            start_time = time.time()
            
            # Add the current input to kwargs
            stage_kwargs = kwargs.copy()
            stage_kwargs['image'] = current_input
            
            # Run the stage
            try:
                result = await self.run_async(func, **stage_kwargs)
                
                # Store the result
                results[stage_name] = result
                
                # Use result as input for next stage if needed
                if i < len(pipeline_stages) - 1:
                    next_stage = pipeline_stages[i+1]
                    if len(next_stage) >= 3 and isinstance(next_stage[2], dict) and next_stage[2].get('use_previous_result', False):
                        current_input = result
                    
                elapsed = time.time() - start_time
                timing[stage_name] = elapsed
                logger.info(f"Pipeline stage {stage_name} completed in {elapsed:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in pipeline stage {stage_name}: {str(e)}")
                timing[stage_name] = time.time() - start_time
                results[stage_name] = None
        
        return {
            'results': results,
            'timing': timing
        }
    
    def close(self):
        """Close the thread pool executor."""
        self.executor.shutdown()


# Singleton instance for global use
async_processor = AsyncProcessor()


async def process_images_concurrently(images: List[np.ndarray], 
                                   processor_func: Callable, 
                                   **kwargs) -> List[Any]:
    """
    Process multiple images concurrently.
    
    Args:
        images: List of images to process
        processor_func: Function to process each image
        **kwargs: Additional arguments to pass to the processor function
        
    Returns:
        List of results
    """
    return await async_processor.async_process_batch(images, processor_func, **kwargs)


def run_in_background(func: Callable) -> Callable:
    """
    Decorator to run a function in a background thread.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that returns a Future
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return async_processor.run_in_thread(func, *args, **kwargs)
    return wrapper