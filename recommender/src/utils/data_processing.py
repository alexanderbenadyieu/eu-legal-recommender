"""
Data processing utilities for the EU Legal Recommender system.

This module provides functions for processing and transforming data used in the recommender system.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data as a dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    file_path = Path(file_path)
    logger.debug(f"Loading JSON from {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the output file
        indent: Indentation level for pretty printing
        
    Raises:
        IOError: If the file cannot be written
    """
    file_path = Path(file_path)
    logger.debug(f"Saving JSON to {file_path}")
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error writing to {file_path}: {e}")
        raise


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def batch_process(items: List[Any], batch_size: int, process_func: callable, *args, **kwargs) -> List[Any]:
    """
    Process a list of items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to apply to each batch
        *args: Additional positional arguments for process_func
        **kwargs: Additional keyword arguments for process_func
        
    Returns:
        List of processed items
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch, *args, **kwargs)
        results.extend(batch_results)
    return results


def filter_dict_by_keys(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Filter a dictionary to include only specified keys.
    
    Args:
        d: Input dictionary
        keys: List of keys to include
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}


def merge_dicts(dicts: List[Dict[str, Any]], overwrite: bool = True) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        dicts: List of dictionaries to merge
        overwrite: Whether to overwrite existing keys
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if overwrite:
            result.update(d)
        else:
            # Only add keys that don't already exist
            for k, v in d.items():
                if k not in result:
                    result[k] = v
    return result
