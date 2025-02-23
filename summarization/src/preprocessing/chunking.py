import numpy as np

def calculate_compression_ratio(doc_length: int, model_context: int) -> float:
    return min(1.0, model_context / doc_length)

def dynamic_chunker(text: str, max_length: int=4096) -> list[str]:
    """Split documents using legal structure markers"""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]
