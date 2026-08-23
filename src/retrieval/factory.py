from .base import Retriever
from .simple import SimpleRetriever
from .semantic import SemanticRetriever
from .hybrid import HybridRetriever
from .rerank import RerankRetriever

def get_retriever(strategy: str, top_k: int = 5, **kwargs) -> Retriever:
    """
    Factory function to instantiate retrieval strategies by name.
    
    WHY FACTORY PATTERN:
    - Decouples the client (e.g. an API endpoint or evaluation script) from the concrete classes.
    - Makes it trivial to swap out retrieval logic just by changing a string config.
    - Extensibility: Adding a new strategy only requires registering it in this dictionary.
    """
    strategies = {
        'simple': lambda: SimpleRetriever(top_k=top_k, **kwargs),
        'semantic': lambda: SemanticRetriever(top_k=top_k, **kwargs),
        'hybrid': lambda: HybridRetriever(top_k=top_k, **kwargs),
        'rerank': lambda: RerankRetriever(top_k=top_k, **kwargs),
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: '{strategy}'. Choose from {list(strategies.keys())}")
        
    return strategies[strategy]()
