import ollama
from tqdm import tqdm

class OllamaEmbedder:
    def __init__(self, model='nomic-embed-text', batch_size=32):
        """
        Embeds text using Ollama.
        WHY: batch_size=32 trade-off: Larger batch sizes reduce the number of API calls 
        (improving throughput), but smaller batch sizes reduce peak RAM usage. 32 is a 
        sweet spot for avoiding OOM errors on 8GB machines while remaining efficient.
        """
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of texts in batches.
        RAM rule: ensure embedder is the only model loaded in VRAM.
        """
        embeddings = []
        
        # Batch processing with tqdm progress bar
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch = texts[i:i + self.batch_size]
            
            # WHY: keep_alive is handled by Ollama, so the model stays loaded for 
            # the duration of the batch processing. We do NOT explicitly unload 
            # after each batch; the next system call (e.g., LLM generation) will 
            # handle eviction via OllamaModelManager.
            
            # Since ollama 0.4.x, embed supports list of strings.
            response = ollama.embed(model=self.model, input=batch)
            
            # The 'embeddings' key contains a list of vectors corresponding to the input
            if 'embeddings' in response:
                 embeddings.extend(response['embeddings'])
            else:
                 # Fallback if list embedding behavior differs
                 for t in batch:
                     single_res = ollama.embed(model=self.model, input=t)
                     embeddings.append(single_res['embeddings'][0])
                     
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """
        Embeds a single piece of text.
        WHY: Used for query embedding at retrieval time, optimizing for latency.
        """
        response = ollama.embed(model=self.model, input=text)
        # Returns list of floats
        return response['embeddings'][0]
