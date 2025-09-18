from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel(object):
    _embedder = SentenceTransformer('all-MiniLM-L6-v2')

    _tokenizer = None
    _model = None    
    _action = _embedder.encode
    _device="cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _model_embed(texts,device="cuda" if torch.cuda.is_available() else "cpu"):
        inputs = EmbeddingModel._tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            #outputs = EmbeddingModel._model(**inputs)
            #hidden_states = outputs.last_hidden_state
            #embeddings = hidden_states.mean(dim=1)


            outputs = EmbeddingModel._model(**inputs)
            hidden_states = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
            embeddings = (hidden_states * mask).sum(1) / mask.sum(1)

        return embeddings.cpu().numpy()
    
    @staticmethod
    def set_model(model_name,device="cuda" if torch.cuda.is_available() else "cpu"):
        EmbeddingModel._device = device
        try:

            EmbeddingModel._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if EmbeddingModel._tokenizer.pad_token is None:
                EmbeddingModel._tokenizer.pad_token = EmbeddingModel._tokenizer.eos_token

            #EmbeddingModel._tokenizer = AutoTokenizer.from_pretrained(model_name)
            EmbeddingModel._model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                EmbeddingModel._model = EmbeddingModel._model.to(torch.device(device))

            EmbeddingModel._action = EmbeddingModel._model_embed
        except:
            EmbeddingModel._embedder = SentenceTransformer(model_name, device=device)
            EmbeddingModel._action = EmbeddingModel._embedder.encode

    @staticmethod
    def embed(input):
        return EmbeddingModel._action(input,device=EmbeddingModel._device)

    @staticmethod
    def get_embeddings(target):    
        res = [EmbeddingModel.embed(t) for t in target]
        return res