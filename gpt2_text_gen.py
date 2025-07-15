import numpy as np

def load_encoder_and_model(model_size="124M", models_dir="models"):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict ={"hello": 1, "world": 2, "<UNK>": 0}
            
        def encode(self, text):
            tokens = text.strip().split()
            return [self.encoder_dict.get(tok, 0) for tok in tokens]
        
        def decode(self, token_ids):
            rev = {v:k for k, v in self.encoder_dict.items()}
            return " ".join(rev.get(i, "UNK") for i in token_ids)
    
    hyperparams = {
        "context_size": 1024,
        "embedding_dim": 10,
        "num_heads": 2,
        "num_layers": 1
    }
    
    weights = {
        "token_embedding": np.random.rand(3, hyperparams["embedding_dim"]),
        "position_embedding": np.random.randn(hyperparams["context_size"], hyperparams["embedding_dim"]),
        "layers": [{
            "ln_before_attn": {"gain": np.ones(hyperparams["embedding_dim"]), "bias": np.zeros(hyperparams["embedding_dim"])},
            "attention": {
                "qkv_projection": np.random.randn(hyperparams["embedding_dim"], 3*hyperparams["embedding_dim"]),
                "output_projection": np.random.randn(hyperparams["embedding_dim"], hyperparams["embedding_dim"]),            
            },
           "ln_before_mlp": {"gain": np.ones(hyperparams["embedding_dim"]), "bias": np.zeros(hyperparams["embedding_dim"])},
           "mlp": {
               "hidden_projection": np.random.randn(hyperparams["embedding_dim"], 4*hyperparams["embedding_dim"]),
               "output_projection": np.random.randn(4*hyperparams["embedding_dim"], hyperparams["embedding_dim"]),
             }
        }],
        "ln_final": {"gain": np.ones(hyperparams["embedding_dim"]), "bias": np.zeros(hyperparams["embedding_dim"])}
        
    }
    
    return DummyBPE(), hyperparams, weights 

def layer_norm(x, gain, bias, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    sigma = np.sqrt(((x - mu)**2).mean(-1, keepdims = True) + eps)
    return gain * (x - mu) / sigma + bias

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*(x**3))))

def split_heads(x, num_heads):
    seq_len, dim = x.shape
    head_dim = dim // num_heads
    return x.reshape(seq_len, num_heads, head_dim). transpose(1, 0, 2)

def merge_heads(x):
    num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, num_heads*head_dim)

def scaled_causal_self_attention(hidden, attn_wts, num_heads):
    qkv = hidden @ attn_wts["qkv_projection"]
    q, k, v = np.split(qkv, 3, axis = 1)
    
    qh = split_heads(q, num_heads); kh = split_heads(k, num_heads); vh = split_heads(v, num_heads)
    scores = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(q.shape[1]//num_heads)
    mask = np.triu(np.full(scores.shape[-2:], -1e9), k=1)
    scores = scores + mask
    probs = np.exp(scores - scores.max(-1, keepdims = True))
    probs /= probs.sum(-1, keepdims = True)
    context = probs @ vh
    concat = merge_heads(context)
    return concat @ attn_wts["output_projection"]

def feed_forward(hidden, mlp_wts):
    hidden_proj = gelu(hidden @ mlp_wts["hidden_projection"])
    return hidden_proj @ mlp_wts["output_projection"]

def transformer_layer(hidden, layer_wts, num_heads):
    normed = layer_norm(hidden, **layer_wts["ln_before_attn"])
    attn = scaled_causal_self_attention(normed, layer_wts["attention"], num_heads)
    hidden = hidden + attn 
    
    normed2 = layer_norm(hidden, **layer_wts["ln_before_mlp"])
    mlp_out = feed_forward(normed2, layer_wts["mlp"])
    return hidden + mlp_out

def generate_text(prompt, max_new_tokens = 40):
    tokenizer, hparams, model_wts = load_encoder_and_model()
    token_ids = tokenizer.encode(prompt)
    
    for _ in range(max_new_tokens):
        context = token_ids[-hparams["context_size"]:]
        seq_len = len(context)
        
        tok_embeds = model_wts["token_embedding"][context]
        pos_embeds = model_wts["position_embedding"][:seq_len]
        hidden = tok_embeds + pos_embeds
        
        for lyr in model_wts["layers"]:
            hidden = transformer_layer(hidden, lyr, hparams["num_heads"])
            
        hidden = layer_norm(hidden, **model_wts["ln_final"])
        logits = hidden @ model_wts["token_embedding"].T
        
        next_id = int(logits.argmax())
        token_ids.append(next_id)
        
    return tokenizer.decode(token_ids)
        
if __name__ == "__main__":
    out = generate_text("hello world", max_new_tokens=3)
    print("→", out)
        
        
    
