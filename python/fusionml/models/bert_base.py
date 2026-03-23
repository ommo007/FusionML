"""
BERT-base End-to-End Inference
==============================

This module loads pretrained HuggingFace BERT weights and converts the
model architecture into a FusionML PipelineScheduler configuration.
"""

import numpy as np
from typing import Dict, Any

from fusionml._metal.pipeline_scheduler import PipelineScheduler, LayerConfig

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

def get_pytorch_bert():
    import torch
    from transformers import BertModel
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return model

# ============================================================================
# MLX Custom Execution Functions
# ============================================================================

def mlx_layer_norm(x, weight, bias, eps=1e-12):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / mx.sqrt(var + eps) + bias

def _mlx_bert_embeddings(w_mx: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """x is input_ids (B, S). Output is (B, S, H)"""
    # Simply using precomputed embedding lookup? 
    # For actual pipeline we'd do lookup, but we might receive integer tokens.
    # To keep the pipeline purely tensor-in tensor-out, we assume x is already
    # continuous embeddings, or we do the lookup.
    # Let's assume x is input_ids (int32)
    x_mx = mx.array(x)
    
    # Word embeddings
    words = w_mx["word_emb"][x_mx]
    
    # Positional embeddings (assume seq_len is x.shape[1])
    seq_len = x.shape[1]
    positions = mx.arange(seq_len)[None, :]
    pos_emb = w_mx["pos_emb"][positions]
    
    # Token type embeddings (assume 0)
    tok_emb = w_mx["tok_emb"][mx.zeros_like(x_mx)]
    
    embeddings = words + pos_emb + tok_emb
    
    embeddings = mlx_layer_norm(embeddings, w_mx["ln_w"], w_mx["ln_b"])
    # No dropout for inference
    
    mx.eval(embeddings)
    return np.array(embeddings)

def _mlx_bert_layer(w_mx: Dict[str, Any], x: np.ndarray, num_heads: int) -> np.ndarray:
    """Single Transformer Encoder block"""
    x_mx = mx.array(x) # (B, S, H)
    B, S, H = x.shape
    head_size = H // num_heads
    
    # Self-attention
    q = x_mx @ w_mx["q_w"] + w_mx["q_b"]    
    k = x_mx @ w_mx["k_w"] + w_mx["k_b"]
    v = x_mx @ w_mx["v_w"] + w_mx["v_b"]
    
    q = q.reshape(B, S, num_heads, head_size).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, num_heads, head_size).transpose(0, 2, 3, 1)
    v = v.reshape(B, S, num_heads, head_size).transpose(0, 2, 1, 3)
    
    # Attention scores
    scores = (q @ k) / mx.sqrt(float(head_size))
    # We assume no attention mask for simplistic inference pipeline benchmarking
    probs = mx.softmax(scores, axis=-1)
    
    context = (probs @ v).transpose(0, 2, 1, 3).reshape(B, S, H)
    
    # Output projection
    att_out = context @ w_mx["out_w"] + w_mx["out_b"]
    
    # Residual + LayerNorm
    h1 = mlx_layer_norm(x_mx + att_out, w_mx["ln1_w"], w_mx["ln1_b"])
    
    # FFN
    inter = h1 @ w_mx["inter_w"] + w_mx["inter_b"]
    # GELU
    inter = inter * 0.5 * (1.0 + mx.erf(inter / mx.sqrt(2.0)))
    
    out = inter @ w_mx["ffn_w"] + w_mx["ffn_b"]
    
    # Residual + LayerNorm
    h2 = mlx_layer_norm(h1 + out, w_mx["ln2_w"], w_mx["ln2_b"])
    
    mx.eval(h2)
    return np.array(h2)

def _mlx_bert_pooler(w_mx: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Pooler layer (takes first token)"""
    x_mx = mx.array(x)
    first_token = x_mx[:, 0]
    out = first_token @ w_mx["dense_w"] + w_mx["dense_b"]
    out = mx.tanh(out)
    mx.eval(out)
    return np.array(out)


# ============================================================================
# PyTorch/NumPy Fallback CPU Functions
# ============================================================================

def _cpu_torch_module(w: Dict[str, Any], x: np.ndarray, module, is_input_ids=False) -> np.ndarray:
    import torch
    with torch.no_grad():
        if is_input_ids:
            x_t = torch.tensor(x, dtype=torch.long)
        else:
            x_t = torch.from_numpy(x)
        out = module(x_t)
        if isinstance(out, tuple):
            out = out[0]
        return out.numpy()

# ============================================================================
# WEIGHT EXTRACTION
# ============================================================================

def build_fusionml_bert_pipeline(seq_len: int = 128) -> PipelineScheduler:
    """
    Load HuggingFace BERT-base weights and build a PipelineScheduler.
    """
    import torch
    import torch.nn as nn
    from transformers.models.bert.modeling_bert import BertLayer
    
    print("Loading HuggingFace BERT-base weights...")
    pt_model = get_pytorch_bert()
    
    sched = PipelineScheduler(verbose=False)
    
    def prep_weights(w_dict):
        out = {}
        for k, v in w_dict.items():
            if isinstance(v, (int, float)):
                out[k] = v
            else:
                out[k] = v.detach().numpy()
        return out

    # 1. EMBEDDINGS
    class BertEmbeddingsWrapper(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb = emb
        def forward(self, input_ids):
            # We don't use token_type_ids or position_ids explicitly
            return self.emb(input_ids)

    emb_wrapper = BertEmbeddingsWrapper(pt_model.embeddings)
    w_emb = {
        "word_emb": pt_model.embeddings.word_embeddings.weight,
        "pos_emb": pt_model.embeddings.position_embeddings.weight,
        "tok_emb": pt_model.embeddings.token_type_embeddings.weight,
        "ln_w": pt_model.embeddings.LayerNorm.weight,
        "ln_b": pt_model.embeddings.LayerNorm.bias,
    }
    
    sched.add_layer(LayerConfig(
        name="embeddings",
        op_type="custom",
        input_shape=(1, seq_len),
        weights=prep_weights(w_emb),
        params={
            # "torch_module": emb_wrapper, # Disabled for ANE, CoreML compile fails on NLP embeddings
            "mlx_fn": lambda w, x: _mlx_bert_embeddings(w, x),
            "cpu_fn": lambda w, x: (_ for _ in ()).throw(NotImplementedError("CPU not supported for BERT"))
        }
    ))
    
    current_shape = (1, seq_len, 768)
    
    # 2. TRANSFORMER LAYERS
    class BertLayerWrapper(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
        def forward(self, hidden_states):
            # Only return the hidden states
            return self.layer(hidden_states)[0]
            
    for i, layer in enumerate(pt_model.encoder.layer):
        w_layer = {
            "q_w": layer.attention.self.query.weight.T,
            "q_b": layer.attention.self.query.bias,
            "k_w": layer.attention.self.key.weight.T,
            "k_b": layer.attention.self.key.bias,
            "v_w": layer.attention.self.value.weight.T,
            "v_b": layer.attention.self.value.bias,
            "out_w": layer.attention.output.dense.weight.T,
            "out_b": layer.attention.output.dense.bias,
            "ln1_w": layer.attention.output.LayerNorm.weight,
            "ln1_b": layer.attention.output.LayerNorm.bias,
            "inter_w": layer.intermediate.dense.weight.T,
            "inter_b": layer.intermediate.dense.bias,
            "ffn_w": layer.output.dense.weight.T,
            "ffn_b": layer.output.dense.bias,
            "ln2_w": layer.output.LayerNorm.weight,
            "ln2_b": layer.output.LayerNorm.bias,
        }
        
        wrapper = BertLayerWrapper(layer)
        
        # Capture lexical closure
        def make_cpu_fn(mod_arg=wrapper):
            return lambda w, x: _cpu_torch_module(w, x, mod_arg)
        
        # We need num_heads in MLX function
        num_heads = layer.attention.self.num_attention_heads
        
        def make_mlx_fn(heads=num_heads):
            # MLX fn takes (w_mx, x). We pass num_heads by capturing it.
            # However _mlx_bert_layer expects it in w_mx. We can just add it back.
            # But w_mx is built by PipelineScheduler. 
            # We will instead wrap the Custom mlx_fn to inject it.
            return lambda w, x: _mlx_bert_layer(w, x, heads)

        sched.add_layer(LayerConfig(
            name=f"encoder_layer_{i}",
            op_type="custom",
            input_shape=current_shape,
            weights=prep_weights(w_layer),
            params={
                # "torch_module": wrapper, # Disabled for ANE, CoreML isn't great for split encoder layers
                "mlx_fn": make_mlx_fn(),
                "cpu_fn": lambda w, x: (_ for _ in ()).throw(NotImplementedError("CPU not supported for BERT"))
            }
        ))

    # 3. POOLER
    w_pooler = {
        "dense_w": pt_model.pooler.dense.weight.T,
        "dense_b": pt_model.pooler.dense.bias
    }
    
    sched.add_layer(LayerConfig(
        name="pooler",
        op_type="custom",
        input_shape=current_shape,
        weights=prep_weights(w_pooler),
        params={
            # "torch_module": pt_model.pooler,
            "mlx_fn": _mlx_bert_pooler,
            "cpu_fn": lambda w, x: (_ for _ in ()).throw(NotImplementedError("CPU not supported for BERT"))
        }
    ))
    
    return sched

__all__ = ["build_fusionml_bert_pipeline", "get_pytorch_bert"]
