"""
Vision Transformer (ViT-B/16) End-to-End Inference
==================================================

This module loads pretrained PyTorch ViT weights and converts the
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

def get_pytorch_vit():
    import torch
    import torchvision.models as models
    from torchvision.models.vision_transformer import ViT_B_16_Weights
    
    # Needs to be called with pretrained=True or weights=...
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.eval()
    return model

# ============================================================================
# MLX Custom Execution Functions
# ============================================================================

def mlx_conv2d(x, weight, stride=16, padding=0):
    """x is NHWC, weight is OHWI"""
    return mx.conv2d(x, weight, stride=stride, padding=padding)

def mlx_layer_norm(x, weight, bias, eps=1e-6):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / mx.sqrt(var + eps) + bias

def _mlx_vit_patch_embed(w_mx: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """x is (B, C, H, W). Conv2d -> flatten -> transpose -> add class_token/pos_emb"""
    # NCHW -> NHWC
    x_mx = mx.array(x.transpose(0, 2, 3, 1))
    
    # Conv Projection
    out = mlx_conv2d(x_mx, w_mx["conv_w"], stride=16, padding=0)
    
    # out is (B, 14, 14, 768) -> (B, 196, 768)
    B, H, W, C = out.shape
    out = out.reshape(B, H * W, C)
    
    # Add class token (1, 1, 768) to (B, 196, 768) -> (B, 197, 768)
    # Broadcast class_token to B
    cls_token = mx.broadcast_to(w_mx["cls_token"], (B, 1, C))
    out = mx.concatenate([cls_token, out], axis=1)
    
    # Add positional embedding
    out = out + w_mx["pos_emb"]
    
    mx.eval(out)
    return np.array(out)


def _mlx_vit_layer(w_mx: Dict[str, Any], x: np.ndarray, num_heads: int) -> np.ndarray:
    """Single ViT Encoder block"""
    x_mx = mx.array(x) # (B, S, H)
    B, S, H = x.shape
    head_size = H // num_heads
    
    # LayerNorm 1
    h1 = mlx_layer_norm(x_mx, w_mx["ln1_w"], w_mx["ln1_b"])
    
    # Self-attention: QKV projection
    # PyTorch ViT projects Q, K, V separately or uses merged linear, let's assume separate based on how we extract it
    q = h1 @ w_mx["q_w"] + w_mx["q_b"]
    k = h1 @ w_mx["k_w"] + w_mx["k_b"]
    v = h1 @ w_mx["v_w"] + w_mx["v_b"]
    
    q = q.reshape(B, S, num_heads, head_size).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, num_heads, head_size).transpose(0, 2, 3, 1)
    v = v.reshape(B, S, num_heads, head_size).transpose(0, 2, 1, 3)
    
    # Attention scores
    scores = (q @ k) / mx.sqrt(float(head_size))
    probs = mx.softmax(scores, axis=-1)
    
    context = (probs @ v).transpose(0, 2, 1, 3).reshape(B, S, H)
    
    # Output projection
    att_out = context @ w_mx["out_w"] + w_mx["out_b"]
    
    # Residual 1
    x1 = x_mx + att_out
    
    # LayerNorm 2
    h2 = mlx_layer_norm(x1, w_mx["ln2_w"], w_mx["ln2_b"])
    
    # FFN
    inter = h2 @ w_mx["mlp1_w"] + w_mx["mlp1_b"]
    # GELU
    inter = inter * 0.5 * (1.0 + mx.erf(inter / mx.sqrt(2.0)))
    
    out = inter @ w_mx["mlp2_w"] + w_mx["mlp2_b"]
    
    # Residual 2
    x2 = x1 + out
    
    mx.eval(x2)
    return np.array(x2)

def _mlx_vit_head(w_mx: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Extract class token -> layer norm -> linear"""
    x_mx = mx.array(x)
    
    # Extract class token (index 0)
    cls_token = x_mx[:, 0]
    
    # Final LayerNorm before head (ViT pattern)
    # Actually torchvision does this before the heads layer
    h = mlx_layer_norm(cls_token, w_mx["ln_w"], w_mx["ln_b"])
    
    # Linear classification head
    out = h @ w_mx["head_w"] + w_mx["head_b"]
    
    mx.eval(out)
    return np.array(out)


# ============================================================================
# PyTorch/NumPy Fallback CPU Functions
# ============================================================================

def _cpu_torch_module(w: Dict[str, Any], x: np.ndarray, module) -> np.ndarray:
    import torch
    with torch.no_grad():
        x_t = torch.from_numpy(x)
        out = module(x_t)
        if isinstance(out, tuple):
            out = out[0]
        return out.numpy()

# ============================================================================
# WEIGHT EXTRACTION
# ============================================================================

def build_fusionml_vit_pipeline() -> PipelineScheduler:
    """
    Load PyTorch pretrained weights and build a PipelineScheduler for ViT-B/16.
    """
    import torch
    import torch.nn as nn
    
    print("Loading PyTorch ViT weights...")
    pt_model = get_pytorch_vit()
    
    sched = PipelineScheduler(verbose=False)
    
    def prep_weights(w_dict):
        out = {}
        for k, v in w_dict.items():
            out[k] = v.detach().numpy()
        return out

    # 1. PATCH EMBEDDINGS (Conv2d + Class Token + Pos Embedding)
    w_embed = {
        "conv_w": pt_model.conv_proj.weight.permute(0, 2, 3, 1), # NCHW -> OHWI for mlx
        "cls_token": pt_model.class_token,
        "pos_emb": pt_model.encoder.pos_embedding
    }
    
    class PatchEmbedWrapper(nn.Module):
        def __init__(self, conv, cls_token, pos_embed):
            super().__init__()
            self.conv = conv
            self.cls_token = cls_token
            self.pos_embed = pos_embed
        def forward(self, x):
            x = self.conv(x)
            x = x.flatten(2).transpose(1, 2)
            n = x.shape[0]
            batch_class_token = self.cls_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            return x + self.pos_embed

    embed_module = PatchEmbedWrapper(pt_model.conv_proj, pt_model.class_token, pt_model.encoder.pos_embedding)

    sched.add_layer(LayerConfig(
        name="patch_embed",
        op_type="custom",
        input_shape=(1, 3, 224, 224),
        weights=prep_weights(w_embed),
        params={
            "torch_module": embed_module,
            "mlx_fn": _mlx_vit_patch_embed,
            "cpu_fn": lambda w, x: _cpu_torch_module(w, x, embed_module)
        }
    ))
    
    current_shape = (1, 197, 768)
    
    # 2. ENCODER BLOCKS
    for i, block in enumerate(pt_model.encoder.layers):
        w_block = {
            "ln1_w": block.ln_1.weight, "ln1_b": block.ln_1.bias,
            # torchvision's MultiheadAttention uses packed in_proj_weight. We split it:
            "q_w": block.self_attention.in_proj_weight[:768, :].T,
            "k_w": block.self_attention.in_proj_weight[768:1536, :].T,
            "v_w": block.self_attention.in_proj_weight[1536:, :].T,
            "q_b": block.self_attention.in_proj_bias[:768],
            "k_b": block.self_attention.in_proj_bias[768:1536],
            "v_b": block.self_attention.in_proj_bias[1536:],
            
            "out_w": block.self_attention.out_proj.weight.T,
            "out_b": block.self_attention.out_proj.bias,
            
            "ln2_w": block.ln_2.weight, "ln2_b": block.ln_2.bias,
            
            "mlp1_w": block.mlp[0].weight.T, "mlp1_b": block.mlp[0].bias,
            "mlp2_w": block.mlp[3].weight.T, "mlp2_b": block.mlp[3].bias,
        }
        
        # We need num_heads
        num_heads = block.self_attention.num_heads
        
        def make_mlx_fn(h=num_heads):
            return lambda w_mx, x: _mlx_vit_layer(w_mx, x, h)
            
        def make_cpu_fn(m=block):
            return lambda w, x: _cpu_torch_module(w, x, m)

        sched.add_layer(LayerConfig(
            name=f"encoder_{i}",
            op_type="custom",
            input_shape=current_shape,
            weights=prep_weights(w_block),
            params={
                "torch_module": block,
                "mlx_fn": make_mlx_fn(),
                "cpu_fn": make_cpu_fn()
            }
        ))
        
    # 3. CLASSIFICATION HEAD
    w_head = {
        "ln_w": pt_model.encoder.ln.weight,
        "ln_b": pt_model.encoder.ln.bias,
        "head_w": pt_model.heads[0].weight.T,
        "head_b": pt_model.heads[0].bias
    }
    
    class HeadWrapper(nn.Module):
        def __init__(self, ln, head):
            super().__init__()
            self.ln = ln
            self.head = head
        def forward(self, x):
            x = x[:, 0]
            x = self.ln(x)
            return self.head(x)
            
    head_module = HeadWrapper(pt_model.encoder.ln, pt_model.heads.head)
    
    sched.add_layer(LayerConfig(
        name="head",
        op_type="custom",
        input_shape=current_shape,
        weights=prep_weights(w_head),
        params={
            "torch_module": head_module,
            "mlx_fn": _mlx_vit_head,
            "cpu_fn": lambda w, x: _cpu_torch_module(w, x, head_module)
        }
    ))

    return sched

__all__ = ["build_fusionml_vit_pipeline", "get_pytorch_vit"]
