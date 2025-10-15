import gc
from neuronx_distributed_inference.models.config import (
    InferenceConfig
)

def convert_hf_to_neuron_state_dict(state_dict: dict, cfg: InferenceConfig) -> dict:
    """Convert HF Qwen2-VL weights to Neuron format.
    
    This function handles both the text model weights
    and the vision model weights, maintaining the structure expected by the NeuronQwen2VLForCausalLM
    model. The conversion follows the architecture split:
    
    1. Vision backbone weights (under "visual.*")
    2. Text model weights (attention, MLP, embedding, etc.)
    
    Args:
        state_dict: HF state dictionary with Qwen2-VL weights
        cfg: Neuron inference configuration
        
    Returns:
        Converted state dictionary compatible with NeuronQwen2VLForCausalLM
    """
    text_config = cfg.text_config
    vision_config = cfg.vision_config
    
    # 1. Text model conversion 
    
    # Handle embedding weights
    if "model.embed_tokens.weight" in state_dict:
        state_dict["text_model.embed_tokens.weight"] = state_dict.pop("model.embed_tokens.weight")
    
    # Handle norm and lm_head weights
    if "model.norm.weight" in state_dict:
        state_dict["text_model.norm.weight"] = state_dict.pop("model.norm.weight")
    if "lm_head.weight" in state_dict:
        state_dict["text_model.lm_head.weight"] = state_dict.pop("lm_head.weight")
    
    # Handle decoder layers
    for i in range(text_config.num_hidden_layers):
        # Input layernorm
        old_key = f"model.layers.{i}.input_layernorm.weight"
        if old_key in state_dict:
            state_dict[f"text_model.layers.{i}.input_layernorm.weight"] = state_dict.pop(old_key)
        
        # Self attention components
        # Q, K, V projections
        for proj in ["q_proj", "k_proj", "v_proj"]:
            old_key = f"model.layers.{i}.self_attn.{proj}.weight"
            if old_key in state_dict:
                state_dict[f"text_model.layers.{i}.self_attn.qkv_proj.{proj}.weight"] = state_dict.pop(old_key)
        # Add bias keys
        for proj in ["q_proj", "k_proj", "v_proj"]:
            old_key = f"model.layers.{i}.self_attn.{proj}.bias"
            if old_key in state_dict:
                state_dict[f"text_model.layers.{i}.self_attn.qkv_proj.{proj}.bias"] = state_dict.pop(old_key)
        
        # Output projection
        old_key = f"model.layers.{i}.self_attn.o_proj.weight"
        if old_key in state_dict:
            state_dict[f"text_model.layers.{i}.self_attn.o_proj.weight"] = state_dict.pop(old_key)
        
        # Post attention layernorm
        old_key = f"model.layers.{i}.post_attention_layernorm.weight"
        if old_key in state_dict:
            state_dict[f"text_model.layers.{i}.post_attention_layernorm.weight"] = state_dict.pop(old_key)
        
        # MLP weights
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            old_key = f"model.layers.{i}.mlp.{proj}.weight"
            if old_key in state_dict:
                state_dict[f"text_model.layers.{i}.mlp.{proj}.weight"] = state_dict.pop(old_key)
    
    # 2. Vision model conversion
    # Convert vision model weights - rename from 'visual.*' to 'vision.*'
    vision_keys = [k for k in list(state_dict.keys()) if k.startswith("visual.")]
    for k in vision_keys:
        # Replace the prefix and keep the rest of the key path the same
        new_key = "vision_model." + k[len("visual."):]
        #Note we use fused qkv which is represented as Wqkv in Neuron Attn implementation
        if "qkv" in new_key:
            new_key = new_key.replace("qkv", "Wqkv")
        state_dict[new_key] = state_dict.pop(k)
    
    # Remove any unused tensors to save memory
    to_remove = []
    for k in state_dict.keys():
        # Remove rotary embedding frequency caches that will be recomputed
        if "rotary_emb.inv_freq" in k or "rope.freqs" in k:
            to_remove.append(k)
    
    for k in to_remove:
        state_dict.pop(k, None)
    
    # Force garbage collection to free memory after operations
    gc.collect()
    
    return state_dict