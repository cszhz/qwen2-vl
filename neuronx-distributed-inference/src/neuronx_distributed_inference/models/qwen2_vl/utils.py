from typing import List 
import numpy as np
import torch
from PIL import Image
from neuronx_distributed_inference.modules.attention.utils import _rotate_half

#Map of seq_lenght and maximum supported pixel values
PIXEL_SIZE_MAP = {1024: 1296, 2048: 2592, 4096: 8192, 8192: 8192}


def prepare_scatter_positions(input_ids, image_token_id, seq_length):
    batch_size = input_ids.shape[0]
   
    # PREPROCESSING: Create positions and valid_mask
    # Find positions where input_ids equals 100
    image_token_mask = (input_ids == image_token_id)

    # Initialize positions and valid_mask tensors
    positions = torch.zeros(batch_size, seq_length, dtype=torch.long)
    valid_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

    # Fill positions and valid_mask tensors
    for batch_idx in range(batch_size):
        # Get positions of image tokens in this sequence
        seq_positions = torch.nonzero(image_token_mask[batch_idx], as_tuple=True)[0]
        num_images = len(seq_positions)
        
        if num_images > 0:
            # Fill positions with sequence positions where image tokens were found
            positions[batch_idx, :num_images] = seq_positions
            # Mark these positions as valid
            valid_mask[batch_idx, :num_images] = True

    return positions, valid_mask
    
def pad_inputs(input_ids, attention_mask, img_start_id, img_end_id, img_token_id, max_image_tokens):
    # Get batch size and create empty result lists
    batch_size = input_ids.size(0)
    result_ids_batch = []
    result_mask_batch = []
    
    # Process each item in the batch
    for batch_idx in range(batch_size):
        input_ids_list = input_ids[batch_idx].tolist()
        attention_mask_list = attention_mask[batch_idx].tolist()
        
        # Count number of images in this batch item
        n_images = (input_ids[batch_idx] == img_end_id).sum().item()
        
        if n_images == 0:
            # No images, just copy the sequence as is
            result_ids_batch.append(input_ids_list)
            result_mask_batch.append(attention_mask_list)
            continue
        
        # Calculate base tokens per image and leftover tokens
        base_tokens_per_image = max_image_tokens // n_images
        leftover_tokens = max_image_tokens % n_images
        
        result_ids = []
        result_mask = []
        i = 0
        img_count = 0
        total_img_tokens_added = 0
        
        while i < len(input_ids_list):
            # Add the current token
            result_ids.append(input_ids_list[i])
            result_mask.append(attention_mask_list[i])
            
            # Check if this is an img_start_id
            if input_ids_list[i] == img_start_id:
                img_count += 1
                is_last_image = (img_count == n_images)
                
                # Find the corresponding img_end_id
                j = i + 1
                img_token_count = 0
                current_img_tokens = []
                current_img_masks = []
                
                while j < len(input_ids_list) and input_ids_list[j] != img_end_id:
                    if input_ids_list[j] == img_token_id:
                        img_token_count += 1
                        current_img_tokens.append(input_ids_list[j])
                        current_img_masks.append(attention_mask_list[j])
                    else:
                        # Non-image tokens between start and end should be kept
                        current_img_tokens.append(input_ids_list[j])
                        current_img_masks.append(attention_mask_list[j])
                    j += 1
                
                # Calculate target tokens for this image
                # Last image gets any leftover tokens
                target_tokens = base_tokens_per_image
                if is_last_image:
                    target_tokens += leftover_tokens
                
                # Make sure we don't exceed the total max_image_tokens
                remaining_tokens = max_image_tokens - total_img_tokens_added
                target_tokens = min(target_tokens, remaining_tokens)
                
                # Add the original tokens first (only the img_token_id count towards the total)
                for k in range(len(current_img_tokens)):
                    result_ids.append(current_img_tokens[k])
                    result_mask.append(current_img_masks[k])
                    if current_img_tokens[k] == img_token_id:
                        total_img_tokens_added += 1
                
                # Add padding if necessary
                padding_needed = target_tokens - img_token_count
                if padding_needed > 0:
                    # Add padding img_token_id with attention mask 0
                    for _ in range(padding_needed):
                        result_ids.append(img_token_id)
                        result_mask.append(0)  # Set attention to 0 for padding
                        total_img_tokens_added += 1
                
                # Add the end token if we found it
                if j < len(input_ids_list):
                    result_ids.append(input_ids_list[j])
                    result_mask.append(attention_mask_list[j])
                    i = j  # Skip to after the end token
                else:
                    break
            
            i += 1
        
        result_ids_batch.append(result_ids)
        result_mask_batch.append(result_mask)
    
    # Pad all sequences to the same length
    max_len = max(len(seq) for seq in result_ids_batch)
    
    for i in range(batch_size):
        padding_len = max_len - len(result_ids_batch[i])
        if padding_len > 0:
            # Use padding token 0 for input_ids and 0 for attention_mask
            result_ids_batch[i].extend([0] * padding_len)
            result_mask_batch[i].extend([0] * padding_len)
    
    # Convert back to tensors
    return torch.tensor(result_ids_batch), torch.tensor(result_mask_batch)


def pad_inputs_simple(input_ids, attention_mask, img_start_id, img_end_id, img_token_id, max_image_tokens):
    n_image_tokens = torch.sum(input_ids==img_token_id)
    n_image_left = max_image_tokens-n_image_tokens
    new_input_ids = torch.zeros((1, input_ids.shape[1]+ n_image_left))
    new_attention_mask = torch.zeros((1, input_ids.shape[1]+ n_image_left))
    
    new_input_ids[0,:input_ids.shape[1]] = input_ids[0,:]
    new_input_ids[0,input_ids.shape[1]:] = img_token_id
    new_attention_mask[0,:input_ids.shape[1]] = attention_mask[0,:]
    
    return new_input_ids, new_attention_mask
    


def update_messages(messages, pixel_size_map, seq_length, patch_size):
    
    #Count number of images in input messages
    n_images = 0
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image":
                n_images += 1
    
    #Distribute pixels equally between input images
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image":
                content["max_pixels"] = np.floor(pixel_size_map[seq_length]*patch_size*patch_size/n_images)
                content["min_pixels"] = 64*28*28
                #print("zz:",content["max_pixels"], content["min_pixels"]) #1 image: 1605632.0 50176
    #print(messages)        
    return messages

def fixed_split_lastdim(x, sizes):
    cum_sizes = [0]
    for s in sizes:
        cum_sizes.append(cum_sizes[-1] + s)

    slices = []
    for i in range(len(sizes)):
        start = cum_sizes[i]
        end = cum_sizes[i + 1]
        slc = [slice(None)] * x.dim()
        slc[-1] = slice(start, end)
        slices.append(x[tuple(slc)])
    
    return tuple(slices)

def custom_mrope_tensor(tensor: torch.Tensor,
                       name: str,
                       mrope_section: list[int],
                       unsqueeze_dim: int = 1) -> torch.Tensor:
    """
    Debug helper for one of the [cos|sin] tensors:
      1) split into chunks
      2) select T/H/W rows via i%3
      3) concat back to head_dim
      4) unsqueeze for broadcasting
      5) test broadcast with q

    Args:
        tensor:     [3, 1, 1, head_dim] the cos or sin input
        name:       a label, e.g. "cos" or "sin"
        q:          the q tensor to check broadcast against
        mrope_section:  e.g. [16,24,24,16,24,24]
        unsqueeze_dim:  where to insert the extra dim for broadcasting
    Returns:
        unsqueezed: [1,1,1,1,head_dim] (if unsqueeze_dim=1)
    """
    full_sections = mrope_section * 2
    #print(f"\n=== Debug {name} ===")
    # 1. split
    #splits = tensor.split(full_sections, dim=-1)
    splits = fixed_split_lastdim(tensor, full_sections)
    
    # print(f"1) {name}.split → {len(splits)} chunks")
    # for i, blk in enumerate(splits):
    #     print(f"   chunk[{i}].shape = {blk.shape}")
    #return splits[0]

    # 2. select T/H/W row
    selected = []
    for i, blk in enumerate(splits):
        row = blk[i % 3]
        #print(f"2) select chunk[{i}] row {i%3} → {row.shape}")
        selected.append(row)
    # 3. concat
    cat = torch.cat(selected, dim=-1)
    #print(f"3) concat → {cat.shape}")

    # 4. unsqueeze
    unsq = cat.unsqueeze(unsqueeze_dim)
    #print(f"4) unsqueeze dim={unsqueeze_dim} → {unsq.shape}")

    return unsq

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # debug both cos and sin:
    cos_dbg = custom_mrope_tensor(cos, "cos", mrope_section, unsqueeze_dim)
    sin_dbg = custom_mrope_tensor(sin, "sin", mrope_section, unsqueeze_dim)
    #return cos_dbg, sin_dbg
    # now do the real rotate:
    q_embed = (q * cos_dbg) + (_rotate_half(q) * sin_dbg)
    k_embed = (k * cos_dbg) + (_rotate_half(k) * sin_dbg)
    return q_embed, k_embed

# def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
#     # print('=== mrope_section ', mrope_section)
#     mrope_section = mrope_section * 2
#     cos = torch.cat(
#         [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
#     ).unsqueeze(unsqueeze_dim)
#     sin = torch.cat(
#         [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
#     ).unsqueeze(unsqueeze_dim)

#     q_embed = (q * cos) + (_rotate_half(q) * sin)
#     k_embed = (k * cos) + (_rotate_half(k) * sin)
#     return q_embed, k_embed



def get_qwen2vl_image_tensors(
    config, batch_images: List[List], is_for_context_encoding=True
):
    bsz = len(batch_images)
    image_height = 502
    image_width = 502

    if len(batch_images[0]) == 0 or config.neuron_config.skip_vision:
        print("Setting empty vision inputs...")

        if config.neuron_config.skip_vision or (not is_for_context_encoding):
            empty_pixel_values = torch.tensor(
                [0] * config.neuron_config.batch_size, dtype=torch.int32
            )
        else:
            empty_pixel_values = torch.zeros(
                [
                    bsz,
                    config.vision_config.in_channels,  
                    image_height,
                    image_width,
                ],
                dtype=config.neuron_config.torch_dtype,
            )

        if hasattr(config.vision_config, "patch_size"):
            patch_size = config.vision_config.patch_size
        else:
            patch_size = 14  


        h_patches = image_height // patch_size
        w_patches = image_width // patch_size
        print('h_patches w_patches', h_patches, w_patches)

        empty_grid_thw = torch.zeros((bsz, 1, 3), dtype=torch.int32)
        empty_grid_thw[:, :, 0] = 1  
        empty_grid_thw[:, :, 1] = h_patches  
        empty_grid_thw[:, :, 2] = w_patches  

        has_image = torch.zeros([bsz], dtype=torch.int32)
        return empty_pixel_values, empty_grid_thw, has_image


def resize_image(img_file_path, max_tokens = 1296):
    
    img = Image.open(img_file_path)

    w, h = img.size

    factor = w*h/(14*14*max_tokens)


    factor_sqrt = np.sqrt(factor)
    new_width = np.floor(w/factor_sqrt)
    new_height = np.floor(h/factor_sqrt)

    new_width = (new_width//28) * 28
    new_height = (new_height//28) * 28

    img_resize = img.resize((int(new_width), int(new_height)))
    
    # grid_h, grid_w = new_height // 14, resized_width // self.patch_size

    return img_resize 
