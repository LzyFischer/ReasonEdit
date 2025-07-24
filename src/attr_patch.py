from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pdb

model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
model.eval()

text_clean = "If John is happy and Mary is sad, then the world is balanced."
text_corrupt = "If John is angry and Mary is sad, then the world is balanced."

inputs_clean = tokenizer(text_clean, return_tensors="pt").to("cuda")
inputs_corrupt = tokenizer(text_corrupt, return_tensors="pt").to("cuda")

with torch.no_grad():
    logits_clean = model(**inputs_clean).logits
    logits_corrupt = model(**inputs_corrupt).logits

cached_outputs = {}

def save_hook(module, input, output):
    hidden_states = output[0]  # output is a tuple
    cached_outputs['attn_out'] = hidden_states.clone().detach()

# 假设我们 patch 第3层 attention 输出
layer_idx = 3
handle = handle = model.model.layers[layer_idx].self_attn.register_forward_hook(save_hook)
model(**inputs_clean)  # 触发 hook，保存 output
handle.remove()

def patch_hook(module, input, output):
    hidden_states, attn_weights = output
    return (cached_outputs['attn_out'], attn_weights)

handle = handle = model.model.layers[layer_idx].self_attn.register_forward_hook(patch_hook)
patched_logits = model(**inputs_corrupt).logits
handle.remove()

def compare_diff(logits1, logits2, token_idx=-1):
    probs1 = F.softmax(logits1[0, token_idx], dim=-1)
    probs2 = F.softmax(logits2[0, token_idx], dim=-1)
    return (probs1 - probs2).abs().sum().item()

diff = compare_diff(patched_logits, logits_corrupt)
print(f"Importance of attention layer {layer_idx}: {diff:.4f}")


layer_idx = 3
src_head = 0  # 要拷贝的head
tgt_head = 5  # 被替换的head

cached_outputs = {}

def save_head_output_hook(module, input, output):
    hidden_states = output[0]  # output is (hidden_states, attn_weights)
    # Reshape to [batch, seq_len, num_heads, head_dim]
    batch, seq_len, dim = hidden_states.shape
    head_dim = 128  # 也可以设置为 None 然后用 dim // n 自动试探
    num_heads = dim // head_dim
    reshaped = hidden_states.view(batch, seq_len, num_heads, head_dim)
    cached_outputs["head_output"] = reshaped[:, :, src_head, :].clone().detach()

handle_save = model.model.layers[layer_idx].self_attn.register_forward_hook(save_head_output_hook)
_ = model(**inputs_clean)  # 触发 hook
handle_save.remove()

def patch_head_output_hook(module, input, output):
    hidden_states, attn_weights = output
    batch, seq_len, dim = hidden_states.shape
    head_dim = 128  # 也可以设置为 None 然后用 dim // n 自动试探
    num_heads = dim // head_dim

    reshaped = hidden_states.view(batch, seq_len, num_heads, head_dim).clone()
    reshaped[:, :, tgt_head, :] = cached_outputs["head_output"]
    patched_hidden_states = reshaped.view(batch, seq_len, dim)
    return (patched_hidden_states, attn_weights)

handle_patch = model.model.layers[layer_idx].self_attn.register_forward_hook(patch_head_output_hook)
patched_logits = model(**inputs_corrupt).logits
handle_patch.remove()

def compare_logits(logits1, logits2, token_idx=-1):
    probs1 = F.softmax(logits1[0, token_idx], dim=-1)
    probs2 = F.softmax(logits2[0, token_idx], dim=-1)
    return torch.sum(torch.abs(probs1 - probs2)).item()

with torch.no_grad():
    logits_corrupt = model(**inputs_corrupt).logits

diff = compare_logits(patched_logits, logits_corrupt)
print(f"Edge importance from head {src_head} → head {tgt_head} in layer {layer_idx}: {diff:.4f}")