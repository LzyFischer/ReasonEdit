import torch
import pdb


class EdgeAttributePatcher:
    def __init__(self, tokenizer, model, device: str):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.activation_cache = {}
        self.gradient_cache = {}
        self.hooks = []

    def reset(self):
        for n in self.get_all_comp_nodes():
            del self.clean_activations[n], self.corrupted_activations[n]
            del self.activation_cache[n], self.gradient_cache[n]
        del self.clean_activations, self.corrupted_activations
        del self.activation_cache, self.gradient_cache
        for param in self.model.parameters():
            del param.grad
        torch.cuda.empty_cache()
        self.activation_cache, self.gradient_cache = {}, {}

    def _activation_hook(self, name: str):
        """Create a hook function to cache activations for a specific computation node."""
        def hook(module, input, output):
            self.activation_cache[name] = {
                "input": [
                    x if isinstance(x, torch.Tensor) else x for x in input
                ],
                "output": output
                if isinstance(output, torch.Tensor)
                else output,
            }
        return hook

    def _register_hooks(self):
        """Register hooks for all attention heads and feedforward layers."""
        self.hooks = []
        for name in self.get_all_comp_nodes():
            hook = self.model.get_submodule(name).register_forward_hook(
                self._activation_hook(name)
            )
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activation(self, x_clean: str, x_corrupted: str) -> torch.Tensor:
        """
        Get activations for both clean and corrupted inputs.

        Args:
            x_clean (str): Clean input text
            x_corrupted (str): Corrupted input text

        Returns:
            Tuple containing dictionaries of clean and corrupted activations
        """
        self._register_hooks()
        # Process clean input

        inputs_clean = self.tokenizer(x_clean, return_tensors="pt").to(self.device)
        clean_outputs = self.model(**inputs_clean)
        self.clean_activations = {
            k: v["output"] for k, v in self.activation_cache.items()
        }
        # Clear cache and process corrupted input
        self.activation_cache = {}
        inputs_corrupted = self.tokenizer(x_corrupted, return_tensors="pt").to(self.device)
        corrupted_outputs = self.model(**inputs_corrupted)
        self.corrupted_activations = {
            k: v["output"] for k, v in self.activation_cache.items()
        }
        self._remove_hooks()
        del inputs_clean, inputs_corrupted, corrupted_outputs
        return clean_outputs

    def get_gradient(self, clean_outputs, ground_truth):
        # get the position value of the ground truth at the first position in the output.
        topk_logits = torch.topk(clean_outputs.logits[0, -1], 10)
        token_id = [
            id for id in topk_logits[1] if ground_truth in self.tokenizer.decode(id).lower()
        ]
        if len(token_id) == 0:
            token_id = self.tokenizer.encode(ground_truth, add_special_tokens=False)[0]
        else:
            token_id = token_id[0].item()
        clean_outputs.logits[0, -1, token_id].backward()
        # Cache gradients
        for name in self.get_all_comp_nodes():
            module = self.model.get_submodule(name)
            if hasattr(module, "weight") and module.weight.grad is not None:
                self.gradient_cache[name] = module.weight.grad
                # del module.weight.grad
        # for param in self.model.parameters():
        #     del param.grad
        # del topk_logits

    def get_causal_effect(self, comp_node: str) -> tuple[float | list[float], torch.Tensor | list[torch.Tensor]]:
        """
        Compute the causal effect for a specific computation node.

        Args:
            comp_node (str): Name of the computation node

        Returns:
            torch.Tensor: Tensor value representing the causal effect
        """
        clean_activation = self.clean_activations[comp_node][0,-1]
        corrupted_activation = self.corrupted_activations[comp_node][0, -1]
        activation_diff = (clean_activation - corrupted_activation).unsqueeze(0)
        gradient = self.gradient_cache[comp_node]
        if comp_node.split(".")[-1] in ['q_proj', 'v_proj', 'k_proj']:
            head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
            activation_diff = activation_diff.reshape(-1,1,head_dim)
            gradient = gradient.reshape(-1, head_dim, gradient.shape[-1])
            clean_acts = clean_activation.reshape(-1, head_dim)            
            res = torch.bmm(activation_diff, gradient).mean(-1).squeeze()
        else:
            activation_diff = activation_diff.reshape(-1,1,1)
            gradient = gradient.unsqueeze(1)
            clean_acts = clean_activation.reshape(-1, 1)
            res = (activation_diff * gradient).mean(-1).squeeze()
        del gradient, activation_diff, clean_activation, corrupted_activation
        return res, clean_acts

    def get_all_comp_nodes(self) -> list[str]:
        """
        Get a list of all computation nodes in the model.

        Returns:
            List[str]: Names of all computation nodes
        """
        comp_nodes = []
        for name, _ in self.model.named_modules():
            if any(
                layer_type in name.lower()
                and name.split(".")[-1] not in [layer_type, "rotary_emb", "act_fn"]
                for layer_type in ["self_attn", "mlp"]
            ):
                comp_nodes.append(name)
        return comp_nodes