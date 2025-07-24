import torch
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import copy

class CircuitAblator:
    """
    Module for ablating modular circuits in LLMs and measuring performance changes.
    """

    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        """
        Initialize the circuit ablator.

        Args:
            model: The LLM model
            tokenizer: The tokenizer for the model
            device: Device to run the model on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.original_model = copy.deepcopy(model)
        self.ablation_hooks = []

    def _get_module_path(self, node_name: str) -> str:
        """Convert node name to module path."""
        parts = node_name.split(",")
        if len(parts) != 4:
            return ""

        component, layer_idx, idx, component_type = parts
        layer_idx = int(layer_idx)-1

        if component == "self_attn":
            if component_type == "attn_head":
                # For attention heads, we target o_proj and work backwards
                return f"model.layers.{layer_idx}.self_attn.o_proj"
            elif component_type == "o_proj":
                return f"model.layers.{layer_idx}.self_attn.o_proj"
        elif component == "mlp":
            if component_type in ["gate_proj", "up_proj", "down_proj"]:
                return f"model.layers.{layer_idx}.mlp.{component_type}"
            elif component_type == "mlp_in":
                return [f"model.layers.{layer_idx}.mlp.gate_proj", f"model.layers.{layer_idx}.mlp.up_proj"]

        return ""

    def _create_ablation_mask(self, node_name: str, head_idx: Optional[int] = None) -> torch.Tensor:
        """
        Create an appropriate ablation mask for the given node.

        Args:
            node_name: Name of the node to ablate
            head_idx: Index of attention head if applicable

        Returns:
            Ablation mask tensor
        """
        parts = node_name.split(",")
        if len(parts) != 4:
            return None

        component, layer_idx, idx, component_type = parts
        idx = int(idx)

        if component_type == "attn_head":
            # Create a mask for specific attention head
            return idx
        elif component_type in ["o_proj", "gate_proj", "up_proj", "down_proj"]:
            # Create mask for specific neuron in the projection
            module_path = self._get_module_path(node_name)
            if not module_path:
                return None

            module = self.model.get_submodule(module_path)
            if not hasattr(module, "weight"):
                return None

            # Create mask that zeros out specific row/column
            mask = torch.ones_like(module.weight)
            if component_type == "o_proj":
                # For output projection, zero out specific column
                mask[:, idx] = 0
            else:
                # For other projections, zero out specific row
                mask[idx, :] = 0

            return mask

        return None

    def _ablation_hook(self, node_name: str, mask: torch.Tensor):
        """
        Create a forward hook that ablates specific circuit components.

        Args:
            node_name: Name of the node to ablate
            mask: Ablation mask to apply

        Returns:
            Hook function
        """
        parts = node_name.split(",")
        component_type = parts[3] if len(parts) == 4 else ""

        def hook_fn(module, input_tensor, output):
            if component_type == "attn_head":
                # Ablate specific attention head
                head_idx = mask
                # Get input shape and reshape to isolate heads
                if len(output.shape) == 3:  # [batch, seq_len, hidden_dim]
                    batch, seq_len, hidden_dim = output.shape
                    num_heads = self.model.config.num_attention_heads
                    head_dim = hidden_dim // num_heads

                    # Reshape to [batch, seq_len, num_heads, head_dim]
                    reshaped = output.view(batch, seq_len, num_heads, head_dim)

                    # Zero out the specific head
                    reshaped[:, :, head_idx, :] = 0

                    # Reshape back to original shape
                    return reshaped.view(batch, seq_len, hidden_dim)
                return output
            else:
                # Apply the mask to the output for other component types
                if isinstance(mask, torch.Tensor) and mask.shape == module.weight.shape:
                    # Element-wise multiplication with the mask
                    with torch.no_grad():
                        module.weight.data = module.weight.data * mask
                return output

        return hook_fn

    def ablate_circuit(self, mc_nodes: List[str]):
        """
        Ablate a modular circuit by zeroing out its components.

        Args:
            mc_nodes: List of node names in the modular circuit to ablate
        """
        # Remove any existing hooks
        self.remove_ablation()

        # Register hooks for each node in the circuit
        for node in mc_nodes:
            module_path = self._get_module_path(node)
            if not module_path:
                continue

            if isinstance(module_path, list):
                # Handle case where we have multiple modules to target
                for path in module_path:
                    try:
                        module = self.model.get_submodule(path)
                        mask = self._create_ablation_mask(node)
                        if mask is not None:
                            hook = module.register_forward_hook(self._ablation_hook(node, mask))
                            self.ablation_hooks.append(hook)
                    except Exception as e:
                        print(f"Failed to register hook for {path}: {str(e)}")
            else:
                try:
                    module = self.model.get_submodule(module_path)
                    mask = self._create_ablation_mask(node)
                    if mask is not None:
                        hook = module.register_forward_hook(self._ablation_hook(node, mask))
                        self.ablation_hooks.append(hook)
                except Exception as e:
                    print(f"Failed to register hook for {node}: {str(e)}")

    def remove_ablation(self):
        """Remove all ablation hooks to restore normal model function."""
        for hook in self.ablation_hooks:
            hook.remove()
        self.ablation_hooks = []

        # Reset model to original weights
        self.model.load_state_dict(self.original_model.state_dict())

    def measure_performance(self, dataset: List[Dict], metric_fn=None) -> Dict:
        """
        Measure model performance on a dataset.

        Args:
            dataset: List of examples to evaluate
            metric_fn: Optional function to calculate metrics (defaults to accuracy)

        Returns:
            Dictionary of performance metrics
        """
        if metric_fn is None:
            # Default to accuracy for multiple-choice questions
            metric_fn = self._default_accuracy_metric

        results = metric_fn(self.model, self.tokenizer, dataset, self.device)
        return results

    def _default_accuracy_metric(self, model, tokenizer, dataset, device):
        """Default metric function calculating accuracy on multiple-choice tasks."""
        correct = 0
        total = 0

        for example in dataset:
            input_text = example["clean_text"]
            correct_answer = example["answer"]

            # Tokenize input
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a medical assistant. Choose the best answer",
                    },
                    {"role": "user", "content": input_text},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False
                )

            # Decode prediction
            prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer (a, b, c, or d)
            # This is a simple approach - may need to adjust based on model output format
            for char in prediction_text.lower():
                if char in ['a', 'b', 'c', 'd']:
                    prediction = char
                    break
            else:
                prediction = None

            # Check if correct
            if prediction == correct_answer:
                correct += 1
            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total
        }

    def analyze_functional_changes(self, mc_idx: int, mc_nodes: List[str],
                                   dataset: List[Dict], func_interp: Dict) -> Dict:
        """
        Analyze functional changes when a modular circuit is ablated.

        Args:
            mc_idx: Index of the modular circuit
            mc_nodes: List of node names in the modular circuit
            dataset: Evaluation dataset
            func_interp: Functional interpretations of modular circuits

        Returns:
            Dictionary with analysis results
        """
        # Original model performance
        original_performance = self.measure_performance(dataset)

        # Ablate the circuit
        self.ablate_circuit(mc_nodes)

        # Measure ablated performance
        ablated_performance = self.measure_performance(dataset)

        # Calculate degradation
        performance_degradation = {
            "accuracy_change": original_performance["accuracy"] - ablated_performance["accuracy"],
            "percent_degradation": ((original_performance["accuracy"] - ablated_performance["accuracy"])
                                   / original_performance["accuracy"]) * 100 if original_performance["accuracy"] > 0 else 0
        }

        # Get the functional interpretation
        mc_function = func_interp.get(str(mc_idx), "No interpretation available")

        # Restore model
        self.remove_ablation()

        return {
            "mc_idx": mc_idx,
            "original_performance": original_performance,
            "ablated_performance": ablated_performance,
            "performance_degradation": performance_degradation,
            "functional_interpretation": mc_function,
            "num_nodes_ablated": len(mc_nodes)
        }