import torch
from dataclasses import dataclass
import json
from model_llama import Transformer

@dataclass
class ModelArgs:
    dim: int = None
    n_layers: int = None
    n_heads: int = None
    n_kv_heads: int = None
    vocab_size: int = None
    ffn_dim_multiplier: float = None
    multiple_of: int = None
    norm_eps: float = None
    rope_theta: float = None
    use_scaled_rope: bool = None
    max_seq_len: int = None


def load_model(model_path, model_args):
    with torch.device("meta"):
        model = Transformer(model_args)
    
    model = model.to_empty(device="cpu")
    state_dict = torch.load(f"{model_path}/consolidated.00.pth", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, assign=True)

    # Load freqs_cis separately
    with torch.no_grad():
        model.freqs_cis = model._precompute_freqs_cis()
    return model

if __name__ == "__main__":
    model_name_original = "llama_3b_instruct"
    model_path = f"./{model_name_original}/original"
    model_config = f"{model_path}/params.json"
    with open(model_config, "r") as f:
        params = json.load(f)

    params['max_seq_len'] = 131072
    model_args = ModelArgs(**params)

    model = load_model(model_path, model_args)
    # model.to("cuda")

    for layer_id, transformer_block in model.layers.named_children():
        print(layer_id)
        print(transformer_block)
        print()
    for layer_id, transformer_block in model.layers.items():
        print(layer_id)
        print(transformer_block)
        print()
    def get_parameter_names(model, forbidden_layer_types):
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]

        result += list(model._parameters.keys())
        return result


    decay_parameters = get_parameter_names(model, [torch.nn.RMSNorm])

    print(decay_parameters)
    for x in model.parameters():
        print(x.shape)
        print(x)
        break
    sd = model.state_dict()
    for k, v in sd.items():
        print(k)
        print(v.shape)
        print(v)
        print()