import torch

# train.py calls thop.profile() before training, which registers these buffers on every
# module, so they end up in every saved state_dict. They carry no weights.
THOP_KEYS = ("total_ops", "total_params")

# SwinIR rebuilds these buffers for the inference resolution.
SWINIR_BUFFERS = ("attn_mask", "relative_position_index")


def load_checkpoint(model, path, device="cpu", is_swinir=False):
    """Load a state_dict and fail loudly if any weight is missing or unexpected.

    The original scripts used load_state_dict(strict=False) to tolerate the thop keys,
    which would also silently leave a model with random weights on a key mismatch.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    state = {k: v for k, v in state.items() if not k.endswith(THOP_KEYS)}
    if is_swinir:
        state = {k: v for k, v in state.items() if not any(b in k for b in SWINIR_BUFFERS)}

    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys
               if not (is_swinir and any(b in k for b in SWINIR_BUFFERS))]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint {path} does not match the model: "
                           f"missing={missing[:5]} unexpected={result.unexpected_keys[:5]}")
    model.eval()
    return model
