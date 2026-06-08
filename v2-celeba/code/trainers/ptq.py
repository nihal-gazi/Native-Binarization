import torch
from ..logger import logger

def convert_fp16_to_w1a16(fp16_model, w1a16_model):
    """
    Performs Post-Training Quantization (PTQ) conversion:
    Copies weights from the trained FP16 model state dict into the W1A16 container model.
    The W1A16 model will center-binarize these weights on-the-fly during forward passes.
    """
    logger.info("Starting FP16 -> W1A16 PTQ conversion...")
    
    fp16_state = fp16_model.state_dict()
    w1a16_state = w1a16_model.state_dict()
    
    converted_keys = []
    
    for key in w1a16_state.keys():
        if key in fp16_state:
            # Copy parameter tensor
            w1a16_state[key].copy_(fp16_state[key].float())
            converted_keys.append(key)
        else:
            logger.warning(f"Key '{key}' not found in FP16 source state dict.")
            
    w1a16_model.load_state_dict(w1a16_state)
    logger.info(f"Successfully converted {len(converted_keys)} parameter tensors for PTQ.")
    return w1a16_model
