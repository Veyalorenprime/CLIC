"""Reproducibility utilities"""

import random
import os
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all random number generators for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
