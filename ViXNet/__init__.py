"""
ViXNet Package
Vision Transformer with Xception Network for Deepfake Detection
"""

from .model import ViXNet, create_vixnet
from .config import Config
from .dataset import create_data_loaders, check_dataset_availability

__version__ = "1.0.0"
__author__ = "ViXNet Implementation"

__all__ = [
    'ViXNet',
    'create_vixnet',
    'Config',
    'create_data_loaders',
    'check_dataset_availability',
]
