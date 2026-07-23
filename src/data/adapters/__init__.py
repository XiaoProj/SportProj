"""Dataset adapters for the basketball fatigue research project."""

from src.data.adapters.base import BaseDatasetAdapter, DatasetManifest, SplitInfo
from src.data.adapters.epfl_multiview import EPFLMultiViewAdapter
from src.data.adapters.epfl_single import EPFLSingleViewAdapter
from src.data.adapters.fatigue_sleep import FatigueSleepAdapter
from src.data.adapters.humanm3 import HumanM3Adapter

__all__ = [
    "BaseDatasetAdapter",
    "DatasetManifest",
    "SplitInfo",
    "EPFLMultiViewAdapter",
    "EPFLSingleViewAdapter",
    "FatigueSleepAdapter",
    "HumanM3Adapter",
]
