import os
import glob
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class FewShotAudioDataset:
    """
    Audio Dataset organized for Few-Shot Keyword Spotting.
    Groups audio paths by keyword classes for episodic sampling.
    """

    def __init__(self, data_dir: str, classes: List[str], file_ext: str = ".wav"):
        self.data_dir = data_dir
        self.classes = sorted(list(set(classes)))
        self.file_ext = file_ext
        self.class_to_files: Dict[str, List[str]] = {}
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Indexes all audio files under each class folder."""
        for cls in self.classes:
            cls_folder = os.path.join(self.data_dir, cls)
            if not os.path.isdir(cls_folder):
                self.class_to_files[cls] = []
                continue
            files = sorted(glob.glob(os.path.join(cls_folder, f"*{self.file_ext}")))
            self.class_to_files[cls] = files

    def get_class_counts(self) -> Dict[str, int]:
        """Returns the number of files available per class."""
        return {cls: len(files) for cls, files in self.class_to_files.items()}

    def get_valid_classes(self, min_samples: int = 1) -> List[str]:
        """Returns classes that have at least min_samples audio files."""
        return [cls for cls, files in self.class_to_files.items() if len(files) >= min_samples]
