import numpy as np
from kbnet import data_utils
from kbnet.datasets import KBNetInferenceDataset


class CustomKBNetInferenceDataset(KBNetInferenceDataset):
    def __init__(self, *args, **kwargs):
        self.ground_truth_paths = (
            kwargs.pop("ground_truth_paths") if "ground_truth_paths" in kwargs else None
        )
        super().__init__(*args, **kwargs)

    def __len__(self):
        return min(
            len(self.image_paths),
            len(self.sparse_depth_paths),
            len(self.intrinsics_paths),
        )

    def __getitem__(self, index):
        image, sparse_depth, intrinsics = super().__getitem__(index)
        if self.ground_truth_paths is not None:
            path = self.ground_truth_paths[index]
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
            ground_truth = np.stack([ground_truth, validity_map], axis=-1)
            return image, sparse_depth, intrinsics, ground_truth
        else:
            return image, sparse_depth, intrinsics
