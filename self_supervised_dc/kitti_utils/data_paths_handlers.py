import typing as t

import glob
import os
from abc import abstractmethod


class SplitDataPathsHandler:
    def __init__(self, split, subsplit, data_folder):
        self.split = split
        self.subsplit = subsplit
        self.handler_cls = find_data_paths_handler(split, subsplit)
        self.handler = self.handler_cls(data_folder)

    @property
    def paths_d(self) -> list[str]:
        return self.handler.paths_d

    @property
    def paths_img(self) -> list[str]:
        return self.handler.paths_img

    @property
    def paths_gt(self) -> list[str]:
        return self.handler.paths_gt


class AnonymousdDataPathsHandler(SplitDataPathsHandler):
    def __init__(self, glob_img):
        self.glob_img = glob_img

    @property
    def paths_d(self):
        raise NotImplementedError

    @property
    def paths_img(self):
        return sorted(glob.glob(self.glob_img))

    @property
    def paths_gt(self):
        return [None] * len(self.paths_img)


class AnonymousCompletiondDataPathsHandler(AnonymousdDataPathsHandler):
    def __init__(self, data_folder):
        self.glob_img = os.path.join(
            data_folder,
            "depth_selection/test_depth_completion_anonymous/image/*.png",
        )
        self.glob_d = os.path.join(
            data_folder,
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png",
        )
        super().__init__(self.glob_img)

    @property
    def paths_d(self):
        return sorted(glob.glob(self.glob_d))


class AnonymousPredictiondDataPathsHandler(AnonymousdDataPathsHandler):
    def __init__(self, data_folder):
        self.glob_img = os.path.join(
            data_folder,
            "depth_selection/test_depth_prediction_anonymous/image/*.png",
        )
        super().__init__(self.glob_img)

    @property
    def paths_d(self):
        return [None] * len(glob.glob(self.glob_img))


class LabeledDataPathsHandler(SplitDataPathsHandler):
    def __init__(self, glob_d, glob_gt):
        self.glob_d = glob_d
        self.glob_gt = glob_gt

    @property
    def paths_d(self):
        return sorted(glob.glob(self.glob_d))

    @property
    def paths_gt(self):
        return sorted(glob.glob(self.glob_gt))

    @abstractmethod
    def get_img_paths(self, p):
        raise NotImplementedError

    @property
    def paths_img(self):
        return [self.get_img_paths(p) for p in self.paths_gt]


class TrainSplitDataPathsHandler(LabeledDataPathsHandler):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.glob_d = os.path.join(
            data_folder,
            "data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png",
        )
        self.glob_gt = os.path.join(
            data_folder,
            "data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png",
        )
        super().__init__(self.glob_d, self.glob_gt)

    def get_img_paths(self, p):
        ps = p.split("/")
        pnew = "/".join(
            [self.data_folder]
            + ["data_rgb"]
            + ps[-6:-4]
            + ps[-2:-1]
            + ["data"]
            + ps[-1:]
        )
        return pnew


class ValFullSubsplitDataPathsHandler(LabeledDataPathsHandler):
    def __init__(self, data_folder):
        self.glob_d = os.path.join(
            data_folder,
            "data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png",
        )
        self.glob_gt = os.path.join(
            data_folder,
            "data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png",
        )
        super().__init__(self.glob_d, self.glob_gt)

    def get_img_paths(self, p):
        ps = p.split("/")
        pnew = "/".join(
            ps[:-7] + ["data_rgb"] + ps[-6:-4] + ps[-2:-1] + ["data"] + ps[-1:]
        )
        return pnew


class ValSelectSubsplitDataPathsHandler(LabeledDataPathsHandler):
    def __init__(self, data_folder):
        self.glob_d = os.path.join(
            data_folder,
            "depth_selection/val_selection_cropped/velodyne_raw/*.png",
        )
        self.glob_gt = os.path.join(
            data_folder,
            "depth_selection/val_selection_cropped/groundtruth_depth/*.png",
        )
        super().__init__(self.glob_d, self.glob_gt)

    def get_img_paths(self, p):
        return p.replace("groundtruth_depth", "image")


def find_data_paths_handler(split, subsplit=None) -> t.Type[SplitDataPathsHandler]:
    if split == "train":
        return TrainSplitDataPathsHandler
    elif split == "val":
        if subsplit == "full":
            return ValFullSubsplitDataPathsHandler
        elif subsplit == "select":
            return ValSelectSubsplitDataPathsHandler
        raise ValueError("Unrecognized subsplit " + str(subsplit))
    elif split == "test":
        if subsplit == "completion":
            return AnonymousCompletiondDataPathsHandler
        elif subsplit == "prediction":
            return AnonymousPredictiondDataPathsHandler
        else:
            raise ValueError("Unrecognized subsplit " + str(subsplit))
    else:
        raise ValueError("Unrecognized split " + str(split))
