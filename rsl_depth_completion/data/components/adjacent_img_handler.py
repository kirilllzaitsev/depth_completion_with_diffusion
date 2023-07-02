import os

from . import raw_data_loaders as dl


def extract_frame_id_from_img_path(filename: str):
    head, tail = os.path.split(filename)
    number_string = tail[0 : tail.find(".")]
    number = int(number_string)
    return head, number


def get_nearby_img_path(filename: str, new_id: int):
    head, _ = os.path.split(filename)
    new_filename = os.path.join(head, f"{new_id:010d}.png")
    return new_filename


def get_adj_imgs(path: str, config):
    assert path is not None, "path is None"

    _, frame_id = extract_frame_id_from_img_path(path)
    offsets = []
    for i in range(1, config.n_adjacent + 1):
        offsets.append(i)
        offsets.append(-i)
    offsets_asc = sorted(offsets)

    adj_imgs = []
    for frame_offset in offsets_asc:
        path_near = get_nearby_img_path(path, frame_id + frame_offset)
        assert os.path.exists(path_near), f"cannot find two nearby frames for {path}"
        adj_imgs.append(dl.img_read(path_near))

    return adj_imgs
