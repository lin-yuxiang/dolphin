import numpy as np
import torch


def ensure_rng(rng=None):
    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng


def random_boxes(num=1, scale=1, rng=None):
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = torch.from_numpy(tlbr)
    return boxes
    