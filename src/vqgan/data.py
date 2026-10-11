"""Memory-mapped NCHW datasets; normalize only the batch being consumed."""

from pathlib import Path

import numpy as np


class ImageDataset:
    def __init__(self, path: str | Path):
        self.images = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.images.ndim != 4 or min(self.images.shape) <= 0:
            raise ValueError("dataset must be a nonempty NCHW array")
        if self.images.dtype != np.uint8 and not np.issubdtype(self.images.dtype, np.floating):
            raise ValueError("dataset dtype must be uint8 or normalized floating point")

    def __len__(self):
        return len(self.images)

    def batch(self, indices, *, rng=None, horizontal_flip=False):
        batch = np.array(self.images[indices], dtype=np.float32, copy=True)
        if self.images.dtype == np.uint8:
            batch *= np.float32(1 / 127.5)
            batch -= np.float32(1)
        if horizontal_flip:
            if rng is None:
                raise ValueError("provide an RNG for training augmentation")
            flip = rng.random(len(batch)) < 0.5
            batch[flip] = batch[flip, :, :, ::-1]
        return batch

    def evaluation_batches(self, batch_size, *, max_batches=None):
        """Fixed-size batches plus valid count, excluding padded examples in metrics."""
        if batch_size <= 0 or (max_batches is not None and max_batches <= 0):
            raise ValueError("batch_size and max_batches must be positive")
        for number, start in enumerate(range(0, len(self), batch_size)):
            if max_batches is not None and number >= max_batches:
                break
            indices = np.arange(start, min(start + batch_size, len(self)))
            count = len(indices)
            indices = np.pad(indices, (0, batch_size - count), mode="edge")
            yield self.batch(indices), count
