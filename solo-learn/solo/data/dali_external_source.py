import os
from pathlib import Path
from typing import Optional, Union

import torch

import random
import numpy as np

def get_synthetic_image_path(filename, imagenet_synthetic_root, rand_int ,split="train"):
    filename_and_extension = filename.split("/")[-1]
    filename_parent_dir = filename.split("/")[-2]
    image_path = os.path.join(
        imagenet_synthetic_root,
        split,
        filename_parent_dir,
        filename_and_extension.split(".")[0] + f"_{rand_int}.JPEG",
    )
    return image_path

class ExternalInputIterator(object):
    def __init__(self,
                 files: list,
                 labels: list,
                 synthetic_data_path: Optional[Union[str, Path]],
                 synthetic_index_min: int,
                 synthetic_index_max: int,
                 batch_size: int, 
                 shard_id: int = 0,
                 random_shuffle: bool = True,
                 num_shards: int = 1,
                 ):
        
        self.batch_size = batch_size

        self.synthetic_data_path = synthetic_data_path
        self.synthetic_index_min = synthetic_index_min
        self.synthetic_index_max = synthetic_index_max
        self.random_shuffle = random_shuffle

        
        self.files = files
        self.labels = labels
        
        self.synth_files = [[
            get_synthetic_image_path(
                jpeg_filename.absolute().as_posix(), 
                self.synthetic_data_path, 
                i ,
                split="train") for jpeg_filename in files ] for i in range(synthetic_index_min,synthetic_index_max+1)
                ]
        
        # whole data set size
        self.data_set_len = len(self.files)

        # based on the shard_id and total number of GPUs - world size
        # get proper shard
        self.files = self.files[self.data_set_len * shard_id // num_shards:
                                self.data_set_len * (shard_id + 1) // num_shards]
        self.n = len(self.files)


    def __iter__(self):
        self.i = 0
        self.indexes = np.arange(len(self.files))
        if self.random_shuffle:
            np.random.shuffle(self.indexes)
#             self.files[:] = [self.files[i] for i in indexes]
#             self.labels[:] = [self.labels[i] for i in indexes]
#             for j in range(len(self.synth_files)):
#                 self.synth_files[j][:] = [self.synth_files[j][i] for i in indexes]
        return self

    def __next__(self):
        batch = []
        labels = []
        synthetic_batch = []
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            index = self.indexes[self.i % self.n]
            jpeg_filename = self.files[index]
            label = self.labels[index]
            f = open(jpeg_filename, "rb")
            file = np.frombuffer(f.read(), dtype=np.uint8)
            batch.append(file)
#             file = cp.asarray(imageio.imread(jpeg_filename))
#             file = np.fromfile(jpeg_filename, dtype = np.uint8)
#             batch.append(file.astype(cp.uint8))  # we can use numpy
#             labels.append(torch.tensor([int(label)], dtype = torch.int64)) # or PyTorch's native tensors
            labels.append(torch.tensor([label], dtype = torch.int64))
            if self.synthetic_data_path:
                rand_int = random.randint(self.synthetic_index_min, self.synthetic_index_max)
                synth_jpeg_filename = self.synth_files[rand_int][index]
#                 synth_jpeg_filename = get_synthetic_image_path(jpeg_filename.absolute().as_posix(), self.synthetic_data_path, rand_int ,split="train")
                f = open(synth_jpeg_filename, "rb")
                synth_file = np.frombuffer(f.read(), dtype=np.uint8)
#                 synth_file = cp.asarray(imageio.imread(synth_jpeg_filename))
#                 np.fromfile(synth_jpeg_filename, dtype=np.uint8)
                synthetic_batch.append(synth_file)
            else:
                synthetic_batch.append(file)


            self.i += 1

        return (batch, labels, synthetic_batch)

    def __len__(self):
        return self.data_set_len

    next = __next__