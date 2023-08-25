from __future__ import annotations
from typing import List, Dict, Tuple
import json

import numpy as np

from motion import Motion
import os
import h5py
import PIL.Image
import matplotlib
import tensorflow as tf
import random


class MotionDataset:
    dataset: Dict["str", List[Motion]]

    def __init__(self, dataset: Dict["str", List[Motion]]):
        self.dataset = dataset

    def get_classes(self):
        return list(self.dataset.keys())

    def __getitem__(self, item: str) -> List[Motion]:
        return self.dataset[item]

    @staticmethod
    def scan_dataset_dir(dataset_path: str) -> Dict[str, List[str]]:
        class_file_mapping = dict()
        dataset_path = os.path.realpath(dataset_path)
        class_dirs = os.listdir(dataset_path)
        for class_dir in class_dirs:
            class_name = class_dir
            class_dir = os.path.join(dataset_path, class_dir)
            class_samples = [
                os.path.join(class_dir, sample) for sample in os.listdir(class_dir)
            ]
            class_file_mapping.update({class_name: class_samples})
        return class_file_mapping

    def print_stats(self) -> None:
        print(f"Num classes: {len(self.get_classes())}")
        for class_name in self.get_classes():
            print(f"f{class_name}: {len(self.dataset[class_name])} samples")

    def crop_samples(self, crop_size: int, filter_kernel_size: int = None):
        for class_name in self.dataset.keys():
            for n, sample in enumerate(self.dataset[class_name]):
                self.dataset[class_name][n].crop(
                    sample.get_global_extremum_position(), crop_size // 2
                )
                if filter_kernel_size is not None:
                    self.dataset[class_name][n].low_pass_filter(filter_kernel_size)

    def augment_dataset(
        self, shift_ratios: List[float], kernel_sizes: List[int]
    ) -> None:
        for class_name in self.dataset.keys():
            augmented_samples = []
            for n, sample in enumerate(self.dataset[class_name]):
                augmented_sample = sample.get_copy()
                for shift in shift_ratios:
                    augmented_sample.horizontal_shift(shift)
                    augmented_samples.append(augmented_sample.get_copy())
                    for kernel in kernel_sizes:
                        augmented_sample.low_pass_filter(kernel)
                        augmented_samples.append(augmented_sample.get_copy())
            self.dataset[class_name].extend(augmented_samples)

    @staticmethod
    def from_json(dataset_path: str) -> MotionDataset:
        json_mapping = MotionDataset.scan_dataset_dir(dataset_path)
        dataset = dict()
        for class_name in json_mapping.keys():
            samples = []
            for sample_path in json_mapping[class_name]:
                with open(sample_path, "r") as f:
                    sample = Motion.from_json(json.load(f))
                    samples.append(sample)
            dataset.update({class_name: samples})
        return MotionDataset(dataset)

    def to_json(self, output_dir: str) -> None:
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for class_name in self.dataset.keys():
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for n, sample in enumerate(self.dataset[class_name]):
                sample.crop(sample.get_global_extremum_position(), 50)
                sample.low_pass_filter(5)
                savefile = os.path.join(class_dir, f"{n}.json")
                sample.save_as_json(savefile)

    def to_plots(self, output_dir: str) -> None:
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        matplotlib.use("Agg")
        for class_name in self.dataset.keys():
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for n, sample in enumerate(self.dataset[class_name]):
                savefile = os.path.join(class_dir, f"{n}.jpg")
                sample.save_as_plot(savefile)

    def to_bitmap(self, output_dir: str) -> None:
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for class_name in self.dataset.keys():
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for n, sample in enumerate(self.dataset[class_name]):
                sample.crop(sample.get_global_extremum_position(), 50)
                sample.low_pass_filter(5)
                savefile = os.path.join(class_dir, f"{n}.bmp")
                sample.save_as_bitmap(savefile)

    @staticmethod
    def from_bitmap(bitmap_dir: str) -> MotionDataset:
        bitmap_mapping = MotionDataset.scan_dataset_dir(bitmap_dir)
        dataset = dict()
        for class_name in bitmap_mapping.keys():
            samples = []
            for image in bitmap_mapping[class_name]:
                with PIL.Image.open(image, formats=["BMP"]) as im:
                    samples.append(Motion.from_bitmap(im))
            dataset.update({class_name: samples})
        return MotionDataset(dataset)

    def to_hdf5(self, savefile: str) -> None:
        with h5py.File(savefile, "w") as hdf5_file:
            for class_name in self.dataset.keys():
                class_group = hdf5_file.create_group(name=class_name)
                for n, sample in enumerate(self.dataset[class_name]):
                    class_group.create_dataset(str(n), data=sample.get_array())

    @staticmethod
    def from_hdf5(hdf5_file: str) -> MotionDataset:
        dataset = dict()
        with h5py.File(hdf5_file, "r") as hdf5_file:
            for class_name in hdf5_file.keys():
                samples = []
                for sample_name in hdf5_file.get(class_name):
                    samples.append(Motion(hdf5_file[class_name][sample_name][:]))
                dataset.update({class_name: samples})
        return MotionDataset(dataset)

    def even_sample_numbers_in_classes(self, shuffle: bool = True) -> None:
        smallest_class_len = min(
            [len(samples) for samples in list(self.dataset.values())]
        )
        for class_name in self.get_classes():
            if shuffle:
                random.shuffle(self.dataset[class_name])
            self.dataset[class_name] = self.dataset[class_name][:smallest_class_len]

    def to_tf_dataset(self) -> tf.data.Dataset:
        inputs = list()
        targets = list()
        for n_class, class_name in enumerate(self.get_classes()):
            samples = [sample.get_array() for sample in self.dataset[class_name]]
            for sample in samples:
                inputs.append(tf.convert_to_tensor(sample))
                targets.append(tf.one_hot(n_class, depth=len(self.get_classes())))

        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        return dataset

    def split(
        self, ratio: float, shuffle: bool = True
    ) -> (MotionDataset, MotionDataset):
        first, second = dict(), dict()
        for class_name in self.dataset.keys():
            samples = self.dataset[class_name].copy()
            if shuffle:
                random.shuffle(samples)
            first.update({class_name: samples[int(ratio * len(samples)) :]})
            second.update({class_name: samples[: int(ratio * len(samples))]})
        return MotionDataset(first), MotionDataset(second)

    def normalize(self) -> None:
        for class_name in self.dataset.keys():
            for n, _ in enumerate(self.dataset[class_name]):
                self.dataset[class_name][n].normalize_global()
