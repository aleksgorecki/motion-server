from __future__ import annotations

from typing import List, Dict
import json
from motion import Motion
import os
import h5py


class MotionDataset:
    dataset: Dict["str", List[Motion]]

    def get_classes(self):
        return self.dataset.keys()


def scan_json_dataset(dataset_path: str) -> Dict[str, List[str]]:
    class_file_mapping = dict()
    dataset_path = os.path.realpath(dataset_path)
    class_dirs = os.listdir(dataset_path)
    for class_dir in class_dirs:
        class_name = class_dir
        class_dir = os.path.join(dataset_path, class_dir)
        class_samples = [os.path.join(class_dir, sample) for sample in os.listdir(class_dir)]
        class_file_mapping.update({class_name: class_samples})
    return class_file_mapping


def load_json_to_memory(json_mapping: dict) -> Dict[str, List[Motion]]:
    dataset = dict()
    for class_name in json_mapping.keys():
        samples = []
        for sample_path in json_mapping[class_name]:
            with open(sample_path, "r") as f:
                sample = Motion.from_json(json.load(f))
                samples.append(sample)
        dataset.update({class_name: samples})
    return dataset


def dataset_to_bitmap(dataset: Dict["str", List[Motion]], output_dir: str) -> None:
    output_dir = os.path.realpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for class_name in dataset.keys():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for n, sample in enumerate(dataset[class_name]):
            sample.crop(sample.get_global_extremum_position(), 50)
            sample.low_pass_filter(5)
            savefile = os.path.join(class_dir, f"{n}.bmp")
            sample.save_as_bitmap(savefile)


def dataset_to_hdf5(dataset: Dict["str", List[Motion]], savefile: str) -> None:
    with h5py.File(savefile, "w") as hdf5_file:
        for class_name in dataset.keys():
            class_group = hdf5_file.create_group(name=class_name)
            for n, sample in enumerate(dataset[class_name]):
                class_group.create_dataset(str(n), data=sample.get_array())


def dataset_from_hdf5(hdf5_file: str) -> Dict["str", List[Motion]]:
    dataset = dict()
    with h5py.File(hdf5_file, "r") as hdf5_file:
        for class_name in hdf5_file.keys():
            samples = []
            for sample_name in hdf5_file.get(class_name):
                samples.append(Motion(hdf5_file[class_name][sample_name]))
            dataset.update({class_name: samples})
    return dataset

if __name__ == "__main__":
    # #dataset_to_bitmap("./dataset", "./dataset_bitmap")
    # with open("./dataset/zneg/zneg_20221204_182105.json", "r") as json_file_handle:
    #     json_object = json.load(json_file_handle)
    # data = Motion.from_json(json_object)
    # data.crop(data.get_global_extremum_position(), 60)
    # data.save_as_plot("test_plot.jpg")
    # data.horizontal_shift(0.7)
    # data.scale_values(0.2)
    # data.low_pass_filter(5)
    # data.save_as_plot("test_plot_shifted.jpg")
    # tf.train.FloatList
    #dataset = scan_json_dataset("./dataset")
    dataset = scan_json_dataset("./dataset")
    d = load_json_to_memory(dataset)
    # dataset_to_bitmap(d, "./dataset_bitmap2")
    dataset_to_hdf5(d, "./testhdf.hdf5")
    d2 = dataset_from_hdf5("./testhdf.hdf5")
    print()
