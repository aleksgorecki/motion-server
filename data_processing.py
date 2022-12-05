import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import json
import io


def plot_motion(response_data, savefile):

    fig, ax = plt.subplots()
    for motion_axis in ("x", "y", "z"):
        axis_values = [float(x) for x in response_data[motion_axis]]
        ax.plot(range(len(axis_values)), axis_values, label=motion_axis)
        
    ax.legend()
    ax.set_ylabel("acceleration m/s^2")
    ax.set_xlabel("sample number")
    fig.savefig(savefile)

def motion_json_to_object(json_path):
    with open(json_path, "r") as json_file:
        return json.load(json_file)

def json_data_to_array(json_data):
    x_row = np.array(json_data["x"])
    y_row = np.array(json_data["y"])
    z_row = np.array(json_data["z"])
    return np.array([x_row, y_row, z_row])

def normalize_amplitude_values(arr):
    global_min = arr.min()
    global_max = arr.max()
    return (((arr - global_min) / (global_max - global_min)))

def filter_high_frequency_oscillations(arr, ones_len):
    ret_arr = np.empty(dtype=float, shape=arr.shape)
    for row_idx, row in enumerate(arr):
        filtered_row = np.convolve(row, np.ones(ones_len), 'same') / ones_len
        ret_arr[row_idx] = filtered_row
    return ret_arr

def save_array_to_bitmap(arr, savefile):
    scaled_arr = np.array(arr * 255, dtype=np.uint8)
    im = Image.fromarray(scaled_arr)
    im.save(savefile)

def dataset_to_bitmap(dataset_path, output_path):
    
    os.makedirs(output_path, exist_ok=True)

    class_dirs = os.listdir(dataset_path)
    for class_dir in class_dirs:
        os.makedirs(os.path.join(output_path, class_dir), exist_ok=True)
        samples_jsons = os.listdir(os.path.join(dataset_path, class_dir))
        for sample_json in samples_jsons:
            sample_data = motion_json_to_object(os.path.join(dataset_path, class_dir, sample_json))
            
            sample_data_arr = json_data_to_array(sample_data)
            normalized_sample = normalize_amplitude_values(sample_data_arr)
            filtered_sample = filter_high_frequency_oscillations(normalized_sample, 15)
            
            output_sample_path = os.path.join(output_path, class_dir, str(sample_json).replace("json", "bmp") )
            save_array_to_bitmap(filtered_sample, output_sample_path)


def save_bitmap_to_memory(arr) -> io.BytesIO:
    scaled_arr = np.array(arr * 255, dtype=np.uint8)
    im = Image.fromarray(scaled_arr)
    with io.BytesIO as output:
        im.save(output, format="BMP")
    return output.content

if __name__ == "__main__":
    dataset_to_bitmap("./dataset", "./dataset_bitmap")