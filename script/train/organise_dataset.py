''' This script is to organise the dataset into train, validation, test split
Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020

# The workflow follows:
1) generate all 4-directional csv files for each scene

2) merge all scenes into 4 csv file, each one for each direction

3) merge all 4 directional csv files with the option of data balance (using the minimal count) to one file

4) apply ratio to each class, in order to keep the test, validation, test all balanced after shuffling

The final outputs and intermediate outputs are all in the format as .csv

'''

import os
import utils.io as io
import utils.csv_io as c_io
import utils.list_tool as lt
import numpy as np
import utils.filetool as ft
from os.path import expanduser

enforce_regeneration = False

home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)

config = io.load_yaml(os.path.join(proj_path, "config.yml"))

FOV_DIS = float(config["sensor_max_dis"])
FOV_H = float(config["sensor_angle_pan"]) * np.pi
FOV_V = float(config["sensor_angle_tilt"]) * np.pi

dataset_name = "mp3d"

# folder where to find the
src_dataset_folder = os.path.join(proj_path, config["dataset_folder"], dataset_name)
dst_dataset_folder = os.path.join(proj_path, "data", dataset_name, "dataset_cnn_twostep")


train_folder = os.path.join(dst_dataset_folder, "train")
val_folder = os.path.join(dst_dataset_folder, "validation")
test_folder = os.path.join(dst_dataset_folder, "test")

# check the destination folders are already there
if not ft.dir_exist(train_folder):
    print("generate a train folder")
    os.makedirs(train_folder)

if not ft.dir_exist(val_folder):
    print("generate a val folder")
    os.makedirs(val_folder)

if not ft.dir_exist(test_folder):
    print("generate a test folder")
    os.makedirs(test_folder)

all_room_names = ft.grab_directory(os.path.join(proj_path, config["dataset_folder"], dataset_name))
all_room_names.sort()

# set among the all rooms the ones for dataset generation
if dataset_name == "mp3d":
    selected_room_names = all_room_names
else:
    selected_room_names = all_room_names[:-6] # excluding the last 6 rooms are left for testing NBV, not included inside the training

radius = 0.1 # in meter
label_list = ["up", "down", "left", "right"]


# Set the FoV for utility map generation
# @Input: range of the neighbourhood
# @Output: the horizontal and vertical FoV angle
def set_vis_fov(r_range):
    if r_range == 0.05:
        k = 0.1
        g = 0.1
        fov_h = FOV_H+ k*np.pi
        fov_v = FOV_V+ g*np.pi
    elif r_range == 0.1:
        k = 0.28
        g = 0.3
        fov_h = FOV_H + k*np.pi
        fov_v = FOV_V + g*np.pi
    elif r_range == 0.15:
        k = 0.3
        g = 0.3
        fov_h = FOV_H+ k*np.pi
        fov_v = FOV_V+ g*np.pi

    return fov_h, fov_v


# prepare the lines to to write to csv files
# @Input: data to be writeen
# @Output: combined version with the header
def prepare_csv_data(data):
    csv_lines = [["depth_path", "vismap_twostep", "label"]]
    csv_lines = csv_lines + data
    return csv_lines


# produce the csv per scene for each direction
# @Input: scene name
def generate_csv_per_scene_twostep(src_scene):
    print(src_scene)
    fov_h, fov_v = set_vis_fov(radius)

    cached_csv_lines_list = []
    for i in range(len(label_list)):
        cached_csv_lines = [["depth_path", "vismap_twostep", "label"]]
        cached_csv_lines_list.append(cached_csv_lines)

    src_scene_label_path = os.path.join(src_dataset_folder, src_scene, "motionlabeltwostep.json")

    if os.path.isfile(src_scene_label_path):
        skip_csv = True
        csv_file_path_lists = []
        for i in range(len(label_list)):
            csv_file_path = os.path.join(dst_dataset_folder, "{}_{}.csv".format(src_scene, label_list[i]))
            print(csv_file_path)
            csv_file_path_lists.append(csv_file_path)
            skip_csv = skip_csv and os.path.exists(csv_file_path)
        if skip_csv and (not enforce_regeneration):
            print("csv files of scene {} exists. Skip".format(src_scene))
        else:
            print("produce csv files of scene {}".format(src_scene))
            src_scene_depth_path = os.path.join(src_dataset_folder, src_scene, "depth")
            src_scene_vismap_path = os.path.join(src_dataset_folder, src_scene, "{}_{}".format(config["visibility_folder"], str(int(radius*100))))

            # load the json data
            label_raw_data = io.load_json(src_scene_label_path)
            radius_key = str(radius)
            # iterate over each pose
            for pose_id_key in label_raw_data[radius_key]:
                depth_path = os.path.join(src_scene_depth_path, "depth{}.png".format(pose_id_key))
                # iterate over each status
                for status_key in label_raw_data[radius_key][pose_id_key]:
                    status = float(status_key)
                    selected_label_list = label_raw_data[radius_key][pose_id_key][status_key]["selected_label"]
                    # iterate over each label for each combinations of neighbouring poses achieving the status
                    for i, label in enumerate(selected_label_list):
                        cached_csv_line = []
                        cached_csv_line.append(depth_path)
                        vis_map_prefix = os.path.join(src_scene_vismap_path, "pose_{}_recons_{:d}_H{:d}_V{:d}".format(pose_id_key,int(status * 100), int(
                                                              round(fov_h / np.pi * 180)), int(
                                                              round(fov_v / np.pi * 180))))
                        vis_map_path = os.path.join(src_scene_vismap_path, "{}_comb_{:d}.png".format(vis_map_prefix, i))
                        cached_csv_line.append(vis_map_path)
                        cached_csv_line.append(label)
                        cached_csv_lines_list[int(label)].append(cached_csv_line)
            for i in range(len(label_list)):
                c_io.save_csv(csv_file_path_lists[i], cached_csv_lines_list[i])
    else:
        print("motion label is not available yet")


# prepare two splits with the shuffling option
# @Input: size of dataset_to be split
#         split ratio for the part 2
#         boolean to shuffle
#         seed for random process
# @Output: indices list of part 1 and part 2.
def generate_splits(dataset_size,test_split,shuffle,seed = 0):
    # Creating data indices for training and validation splits:
    split = int(np.floor(test_split * dataset_size))
    indices = lt.shuffle_indices(dataset_size, shuffle, random_seed = seed)
    train_ind, test_ind = indices[split:], indices[:split]
    return train_ind, test_ind


# merge all scenes per direction
# @Input: scene names
# @Output: output file list, and the number of items per direction
def merge_all_scenes_to_classes(src_scenes):
    items_for_all_scenes_list = []
    csv_file_path_list = []
    direction_count = []
    for i in range(len(label_list)):
        items_for_all_scenes = []
        items_for_all_scenes_list.append(items_for_all_scenes)
    for scene in src_scenes:
        for i in range(len(label_list)):
            csv_file_path = os.path.join(dst_dataset_folder, "{}_{}.csv".format(scene,label_list[i]))
            if os.path.exists(csv_file_path):
                temp_data = c_io.read_csv(csv_file_path)
                items_for_all_scenes_list[i] = items_for_all_scenes_list[i] + temp_data[1:]  # concatenate two lists together, each list remove the colume names

    for i in range(len(label_list)):
        csv_file_path = os.path.join(dst_dataset_folder, "{}.csv".format(label_list[i]))
        csv_file_path_list.append(csv_file_path)
        direction_count.append(len(items_for_all_scenes_list[i]))
        csv_lines = prepare_csv_data(items_for_all_scenes_list[i])
        c_io.save_csv(csv_file_path, csv_lines)

    return csv_file_path_list, direction_count


# merge all classes into balanced and shuffled single file
# @Input: list of csv files to be merged
# @Output: merged data list
def merge_all_classes_to_dataset(csv_files_list, balance = True):
    count_direction = []
    item_data_list = {}
    item_data_selected = {}
    for i in range(len(csv_files_list)):
        csv_file_path = csv_files_list[i]
        temp_data = c_io.read_csv(csv_file_path)
        items_data = temp_data[1:] # ignore the first row with labels
        count_direction.append(len(items_data))
        item_data_list[csv_files_list[i]] = items_data

    if balance:
        min_count = min(count_direction)
        print("old min", min_count)
        min_count = min(min_count, 100000)
        print("capped min", min_count)
        number_per_class = [min(x, min_count) for x in count_direction]
    else:
        number_per_class = count_direction

    print("number for each class ",number_per_class )
    for i in range(len(csv_files_list)):
        indices = lt.shuffle_indices(count_direction[i], shuffle = True)
        indices = indices[:number_per_class[i]]
        items_selected = lt.get_list_items_with_indices(item_data_list[csv_files_list[i]], indices)
        item_data_selected[csv_files_list[i]] = items_selected

    return item_data_selected


# main function to split and save into folders
# @Input: all items to be split
#         test_split ratio
#         boolean for shuffle_dataset
# @Output: merged data list
def save_folder_csv(items_selected_list, test_split = 0.15, shuffle_dataset = True):
    all_items_for_train = []
    all_items_for_val = []
    all_items_for_test = []
    for key in items_selected_list:
        dataset_size = len(items_selected_list[key])
        train_indices, test_indices = generate_splits(dataset_size, test_split, shuffle_dataset)
        items_for_test = lt.get_list_items_with_indices(items_selected_list[key], test_indices)
        all_items_for_test = all_items_for_test + items_for_test

        # prepare the val and train split
        items_for_train = lt.get_list_items_with_indices(items_selected_list[key], train_indices)
        train_dataset_size = len(items_for_train)
        train_indices, val_indices = generate_splits(train_dataset_size, test_split, shuffle_dataset)

        items_for_train_final = lt.get_list_items_with_indices(items_for_train, train_indices)
        all_items_for_train = all_items_for_train + items_for_train_final

        items_for_val = lt.get_list_items_with_indices(items_for_train, val_indices)
        all_items_for_val = all_items_for_val + items_for_val

    # shuffle the items before saving
    print("The number of items for TRAINING: {:06d}".format(len(all_items_for_train)))
    indices = lt.shuffle_indices(len(all_items_for_train), shuffle_dataset, random_seed=0)
    all_items_for_train = lt.get_list_items_with_indices(all_items_for_train, indices)
    train_csv_lines = prepare_csv_data(all_items_for_train)
    train_file_path = os.path.join(train_folder, 'train.csv')
    print(train_file_path)
    c_io.save_csv(train_file_path, train_csv_lines)

    print("The number of items for VALIDATION: {:06d}".format(len(all_items_for_val)))
    indices = lt.shuffle_indices(len(all_items_for_val), shuffle_dataset, random_seed=0)
    all_items_for_val = lt.get_list_items_with_indices(all_items_for_val, indices)
    val_csv_lines = prepare_csv_data(all_items_for_val)
    val_file_path = os.path.join(val_folder, 'val.csv')
    print(val_file_path)
    c_io.save_csv(val_file_path, val_csv_lines)

    print("The number of items for TEST: {:06d}".format(len(all_items_for_test)))
    indices = lt.shuffle_indices(len(all_items_for_test), shuffle_dataset, random_seed=0)
    all_items_for_test = lt.get_list_items_with_indices(all_items_for_test, indices)
    test_csv_lines = prepare_csv_data(all_items_for_test)
    test_file_path = os.path.join(test_folder, 'test.csv')
    print(test_file_path)
    c_io.save_csv(test_file_path, test_csv_lines)


# create all necessary folders
folder_list = [train_folder, val_folder, test_folder]

for folder in folder_list:
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('Created directory: ', folder)

# generate all 4-directional csv files for all scenes
for scene in selected_room_names:
    generate_csv_per_scene_twostep(scene)

# merge all scenes into 4 csv file, each one for each direction
csv_files_dict, direction_count_all = merge_all_scenes_to_classes(selected_room_names)
print(csv_files_dict)
print("The counts at each direction")
print(direction_count_all)

# merge 4 directional csv files with the option of data balance (using the minimal count)
data_balance = True
item_selected_list = merge_all_classes_to_dataset(csv_files_dict, balance = data_balance)

# apply ratio to each class, in order to keep the test, validation, test all balanced
save_folder_csv(item_selected_list, test_split = 0.15, shuffle_dataset=True)

# random for test and validation
# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
