''' This script is to organise the dataset with only depth as the input.
This dataset is prepared separately as we do not have to account for the visibility map

Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020
'''

import os
import utils.io as io
import utils.csv_io as c_io
import utils.list_tool as lt
import utils.filetool as ft
import csv
import glob
import numpy as np

from os.path import expanduser

home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)

dataset_name = "suncg"

enforce_regeneration = True

src_dataset_folder = os.path.join(proj_path, "dataset", dataset_name)
dst_dataset_folder = os.path.join(proj_path, "data", dataset_name, "depth_cnn")

types = ["train", "validation", "test"]
folders = {}
for item in types:
    folder = os.path.join(dst_dataset_folder, item)
    folders[item] = folder
    if not ft.dir_exist(folder):
        os.makedirs(folder)

all_src_scenes = glob.glob(os.path.join(src_dataset_folder,"{}*".format("room")))
all_src_scenes.sort()
print(all_src_scenes)
src_scenes = [item[-7:] for item in all_src_scenes[:-5]] # leave 5 scenes for testing  exploration

# at test time
# src_scenes = {}
neigh_range = 0.1 # in meters (10 cm)
label_list = ["up","down","left","right"]


def prepare_csv_data(data):
    csv_lines = [["depth_path", "label"]]
    csv_lines = csv_lines + data
    return csv_lines


def generate_csv_per_scene(src_scene):
    print(src_scene)
    cached_csv_lines_list = []
    for i in range(len(label_list)):
        cached_csv_lines = [["depth_path", "label"]]
        cached_csv_lines_list.append(cached_csv_lines)

    src_scene_label_path = os.path.join(src_dataset_folder, src_scene, "motionlabeltwostep.json")
    if os.path.isfile(src_scene_label_path):
        skip_csv = True
        csv_file_path_lists = []
        for i in range(len(label_list)):
            csv_file_path = os.path.join(dst_dataset_folder, "{}_{}.csv".format(src_scene, label_list[i]))
            csv_file_path_lists.append(csv_file_path)
            skip_csv = skip_csv and os.path.exists(csv_file_path)
        if skip_csv and (not enforce_regeneration):
            print("csv files of scene {} exists. Skip".format(src_scene))
        else:
            print("produce csv files of scene {}".format(src_scene))
            src_scene_depth_path = os.path.join(src_dataset_folder, src_scene, "depth")
            src_scene_label_path = os.path.join(src_dataset_folder, src_scene, "motionlabeltwostep.json")
            # load the json data
            label_raw_data = io.load_json(src_scene_label_path)
            neigh_range_key = str(neigh_range)
            # for each item in label
            # data_result[radius_key][pose_id_key][status_key]["selected_label"]
            for pose_id_key in label_raw_data[neigh_range_key]:
                depth_path = os.path.join(src_scene_depth_path, "depth{}.png".format(pose_id_key))
                selected_label_list = label_raw_data[neigh_range_key][pose_id_key]["0"]["selected_label"]
                for i, label in enumerate(selected_label_list):
                    cached_csv_line = []
                    cached_csv_line.append(depth_path)
                    cached_csv_line.append(label)
                    cached_csv_lines_list[int(label)].append(cached_csv_line)
            for i in range(len(label_list)):
                c_io.save_csv(csv_file_path_lists[i], cached_csv_lines_list[i])

    else:
        print("motion label is not available yet")


def generate_splits(dataset_size, test_split,shuffle,random_seed = 0):
    # Creating data indices for training and validation splits:
    split = int(np.floor(test_split * dataset_size))
    indices = lt.shuffle_indices(dataset_size, shuffle, random_seed=0)
    train_ind, test_ind = indices[split:], indices[:split]
    return train_ind, test_ind


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
        number_per_class = [min(x, min_count) for x in count_direction]
    else:
        number_per_class = count_direction

    print("number for each class ", number_per_class)
    for i in range(len(csv_files_list)):
        indices = lt.shuffle_indices(count_direction[i], shuffle = True)
        indices = indices[:number_per_class[i]]
        items_selected = lt.get_list_items_with_indices(item_data_list[csv_files_list[i]], indices)
        item_data_selected[csv_files_list[i]] = items_selected

    return item_data_selected


def save_folder_csv(items_selected_list, test_split = 0.15, shuffle_dataset = True):
    all_items_for_train = []
    all_items_for_val = []
    all_items_for_test = []
    for key in items_selected_list:
        dataset_size = len(items_selected_list[key])
        train_indices, test_indices = generate_splits(dataset_size, test_split, shuffle_dataset)
        items_for_test = lt.get_list_items_with_indices(items_selected_list[key], test_indices)
        all_items_for_test = all_items_for_test + items_for_test

        ### prepare the val and train split
        items_for_train = lt.get_list_items_with_indices(items_selected_list[key], train_indices)
        train_dataset_size = len(items_for_train)
        train_indices, val_indices = generate_splits(train_dataset_size, test_split, shuffle_dataset)

        items_for_train_final = lt.get_list_items_with_indices(items_for_train, train_indices)
        all_items_for_train = all_items_for_train + items_for_train_final

        items_for_val = lt.get_list_items_with_indices(items_for_train, val_indices)
        all_items_for_val = all_items_for_val + items_for_val

    ## shuffle the items before saving
    print("The number of items for TRAINING: {:06d}".format(len(all_items_for_train)))
    indices = lt.shuffle_indices(len(all_items_for_train), shuffle_dataset, random_seed=0)
    all_items_for_train = lt.get_list_items_with_indices(all_items_for_train, indices)
    train_csv_lines = prepare_csv_data(all_items_for_train)
    train_file_path = os.path.join(folders["train"], 'train.csv')
    c_io.save_csv(train_file_path, train_csv_lines)

    print("The number of items for VALIDATION: {:06d}".format(len(all_items_for_val)))
    indices = lt.shuffle_indices(len(all_items_for_val), shuffle_dataset, random_seed=0)
    all_items_for_val = lt.get_list_items_with_indices(all_items_for_val, indices)
    val_csv_lines = prepare_csv_data(all_items_for_val)
    val_file_path = os.path.join(folders["validation"], 'val.csv')

    c_io.save_csv(val_file_path, val_csv_lines)

    print("The number of items for TEST: {:06d}".format(len(all_items_for_test)))
    indices = lt.shuffle_indices(len(all_items_for_test), shuffle_dataset, random_seed=0)
    all_items_for_test = lt.get_list_items_with_indices(all_items_for_test, indices)
    test_csv_lines = prepare_csv_data(all_items_for_test)
    test_file_path = os.path.join(folders["test"], 'test.csv')
    c_io.save_csv(test_file_path, test_csv_lines)


# generate all 4-directional csv files for all scenes
for scene in src_scenes:
    generate_csv_per_scene(scene)

# merge all scenes into 4 csv file, each one for each direction
csv_files_dict, direction_count_all = merge_all_scenes_to_classes(src_scenes)
print("The counts at each direction")
print(direction_count_all)

# merge 4 directional csv files with the option of data balance (using the minimal count)
data_balance = True
item_selected_list = merge_all_classes_to_dataset(csv_files_dict, balance = data_balance)

## apply ratio to each class, in order to keep the test, validation, test all balanced
save_folder_csv(item_selected_list, test_split = 0.15, shuffle_dataset=True)

