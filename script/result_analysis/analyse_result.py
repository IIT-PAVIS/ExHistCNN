"""
The script for analysing results for ECCV2020
Yiming
Updated on 27 April 2019

"""

import utils.io as io_local
import utils.filetool as ft
import utils.improc as improc
from os.path import expanduser
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler


def analyse_data_list_coverage(rooms, methods, dome_data_result, vox_num_option):
    coverage_array_dict = {}
    coverage_std_array_dict = {}
    mean_coverage_array_dict = {}
    std_coverage_array_dict = {}
    for room in rooms:
        file_path = os.path.join(proj_path, result_folder, "{}_{}_{}.json".format(dataset_name, room, vox_num_option))
        raw_data = io_local.load_json(file_path)
        coverage_array_dict_room = {}
        for start_index in raw_data:
            for method in raw_data[start_index]:
                temp_array = np.asarray(raw_data[start_index][method]["volume_size"])/float(dome_data_result[dataset_name][room])
                temp_array[temp_array > 1] = 1.0
                if method in coverage_array_dict:
                    coverage_array_dict[method] = np.vstack((coverage_array_dict[method], temp_array))
                else:
                    coverage_array_dict[method] = temp_array
                if method in coverage_array_dict_room:
                    coverage_array_dict_room[method] = np.vstack((coverage_array_dict_room[method], temp_array))
                else:
                    coverage_array_dict_room[method] = temp_array

        # compute the std for each room, the final one will be averaged by rooms
        for method in methods:

            temp_std_array = np.std(coverage_array_dict_room[method], axis=0)
            if method in coverage_std_array_dict:
                coverage_std_array_dict[method] = np.vstack((coverage_std_array_dict[method], temp_std_array))
            else:
                coverage_std_array_dict[method] = temp_std_array

    for method in methods:
        mean_coverage_array_dict[method] = np.mean(coverage_array_dict[method], axis=0)
        std_coverage_array_dict[method] = np.mean(coverage_std_array_dict[method], axis=0)
    return mean_coverage_array_dict, std_coverage_array_dict


def analyse_data_list_computation(rooms, methods, key, vox_num_option):
    computation_array_dict = {}
    mean_computation_array_dict = {}
    std_computation_array_dict = {}
    # data_result[start_index_key][method]["volume_size"] = volume_sizes
    for room in rooms:
        file_path = os.path.join(proj_path, result_folder, "{}_{}_{}.json".format(dataset_name, room,vox_num_option))
        raw_data = io_local.load_json(file_path)
        for start_index in raw_data:
            for method in raw_data[start_index]:
                temp_array = np.asarray(raw_data[start_index][method][key])
                if method in computation_array_dict:
                    computation_array_dict[method] = np.vstack((computation_array_dict[method], temp_array))
                else:
                    computation_array_dict[method] = temp_array
    for method in methods:
        mean_computation_array_dict[method] = np.mean(computation_array_dict[method], axis=0)
        std_computation_array_dict[method] = np.std(computation_array_dict[method], axis=0)
    return mean_computation_array_dict, std_computation_array_dict


def get_label_name(method):
    if method == "CNNmemo":
        label_name = "CNNcuboid"
    elif method == "CNNmemoResNet":
        label_name = "CNNResConcat"
    elif method == "GainLargeTwo":
        label_name = "BaseGain"
    elif method == "BaseInfoGain":
        label_name = "UnknownCount"
    elif method == "GainLargeTwoRec":
        label_name = "BaseGainRec"
    elif method == "CNNmemoTwoStep5D":
        label_name = "CNN5D"
    elif method == "CNNmemoTwoStep2D":
        label_name = "CNN2D"
    elif method == "CNNmemoTwoStep2DScaled":
        label_name = "CNN2DScaled"
    elif method == "CNNmemoTwoStepUtilityOnly":
        label_name = "CNNUtility"
    elif method == "KnownDepthTwo":
        label_name = "OracleDepth2"
    elif method == "KnownDepth":
        label_name = "OracleDepth"
    elif method == "KnownDepthThree":
        label_name = "OracleDepth3"
    elif method == "CNNMLPTwoStep2D":
        label_name = "MLP2D"
    elif method == "CNNMLPTwoStep2DScaled":
        label_name = "MLP2DScaled"
    elif method == "CNNMLPTwoStep5D":
        label_name = "MLP5D"
    elif method == "CNNMLPTwoStep4D":
        label_name = "MLP4D"
    elif method == "CNNmemoTwoStep4D":
        label_name = "CNN4D"
    elif method == "CNNCombInfoGain":
        label_name = "CombGain"
    else:
        label_name = method
    return label_name

home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)

dataset_name = "suncg" #"mp3d" #"suncg"
config = io_local.load_yaml(os.path.join(proj_path, "config.yml"))


all_room_names = ft.grab_directory(os.path.join(proj_path, config["dataset_folder"],dataset_name), extra_key="room0")
all_room_names.sort()
selected_room_names = all_room_names[-6:]

NBV_strategy_ablation = ["KnownDepth","KnownDepthTwo", "KnownDepthThree", "GainLargeTwo", "GainLargeTwoRec"]
NBV_strategy_comparison = ["KnownDepthTwo", "GainLargeTwo", "BaseInfoGain", "CNNCombInfoGain","Random", "CNNdepth", "CNNmemoTwoStepUtilityOnly", "CNNmemoTwoStep4D", "CNNmemoTwoStep2D", "CNNmemoTwoStep5D", "CNNMLPTwoStep4D", "CNNMLPTwoStep2D", "CNNMLPTwoStep5D"]

NBV_strategy = ["KnownDepth","KnownDepthTwo", "KnownDepthThree", "BaseInfoGain", "GainLargeTwo", "GainLargeTwoRec", "CNNCombInfoGain", "Random", "CNNdepth", "CNNmemoTwoStepUtilityOnly", "CNNmemoTwoStep4D","CNNmemoTwoStep2D", "CNNmemoTwoStep5D","CNNMLPTwoStep4D", "CNNMLPTwoStep2D", "CNNMLPTwoStep5D"] #"CNNmemoTwoStep5D", "Random", "CNNdepth", "CNNmemo", "CNNmemoResNet", "BaseInfoGain", "InfoGain"] # "Random", "CNNdepth", "CNNmemo", "CNNmemoResNet", "InfoGain","BaseInfoGain", "GT"
NBV_strategy_time = ["GainLargeTwo", "BaseInfoGain", "CNNCombInfoGain", "Random", "CNNdepth", "CNNmemoTwoStepUtilityOnly", "CNNmemoTwoStep2D", "CNNmemoTwoStep5D","CNNMLPTwoStep2D", "CNNMLPTwoStep5D"] #"CNNmemoTwoStep5D", "Random", "CNNdepth", "CNNmemo", "CNNmemoResNet", "BaseInfoGain", "InfoGain"] # "Random", "CNNdepth", "CNNmemo", "CNNmemoResNet", "InfoGain","BaseInfoGain", "GT"
NBV_strategy_mp3d = ["KnownDepth", "KnownDepthTwo", "BaseInfoGain", "GainLargeTwo", "GainLargeTwoRec", "CNNCombInfoGain", "Random", "CNNdepth", "CNNmemoTwoStepUtilityOnly", "CNNmemoTwoStep4D","CNNmemoTwoStep2D", "CNNmemoTwoStep5D","CNNMLPTwoStep4D", "CNNMLPTwoStep2D", "CNNMLPTwoStep5D"] #"CNNmemoTwoStep5D", "Random", "CNNdepth", "CNNmemo", "CNNmemoResNet", "BaseInfoGain", "InfoGain"] # "Random", "CNNdepth", "CNNmemo", "CNNmemoResNet", "InfoGain","BaseInfoGain", "GT"

start_index_list = [45, 168, 250, 313, 467]

vox_num_option = "o3d"

time_plot_key = ["nbv selection", "octomap update"]

result_folder = "result"

# prepare the path for the visualised image
result_coverage_time = os.path.join(proj_path, result_folder, "with_time_cov_growth_{}_{}.png".format(dataset_name, vox_num_option))
result_coverage_time_ablation = os.path.join(proj_path, result_folder, "with_time_cov_growth_ablation_{}_{}.png".format(dataset_name, vox_num_option))
result_coverage_time_comparison = os.path.join(proj_path, result_folder, "with_time_cov_growth_comparison_{}_{}.png".format(dataset_name, vox_num_option))
result_coverage_std_comparison = os.path.join(proj_path, result_folder, "cov_std_comparison_{}_{}.png".format(dataset_name, vox_num_option))
result_ocotomap_update_time = os.path.join(proj_path, result_folder, "octomap_update_time_{}_{}.png".format(dataset_name, vox_num_option))
result_NBV_time = os.path.join(proj_path, result_folder, "NBV_time_{}_{}.png".format(dataset_name, vox_num_option))

# load dome data as GT exploration performance
dome_result_file = os.path.join(proj_path, result_folder, "dome_volume_{}.json".format(vox_num_option))
dome_data_result = io_local.load_json(dome_result_file)

# load meta result data
if dataset_name == "mp3d":
    NBV_strategy = NBV_strategy_mp3d
mean_coverage_array_dict, std_coverage_array_dict = analyse_data_list_coverage(selected_room_names, NBV_strategy, dome_data_result, vox_num_option)
mean_update_octomap_time_dict, std_update_octomap_time_dict = analyse_data_list_computation(selected_room_names, NBV_strategy, "time_octomap_update", vox_num_option)
mean_NBV_time_dict, std_NBV_time_dict = analyse_data_list_computation(selected_room_names, NBV_strategy, "time_NBV", vox_num_option)


# start visualisation
linestyle_all = ['-', '--', ':', '-.', (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5))]
markers_all = ["X", "s", "h", "D", "H", "8", "o", "v", "P", "*", "+" , "2","1","3","4","|","_"]
# pdb.set_trace()
color_unit = int(255.0/len(NBV_strategy))
if len(NBV_strategy)>10:
    colors1 = plt.get_cmap("tab10")(range(10))
    colors2 = plt.get_cmap("tab10")(range(len(NBV_strategy)-10))
    colors = np.vstack((colors1, colors2))
else:
    colors = plt.get_cmap("tab10")(range(len(NBV_strategy)))#(range(0, len(NBV_strategy)*color_unit, color_unit ))
style_cycler = cycler(color=colors, marker=markers_all[:len(NBV_strategy)]) * cycler(linestyle=linestyle_all[:1])
#
plt.rc('axes', prop_cycle=style_cycler)

if dataset_name == "suncg":
    print("Save to figure for ablation growth ... ")
    font_size = 40

    plt.figure(figsize=(28, 22), dpi=100, facecolor='w', edgecolor='k')
    rooms = selected_room_names
    for method in NBV_strategy_ablation:
        mean_temp = mean_coverage_array_dict[method]
        label_name = get_label_name(method)
        plt.plot(mean_temp, label=label_name, linewidth=6, markersize= 25, markevery=10)

    plt.xlabel('#Steps', fontsize=font_size + 3)
    plt.ylabel("Coverage ratio ", fontsize=font_size + 3)
    plt.xlim(left=0)
    plt.ylim(top=1.0)
    #plt.grid()
    matplotlib.pyplot.xticks(range(0, 151, 10), fontsize=font_size + 3)
    matplotlib.pyplot.yticks([0.1 * j for j in range(0, 11, 1)], fontsize=font_size + 3)
    plt.legend(loc="lower right", ncol=1, prop={'size': font_size + 20})

    plt.savefig(result_coverage_time_ablation)
    img = improc.crop_border(result_coverage_time_ablation, 0.1, 0.1)
    improc.imwrite(result_coverage_time_ablation, img)
    plt.clf()

print("Save to figure for comparison growth ... ")
font_size = 43

plt.figure(figsize=(28, 22), dpi=100, facecolor='w', edgecolor='k')
rooms = selected_room_names
for method in NBV_strategy_comparison:
    mean_temp = mean_coverage_array_dict[method]
    label_name = get_label_name(method)
    plt.plot(mean_temp, label=label_name, linewidth=6, markersize= 25, markevery=10)

    # x_array = np.arange(0, mean_temp.shape[0])
    # std_temp = std_coverage_array_dict[method]
    # plt.errorbar(x_array, mean_temp, yerr=std_temp, elinewidth=6, errorevery=10)


plt.xlabel('#Steps', fontsize=font_size + 3)
plt.ylabel("Coverage ratio ", fontsize=font_size + 3)
plt.xlim(left=0)
plt.ylim(top=1.0)

matplotlib.pyplot.xticks(range(0, 151, 10), fontsize=font_size + 3)
matplotlib.pyplot.yticks([0.1 * j for j in range(0, 11, 1)], fontsize=font_size + 3)
plt.legend(loc="upper center", ncol=5, prop={'size': font_size-13})

plt.savefig(result_coverage_time_comparison)
img = improc.crop_border(result_coverage_time_comparison, 0.1, 0.1)
improc.imwrite(result_coverage_time_comparison, img)
plt.clf()

print("Save to figure for comparison std ... ")
font_size = 43

plt.figure(figsize=(28, 22), dpi=100, facecolor='w', edgecolor='k')
rooms = selected_room_names
count = 0

line1 = ""
line2 = ""

for method in NBV_strategy_comparison:
    mean_temp = mean_coverage_array_dict[method]
    std_temp = std_coverage_array_dict[method]
    label_name = get_label_name(method)
    plt.errorbar(count, mean_temp[-1], yerr=std_temp[-1], elinewidth=6, capsize= 25, markersize= 25, label=label_name)
    if count == len(NBV_strategy_comparison) - 1:
        line1 = line1 + label_name + "\\"
        line2 = line2 + "{:.02f}({:.02f})".format(mean_temp[-1],std_temp[-1]) + "\\"
    else:
        line1 = line1 + label_name + "&"
        line2 = line2 + "{:.02f}({:.02f})".format(mean_temp[-1],std_temp[-1]) + "&"
    count = count + 1
print(line1)
print(line2)

plt.xlabel('Method', fontsize=font_size + 3)
plt.ylabel("Coverage ratio ", fontsize=font_size + 3)

plt.ylim(top=1.1)

matplotlib.pyplot.xticks(range(len(NBV_strategy_comparison)), fontsize=font_size + 3)
matplotlib.pyplot.yticks([0.1 * j for j in range(0, 12, 1)], fontsize=font_size + 3)


plt.savefig(result_coverage_std_comparison)
img = improc.crop_border(result_coverage_std_comparison, 0.1, 0.1)
improc.imwrite(result_coverage_std_comparison, img)
plt.clf()

print("Save to figure for all growth ... ")
font_size = 38

plt.figure(figsize=(28, 22), dpi=100, facecolor='w', edgecolor='k')
rooms = selected_room_names
for method in NBV_strategy:
    mean_temp = mean_coverage_array_dict[method]
    label_name = get_label_name(method)
    plt.plot(mean_temp, label=label_name, linewidth=6, markersize= 25, markevery=10)

plt.xlabel('#Steps', fontsize=font_size + 3)
plt.ylabel("Coverage ratio ", fontsize=font_size + 3)
plt.xlim(left=0)
plt.ylim(top=1.1)
plt.grid()
matplotlib.pyplot.xticks(range(0, 151, 10), fontsize=font_size + 3)
matplotlib.pyplot.yticks([0.1 * j for j in range(0, 12, 1)], fontsize=font_size + 3)
plt.legend(loc="lower center", ncol=4, prop={'size': font_size -5})

plt.savefig(result_coverage_time)
img = improc.crop_border(result_coverage_time, 0.1, 0.1)
improc.imwrite(result_coverage_time, img)
plt.clf()


print("Save to figure for computational time - octomap update... ")

font_size = 38

plt.figure(figsize=(28, 22), dpi=100, facecolor='w', edgecolor='k')
rooms = selected_room_names
for method in NBV_strategy_time:
    mean_temp = mean_update_octomap_time_dict[method]

    label_name = get_label_name(method)
    plt.plot(mean_temp, label=label_name, linewidth=6, markersize= 25, markevery=10)

plt.xlabel('#Steps', fontsize=font_size + 3)
plt.ylabel("Octomap update time ", fontsize=font_size + 3)
plt.xlim(left=0)
plt.ylim(top=1.5)

matplotlib.pyplot.xticks(range(0, 151, 10), fontsize=font_size + 3)
matplotlib.pyplot.yticks([0.1 * j for j in range(0, 15, 1)], fontsize=font_size + 3)
plt.legend(loc="upper center", ncol=3, prop={'size': font_size -3})

plt.savefig(result_ocotomap_update_time)
img = improc.crop_border(result_ocotomap_update_time, 0.1, 0.1)
improc.imwrite(result_ocotomap_update_time, img)
plt.clf()


print("Save to figure for computational time - NBV... ")
font_size = 38

plt.figure(figsize=(28, 22), dpi=100, facecolor='w', edgecolor='k')
rooms = selected_room_names
line1 = ""
line2 = ""
count = 0
for method in NBV_strategy_time:

    mean_temp = mean_NBV_time_dict[method]
    mean_value = np.mean(mean_temp)
    std_value = np.std(mean_temp)
    threshold = mean_value+std_value
    idx = np.argwhere(mean_temp > threshold)
    mean_temp[idx] = mean_value

    label_name = get_label_name(method)
    plt.plot(mean_temp[2:], label=label_name, linewidth=6, markersize= 25, markevery=10)
    if count == len(NBV_strategy_time)-1:
        line1 = line1+label_name+"\\"
        line2 = line2+"{:.02f}".format(mean_value)+"\\"
    else:
        line1 = line1+label_name+"&"
        line2 = line2 + "{:.02f}".format(mean_value) + "&"
    count = count+1
print(line1)
print(line2)
plt.xlabel('#Steps', fontsize=font_size + 3)
plt.ylabel("NBV time ", fontsize=font_size + 3)
plt.xlim(left=0)
plt.ylim(top= 0.13)

matplotlib.pyplot.xticks(range(0, 151, 10), fontsize=font_size + 3)
matplotlib.pyplot.yticks([0.02 * j for j in range(0, 8, 1)], fontsize=font_size + 3)
plt.legend(loc="upper center", ncol=3, prop={'size': font_size -3})

plt.savefig(result_NBV_time)
img = improc.crop_border(result_NBV_time, 0.1, 0.1)
improc.imwrite(result_NBV_time, img)
plt.clf()
