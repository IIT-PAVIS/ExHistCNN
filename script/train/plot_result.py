import numpy as np
import matplotlib.pyplot as plt
import os

from os.path import expanduser

home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)


def load_train_result(path):
    result_data = np.load(path)

    acc_val=result_data["acc_val"]
    acc_train=result_data["acc_train"]
    loss_val = result_data["loss_val"]
    loss_train=result_data["loss_train"]

    return acc_val,acc_train,loss_val,loss_train


def check_option_name(input_option):
    if input_option == "CNNMLPTwoStep2D":
        option = "MLPTwoStep2D"
        title_name = "MLP2D"
    elif input_option == "CNNMLPTwoStep2DScaled":
        option = "MLPTwoStep2DScaled"
        title_name = "MLP2DScaled"
    elif input_option == "CNNMLPTwoStep5D":
        option = "MLPTwoStep5D"
        title_name = "MLP5D"
    elif input_option == "CNNMLPTwoStep4D":
        option = "MLPTwoStep4D"
        title_name = "MLP4D"
    elif input_option == "CNNmemoTwoStep4D":
        option = "CuboidTwoStep4D"
        title_name = "CNN4D"
    elif input_option =="CNNmemoTwoStep5D":
        option = "CuboidTwoStep5D"
        title_name = "CNN5D"
    elif input_option == "CNNmemoTwoStep2D":
        option = "CuboidTwoStep2D"
        title_name = "CNN2D"
    elif input_option == "CNNmemoTwoStep2DScaled":
        option = "CuboidTwoStep2DScaled"
        title_name = "CNN2DScaled"
    elif input_option == "CNNmemoTwoStepUtilityOnly":
        option = "CuboidTwoStepUtilityOnly"
        title_name = "CNNUtility"
    elif input_option == "CNNdepth":
        option = "Depth"
        title_name = "CNNDepth"

    return option, title_name

classes = ("up","down","left","right")


NBV_strategy = ["CNNMLPTwoStep2D", "CNNMLPTwoStep2DScaled", "CNNMLPTwoStep5D","CNNMLPTwoStep4D", "CNNmemoTwoStep4D","CNNmemoTwoStep5D", "CNNmemoTwoStep2D", "CNNmemoTwoStep2DScaled", "CNNmemoTwoStepUtilityOnly","CNNdepth"]

plot_types = ['accuracy']

for strategy in NBV_strategy:
    epoch_num = 199
    stop_epoch_num = 199
    option,title_name = check_option_name(strategy)

    acc_loss_file_path = os.path.join(proj_path, "checkpoint", option, "training_result_{:03d}.npz".format(epoch_num))
    if not os.path.exists(acc_loss_file_path):
        epoch_num = 149
        stop_epoch_num = 149
        acc_loss_file_path = os.path.join(proj_path, "checkpoint", option, "training_result_{:03d}.npz".format(epoch_num))

    for plot_type in plot_types:
        result_fig_file = os.path.join(proj_path, "result", "{}_{}_{:03d}.png".format(plot_type, option, stop_epoch_num))

        if os.path.exists(result_fig_file):
            print("Skip ... ")
        else:
            print("Load data from {}".format(acc_loss_file_path))
            acc_val, acc_train, loss_val, loss_train = load_train_result(acc_loss_file_path)
            epoch = np.arange(1, stop_epoch_num+1, 1)

            fig = plt.figure(num=None, figsize=(8, 8), dpi=100)
            if plot_type == "loss":
                val_data = np.squeeze(loss_val[:stop_epoch_num])
                train_data = np.squeeze(loss_train[:stop_epoch_num])
            else:
                val_data = np.squeeze(acc_val[:stop_epoch_num])
                train_data = np.squeeze(acc_train[:stop_epoch_num])

            plt.plot(epoch, val_data, 'b--', label="Validation")
            plt.plot(epoch, train_data, 'r--', label="Train")

            axis_min = min(np.amin(val_data), np.amin(train_data))
            axis_max = max(np.amax(val_data), np.amax(train_data))

            plt.axis([1, stop_epoch_num+1, axis_min-0.1, axis_max+0.1])
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.legend(fontsize = 25,loc='lower right',ncol = 2)
            #plt.ylabel(plot_type,fontsize = 25)
            plt.xlabel('epochs',fontsize = 25)


            plt.title(title_name,fontsize = 25)

            plt.savefig(result_fig_file, transparent=False)
            plt.close()
    #plt.show()

#print(fig = plt.gcf())


