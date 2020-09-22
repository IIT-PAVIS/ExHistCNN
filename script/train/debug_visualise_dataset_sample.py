
from network import ToTensor
from network import TwoStepH5Loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from os.path import expanduser
import os
home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)

label_list = ["up","down","left","right"]
batch_size = 1024
number_workers = 8
transform = transforms.Compose([ToTensor()])
drop_last_batch = True
option = "2DScaled"

dataset_name = "suncg"

print("Using H5 data")
h5_file = os.path.join(proj_path, 'data', dataset_name, 'dataset_cnn_twostep', 'train', 'train.h5')
csv_file = os.path.join(proj_path, 'data', dataset_name, 'dataset_cnn_twostep', 'train', 'train.csv')
data_loader_kwargs_h5 = {'option': option, 'h5_file': h5_file, 'csv_file': csv_file, 'transform': transform,
                         'batch_size': batch_size, 'shuffle': True, 'drop_last': drop_last_batch,
                         'num_workers': number_workers, }
valloader = TwoStepH5Loader(**data_loader_kwargs_h5)  # batch size is usually set to 4, for debug, we can use 1

print("data loader finished!")

# create an image folder to save the visualisation
# create the result folder if not exists
if os.path.exists(os.path.join(proj_path, "images", option)):
    print(os.path.join(proj_path, "images", option))
    print("image folder exists ..")
else:
    print("generate a image folder ...")
    os.mkdir(os.path.join(proj_path, "images", option))


def draw_cuboid_once(data, label, index):
    if torch.is_tensor(data):
        data_np = data.numpy()
        print("In Tensor type, convert to numpy data type first")
    else:
        data_np = data

    data_np = data_np.transpose(1, 2, 0)
    num_plt = data_np.shape[2]
    for j in range( num_plt ):
        plt.subplot(1,  num_plt , j+1)
        if j == 0:
            plt.title('Sample #{} with label {}'.format(index, label_list[label]))
        plt.imshow(data_np[:,:,j])

    plt.savefig(os.path.join(proj_path, "images", option, "sample{:d}.png").format(index))

    plt.pause(1)
    plt.clf()


for i_batch, sample_batched in enumerate(valloader):
    images_batch, labels_batch = sample_batched["sample"], sample_batched["label"]
    print(i_batch,  images_batch.size(),
          labels_batch.size())

    batch_size = len(images_batch)
    print("batch size", batch_size)

    if i_batch <= 10:
        for batch_index in range(batch_size):
            print("batch index {:03d}".format(batch_index))
            draw_cuboid_once(images_batch[batch_index], labels_batch[batch_index], batch_index)
    else:
        print("Other batch")









