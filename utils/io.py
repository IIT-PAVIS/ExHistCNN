import yaml
import numpy as np
import scipy.io as sio
import json
import os
import transformation as tf

def load_yaml(file_name):
    with open(file_name,'r') as fid:
        try:
            config = yaml.safe_load(fid)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def load_json(file_name):
    if os.path.isfile(file_name):
        if os.stat(file_name).st_size == 0:
            print("WARN: Empty json file!")
            return {}
        else:
            with open(file_name) as json_file:
                data = json.load(json_file)
            return data
    else:
        print("WARN: json file does not exist!")
        return {}
        

def write_json(file_name,data):
    with open(file_name, 'w') as outfile:
        try:
            json.dump(data, outfile) 
        except : # whatever reader errors you care about
              # handle error
            print("Exception occured!")
        
         
def load(filename, deli = " "):
    """
    Load file.
    Supported formats:
        - TXT / LOG
        - YAML
        - MAT
        - NPY
        - NPZ
        - PLY
    
    Parameters
    ----------
    filename : string
        Path of the file to load
    
    Returns
    -------
    data : depends on file format
    """

    ext = filename.split(".")[-1].lower()

    if ext in ["txt", "log"]:
        try:
            data = np.loadtxt(filename, delimiter = deli)
        except:
            print("Np data reading exception:: read as string")
            data = np.loadtxt(filename, str)
            

    elif ext in ["yml", "yaml"]:
        with open(filename, "r") as fp:
            data = yaml.load(fp)

    elif ext == "mat":
        data = sio.loadmat(filename)
        for k in data.keys():
            data[k] = np.squeeze(data[k])

    elif ext == "npy":
        data = np.load(filename)
        if len(data.shape) == 0:
            data = data.any()

    elif ext == "npz":
        data = np.load(filename)
        data = data["data"]
        data = data.any()

    else:
        raise NotImplementedError()

    return data


def save(filename, data):
    """
    Save data into a file.
    Supported formats:
        - TXT / LOG
        - YAML
        - MAT
        - NPY
        - NPZ
        - PLY
    
    Parameters
    ----------
    filename : string
        Destination filename
    data : type depends on output file extension
        Values to store formatted according to the desired output file format
    """

    ext = filename.split(".")[-1].lower()

    if ext in ["txt", "log"]:
        np.savetxt(filename, data, fmt="%s")

    elif ext in ["yml", "yaml"]:
        with open(filename, "w") as fp:
            yaml.dump(data, fp)

    elif ext == "mat":
        sio.savemat(filename, data, do_compression=True)

    elif ext == "npy":
        np.save(filename, data)

    elif ext == "npz":
        np.savez(filename, data=data)
    
    else:
        raise NotImplementedError()


def read_camera_poses(posepath, SynRoom=True):
    if SynRoom:
        data = load(posepath, " ")
    else:
        data = load(posepath, ",")
    pose_num = len(data)
    camera_poses = []
    positions = np.zeros((pose_num, 3))

    for i, row in enumerate(data):
        quat = row[-7:]
        cam_pose = tf.quaternion_2_matrix(quat)
        camera_poses.append(cam_pose)
        positions[i, :] = cam_pose[:3, 3]

    return camera_poses, positions
