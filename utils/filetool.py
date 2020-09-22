import glob
import os


def grab_directory(path, fullpath = False, extra_key = ""):
    if not (extra_key == ""):
        if fullpath:
            dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and (extra_key in d)]
        else:
            dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and (extra_key in d)]
    else:
        if fullpath:
            dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        else:
            dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs


# grab file by defaul is all types, all with a specific type
def grab_files(path, fullpath = False,extra= ""):
    file_list = glob.glob(os.path.join(path,extra))
    
    if fullpath:
        file_list_final = file_list
    else:
        file_list_final = [os.path.split(d)[1] for d in file_list if path in d]
    return file_list_final


def dir_exist(path):
    return os.path.isdir(path)


def file_exist(path):
    return os.path.isfile(path)
