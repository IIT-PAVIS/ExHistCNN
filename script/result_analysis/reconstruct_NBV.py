"""
The script for visualising the selected views of each NBV methods for ECCV2020
Authour: Yiming Wang
Updated on 27 April 2019

Step 1: set VIS_FINAL = True, VIS_ALL = False
Step2: the final pcd will be visualised and you will be prompted to save the view point:
       Use mouse to manipulate, once the view point is done, just close the point cloud.
       the view point will be saved automatically
Step3: rerun the script, you can set VIS_ALL = True to get visualisation at each time step
"""

import utils.filetool as ft
import utils.io as io
import utils.plot as local_plt
import utils.improc as improc
import os
import numpy as np
import time
import open3d as o3d
from os.path import expanduser
from shutil import copyfile


home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)

dataset_name = "mp3d"
vox_num_option = "o3d"


VIS_FINAL = True
VIS_ALL = True

NBV_strategy = ["CNNCombInfoGain","KnownDepthTwo", "GainLargeTwo","Random", "CNNmemoTwoStep5D", "CNNmemoTwoStep2D","CNNdepth","CNNMLPTwoStep5D", "CNNMLPTwoStep2D"]

config = io.load_yaml(os.path.join(proj_path, "config.yml"))
all_room_names = ft.grab_directory(os.path.join(proj_path, config["dataset_folder"], dataset_name))
print(all_room_names)

neighbour_file_name = config["neigh_info_file"]
neigh_direction_id_file_name = config["neigh_direction_id_file"]
motion_label_file_name = config["motion_label_file"]
enforce_generation = config["enforce_generation_neighbour"]
intrinsic_file_name = config["intrinsic_file"]
visibility_folder = config["visibility_folder"]
param_folder = config["param_folder"]
temp_folder = config["temp_folder"] # all info are only saved at run time and will be deleted after use

selected_room_names = all_room_names
radius_list = [0.1] # unit is in meter

label_list = ["up", "down", "left", "right"]
intrinsic_file = os.path.join(proj_path, param_folder, intrinsic_file_name)
intrinsic = o3d.read_pinhole_camera_intrinsic(intrinsic_file)
print(intrinsic)

## empiracally tried,  but the real kinect intrinsics works better
K = np.eye(3)
K[0,0] = 585 #595  #590 #573 #690
K[1,1] = 585 #595  #590 #573 #647
K[0,2] = 320
K[1,2] = 240
intrinsic.intrinsic_matrix = K

DELETE_TMP_PCD = True


selected_room_names = ["room006"] # here to indicate which room to visualise

start_key_list = ["250"] # the start indices being experimented are: [45, 168, 250, 313, 467]


# function to save the visualiser's view point
# (it is not related to the camera pose of the dataset, only for visualisation)
# the new view point file will be saved into filename.
def save_view_point(vis, pcd, filename):
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()

    trajectory = o3d.PinholeCameraTrajectory()
    trajectory.intrinsic = param[0]
    trajectory.extrinsic = o3d.Matrix4dVector([param[1]])
    o3d.write_pinhole_camera_trajectory(filename, trajectory)
    vis.destroy_window()


# main function to reconstruct the 3D scene using the selected NBV from the experiment
# @Input: the scene, the complete set of the camera poses
# all images and screenshots will be saved to /result
def reconstruct(room_name, cam_poses):
    # prepare the result file
    result_file = os.path.join(proj_path, "result", "{}_{}_{}.json".format(dataset_name, room_name, vox_num_option))

    print(result_file)
    if ft.file_exist(result_file):
        # load data
        print("load result json file ...")
        data_result = io.load_json(result_file)
    else:
        print("No result file, please check your folder again!")
        return
    if VIS_ALL or VIS_FINAL:
        vis = o3d.Visualizer()
        vis.create_window()
        print("created a window")

    for start_index_key in start_key_list:
        print("start index", start_index_key)
        result_pcd_folder = os.path.join(proj_path, "result", "pcd", start_index_key)
        if not os.path.exists(result_pcd_folder):
            os.makedirs(result_pcd_folder)
        # regarding the view point for the visualiser
        viewpoint_file = os.path.join(result_pcd_folder, "view_point.json")

        for method in NBV_strategy:
            print("method", method)
            result_pcd_file = os.path.join(result_pcd_folder, "{}_{}_{}.ply".format(dataset_name, room_name, method))
            result_pcd_folder_method = os.path.join(proj_path, "result", start_index_key,method)

            if (not os.path.exists(result_pcd_file)) or (VIS_ALL and (not os.path.exists(result_pcd_folder_method))):
                if VIS_ALL:
                    trajectory = o3d.read_pinhole_camera_trajectory(viewpoint_file)
                    print("created a window")
                if VIS_ALL and (not os.path.exists(result_pcd_folder_method)):
                    os.makedirs(os.path.join(result_pcd_folder_method, "pcd"))
                    os.makedirs(os.path.join(result_pcd_folder_method, "depth"))
                    os.makedirs(os.path.join(result_pcd_folder_method, "rgb"))
                    print("created folders")
                nbv_ids = data_result[start_index_key][method]["NBV_ids"]
                volume = o3d.ScalableTSDFVolume(voxel_length = 0.01, sdf_trunc = 0.03, color_type = o3d.TSDFVolumeColorType.RGB8) # 4.0 / 512.0
                count = 0

                for pose_id in range(len(nbv_ids)):
                    print("Integrate frame at {:d}".format(pose_id))
                    current_pose = cam_poses[nbv_ids[pose_id]]
                    rgb_path = os.path.join(proj_path, config["dataset_folder"], dataset_name, room_name, "image","rgb{:d}.png".format(nbv_ids[pose_id]))
                    depth_path = os.path.join(proj_path, config["dataset_folder"], dataset_name, room_name, "depth","depth{:d}.png".format(nbv_ids[pose_id]))
                    color = o3d.read_image(rgb_path)
                    depth = o3d.read_image(depth_path)
                    rgbd = o3d.create_rgbd_image_from_color_and_depth(color, depth, depth_trunc = 4.5, convert_rgb_to_intensity = False)
                    volume.integrate(rgbd, intrinsic, np.linalg.inv(current_pose))
                    pcd_scene = volume.extract_point_cloud()  # note the points are not isometric
                    if VIS_ALL:
                        print("Visualise each frame")
                        result_pcd_file_temp = os.path.join(result_pcd_folder_method, "pcd", "{}_{}_{:03d}.png".format(dataset_name, room_name, count))
                        result_depth_file_temp = os.path.join(result_pcd_folder_method, "depth", "{}_{}_{:03d}.png".format(dataset_name, room_name, count))
                        result_rgb_file_temp = os.path.join(result_pcd_folder_method, "rgb","{}_{}_{:03d}.png".format(dataset_name, room_name, count))

                        copyfile(depth_path, result_depth_file_temp)
                        copyfile(rgb_path, result_rgb_file_temp)

                        vis.add_geometry(pcd_scene)
                        mesh_frame = local_plt.o3d_coordinate(current_pose)
                        vis.add_geometry(mesh_frame)

                        ctr = vis.get_view_control()
                        #if count == 0:
                        ctr.convert_from_pinhole_camera_parameters(trajectory.intrinsic, trajectory.extrinsic[0])
                        vis.update_geometry()
                        vis.poll_events()
                        vis.update_renderer()

                        vis.capture_screen_image(result_pcd_file_temp)
                        # crop border
                        img = improc.crop_border(result_pcd_file_temp, 0.08, 0.4)
                        # img = improc.imscale(img, 1.5)
                        improc.imwrite(result_pcd_file_temp, img)

                        time.sleep(0.5)

                        pcd_scene.points = o3d.Vector3dVector([])
                        pcd_scene.colors = o3d.Vector3dVector([])

                        vis.update_geometry()
                        vis.poll_events()
                        vis.update_renderer()

                    count = count + 1

                if not os.path.exists(result_pcd_file):
                    pcd_scene = volume.extract_point_cloud() #note the points are not isometric
                    o3d.write_point_cloud(result_pcd_file, pcd_scene)
            else:
                if VIS_FINAL:

                    print("created a visualiser")
                    pcd_scene = o3d.read_point_cloud(result_pcd_file)
                    # only trigger the view point set when there is no tarjectory file
                    if not os.path.exists(viewpoint_file):
                        print("Please manipulate the visualiser for a better view point, close to save!")
                        save_view_point(vis, pcd_scene, viewpoint_file)
                        return

                    image_name = os.path.join(result_pcd_folder, "{}_pcd.png".format(method))

                    trajectory = o3d.read_pinhole_camera_trajectory(viewpoint_file)
                    print("read the view point")

                    vis.add_geometry(pcd_scene)
                    print("Added the view point")

                    ctr = vis.get_view_control()
                    print("obtained view control")
                    ctr.convert_from_pinhole_camera_parameters(trajectory.intrinsic, trajectory.extrinsic[0])

                    print("changed the view point")

                    vis.update_geometry()
                    vis.poll_events()
                    vis.update_renderer()
                    print("updated the rendering")
                    vis.capture_screen_image(image_name)
                    print("captured the screenshot")
                    # crop border
                    img = improc.crop_border(image_name, 0.08, 0.4)
                    #img = improc.imscale(img, 1.5)
                    improc.imwrite(image_name, img)

                    time.sleep(2)

                    pcd_scene.points = o3d.Vector3dVector([])
                    pcd_scene.colors = o3d.Vector3dVector([])

                    vis.update_geometry()
                    vis.poll_events()
                    vis.update_renderer()
                    print("reset the point to empty")
    if VIS_ALL or VIS_FINAL:
        vis.destroy_window()
        print("closed window and visualiser")


def generate_reconstruction(room_name):
    ## prepare data
    # read the camera poses 
    pose_file = os.path.join(proj_path, config["dataset_folder"], config["pose_file"])
    cam_poses, positions = io.read_camera_poses(pose_file)
    reconstruct(room_name, cam_poses)


start_t = time.time()
for name in selected_room_names:
    generate_reconstruction(name)
    
end_t = time.time()
elapsed_t = (end_t-start_t)/float(3600)

print("Program finished cleanly with elapsed time %.2f " % (elapsed_t))