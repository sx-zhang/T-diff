import numpy as np
import quaternion
import _pickle as cPickle
import bz2
import sys
sys.path.append("..")
import h5py
import json
import os
import math
import skimage.morphology
import cv2
from semexp.envs.utils.fmm_planner import FMMPlanner
import numpy.ma as ma
import scipy.signal as signal
import scipy.io as scio
import torch
from semexp.utils.visualize_tools import *

dataset_info_file = '..data/datasets/objectnav/gibson/v1.1/val/val_info.pbz2'
dataset_file_path = '..data/semantic_maps/gibson/semantic_maps/'
with bz2.BZ2File(dataset_info_file, "rb") as f:
    dataset_info_1 = cPickle.load(f)
with open("..data/semantic_maps/gibson/semantic_maps/semmap_GT_info.json",'r') as fp:
    dataset_info = json.load(fp)

LOCAL_MAP_SIZE = 480  # TO DO
OBJECT_BOUNDARY = 1 - 0.5
MAP_RESOLUTION = 5

def convert_3d_to_2d_pose(position, rotation):
    x = -position[2]
    y = -position[0]
    axis = quaternion.as_euler_angles(rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def gt_sem_map(current_episodes):
    # current_episodes = envs.get_current_episodes()[e]
    scene_name = current_episodes['scene_id']
    scene_name = os.path.basename(scene_name).split(".")[0]
    scene_data_file_path = dataset_file_path + scene_name + ".h5"
    goal_idx = current_episodes["object_ids"][0]
    floor_idx = 0
    scene_info = dataset_info_1[scene_name]
    shape_of_gt_map = scene_info[floor_idx]["sem_map"].shape
    f = h5py.File(scene_data_file_path, "r")
    if scene_name=="Corozal":
        sem_map=f['0/map_semantic'][()].transpose()
    else:
        sem_map=f['1/map_semantic'][()].transpose()
        
    w1, h1 = int(sem_map.shape[0]/2), int(sem_map.shape[1]/2)
    w2, h2 = int(shape_of_gt_map[1]/2), int(shape_of_gt_map[2]/2)
    sem_map1 = sem_map[w1-w2:w1+w2,h1-h2:h1+h2]
    central_pos = dataset_info[scene_name]["central_pos"]
    map_world_shift = dataset_info[scene_name]["map_world_shift"]
    map_obj_origin = scene_info[floor_idx]["origin"]
    min_x, min_y = map_obj_origin / 100.0
    pos = current_episodes["start_position"]
    rot = quaternion.from_float_array(current_episodes["start_rotation"])
    x, y, o = convert_3d_to_2d_pose(pos, rot)
    start_x, start_y = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)
    sem_map2 = map_conversion(sem_map1, start_x, start_y, o)
    goal_loc = (sem_map2 == goal_idx+5.0)
    return goal_loc, sem_map2

def map_conversion(sem_map, start_x, start_y, start_o):
    output_map = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE))
    sin = math.sin(np.pi*1 - start_o)
    cos = math.cos(np.pi*1 - start_o)
    for i in range(18): 
        loc = np.where(sem_map==i)
        if len(loc[0]) == 0:
            continue
        a = loc[0] - start_x
        b = loc[1] - start_y
        loc_conversion = (a * cos + b * sin).astype(np.int) + LOCAL_MAP_SIZE//2, (b * cos - a * sin).astype(np.int) + LOCAL_MAP_SIZE//2
        loc_conversion = void_out_of_boundary(loc_conversion)
        if len(loc_conversion[0]) == 0:
            continue
        if i == 0:
            pass
        elif i == 1:
            color_index = 2
            output_map[loc_conversion] = color_index
        elif i == 2:
            color_index = 1
            output_map[loc_conversion] = color_index
        else:
            color_index = i+2
            output_map[loc_conversion] = color_index
    output_map = signal.medfilt(output_map, 3)
    return output_map

def void_out_of_boundary(locs):
    new_locs = [[],[]]
    for i in range(locs[0].shape[0]):
        if 0<locs[0][i]<LOCAL_MAP_SIZE and 0<locs[1][i]<LOCAL_MAP_SIZE:
            new_locs[0].append(locs[0][i])
            new_locs[1].append(locs[1][i])
        else:
            continue
    return [np.array(new_locs[0]), np.array(new_locs[1])]
