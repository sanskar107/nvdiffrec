# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import cv2

import numpy as np
from tqdm import tqdm

factor = 0.5

datasets = ['antman', 'apple', 'chest', 'gamepad', 'ping_pong_racket', 'porcelain_mug', 'tpiece', 'wood_bowl']
# datasets = ["scan37", "scan40", "scan55", "scan63", "scan65", "scan69", "scan83", "scan97"]


datasets = ["/homes/sanskar/data/objrel/llff_data/" + dataset for dataset in datasets]
# datasets = ["/homes/sanskar/data/bmvsdtu/llff_data/" + dataset for dataset in datasets]
folders  = ['images', 'masks']

for dataset in datasets:
    dataset_rescaled = "/homes/sanskar/data/nvdiffrec/input/" + dataset.split('/')[-1]
    print(dataset_rescaled)
    os.makedirs(dataset_rescaled, exist_ok=True)
    for folder in folders:
        os.makedirs(os.path.join(dataset_rescaled, folder), exist_ok=True)
        files = glob.glob(os.path.join(dataset, folder, '*.png')) + glob.glob(os.path.join(dataset, folder, '*.PNG'))
        for file in tqdm(files):
            # print(file)
            img = cv2.imread(file)
            height, width = img.shape[:2]
            new_width, new_height = int(width * factor), int(height * factor)
            factor_x = new_width / width
            factor_y = new_height / height
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            if len(img.shape) == 2:
                img = img[..., None]
                img = img.repeat(1, 1, 3)
            out_file = os.path.join(dataset_rescaled, folder, os.path.basename(file))
            cv2.imwrite(out_file, img)

    poses = np.load(os.path.join(dataset, "poses_bounds.npy"))
    bds = poses[:, -2:]
    poses = poses[:, :-2].reshape([-1, 3, 5])
    poses[:, 0, 4] *= factor_x
    poses[:, 1, 4] *= factor_y
    poses[:, 2, 4] *= factor_x

    poses = poses.reshape((-1, 15))
    poses = np.concatenate([poses, bds], axis=1)
    np.save(os.path.join(dataset_rescaled, "poses_bounds.npy"), poses)

    poses = np.load(os.path.join(dataset, "val_poses_bounds.npy"))
    bds = poses[:, -2:]
    poses = poses[:, :-2].reshape([-1, 3, 5])
    poses[:, 0, 4] *= factor_x
    poses[:, 1, 4] *= factor_y
    poses[:, 2, 4] *= factor_x

    poses = poses.reshape((-1, 15))
    poses = np.concatenate([poses, bds], axis=1)
    np.save(os.path.join(dataset_rescaled, "val_poses_bounds.npy"), poses)
