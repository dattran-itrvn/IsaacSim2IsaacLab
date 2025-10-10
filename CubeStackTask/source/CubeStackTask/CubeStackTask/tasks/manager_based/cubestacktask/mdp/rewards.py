# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_object2hand(
    env: ManagerBasedRLEnv,
    object_name: str,
    object_cfg: SceneEntityCfg | None = None,
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    rightfinger_cfg: SceneEntityCfg = SceneEntityCfg("rightfinger_frame"),
    leftfinger_cfg: SceneEntityCfg = SceneEntityCfg("leftfinger_frame"),
) -> torch.Tensor:
    # distance from hand to the object_red
    object_cfg = SceneEntityCfg(object_name)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[end_effector_cfg.name]
    rightfinger_frame: FrameTransformer = env.scene[rightfinger_cfg.name]
    leftfinger_frame: FrameTransformer = env.scene[leftfinger_cfg.name]

    object_position_world = object.data.root_pos_w
    ee_position_world = ee_frame.data.target_pos_w[:, 0, :]
    rightfinger_position_world = rightfinger_frame.data.target_pos_w[:, 0, :]
    leftfinger_position_world = leftfinger_frame.data.target_pos_w[:, 0, :]


    object_ee_distance = torch.norm(object_position_world - ee_position_world, dim=-1)
    object_rightfinger_distance = torch.norm(object_position_world - rightfinger_position_world, dim=-1)
    object_leftfinger_distance = torch.norm(object_position_world - leftfinger_position_world, dim=-1)

    mean_distance = (object_ee_distance + object_rightfinger_distance + object_leftfinger_distance)/3.0

    return  1 - torch.tanh(10.0 * mean_distance)


def object_lifting(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_name: str,
    object_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    object_cfg = SceneEntityCfg(object_name)
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_align(
    env: ManagerBasedRLEnv,
    red_cube_size: float = 0.05 * 0.8,
    blue_cube_size: float = 0.05 * 1.2,
    object_red_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    object_blue_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
) -> torch.Tensor:
    # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
    object_red: RigidObject = env.scene[object_red_cfg.name]
    object_blue: RigidObject = env.scene[object_blue_cfg.name]

    red_pos = object_red.data.root_pos_w
    blue_pos = object_blue.data.root_pos_w
    offset = (red_cube_size + blue_cube_size) / 2.0

    distance = blue_pos - red_pos
    distance[:,2] += offset

    distance = torch.norm(distance, dim=-1)

    return (1 - torch.tanh(10.0 * distance)) * object_lifting(env, minimal_height= (blue_cube_size // 2), object_name="object_red")


def object_stacking(
    env: ManagerBasedRLEnv,
    red_cube_size: float = 0.05 * 0.8,
    green_cube_size: float = 0.05 * 1.2,
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    atol=0.0001,
    rtol=0.0001,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_red_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    object_blue_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    robot: Articulation = env.scene[robot_cfg.name]
    object_red: RigidObject = env.scene[object_red_cfg.name]
    object_blue: RigidObject = env.scene[object_blue_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    red_pos = object_red.data.root_pos_w[:, :2]
    blue_pos = object_blue.data.root_pos_w[:, :2]
    distance_error = torch.norm(blue_pos - red_pos, dim=-1)
    object_red_align_blue = (distance_error < xy_threshold)

    red_height = object_red.data.root_pos_w[:, 2]
    height_error = torch.abs(red_height - (green_cube_size + red_cube_size/2.0))
    object_red_on_blue = (height_error < height_threshold)
    # print(f"[DEBUG] red_height[0]: {red_height[0].item():.2f}, height_error[0]: {height_error[0].item():.2f}")

    # ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    # # gripper away from object red
    # ee_distance = torch.norm(ee_pos - object_red.data.root_pos_w, dim=-1)
    # gripper_away_from_object = (ee_distance > 0.1)
    # Check gripper positions
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

        gripper_dropped_object = torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[0]],
                torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                atol=atol,
                rtol=rtol,
            )
        gripper_dropped_object = torch.logical_and(
            torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[1]],
                torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                atol=atol,
                rtol=rtol,
            ),
            gripper_dropped_object,
        )
    else:
        raise ValueError("No gripper_joint_names found in environment config")
    
    return torch.where(object_red_align_blue & object_red_on_blue & gripper_dropped_object, 1.0, 0.0)

