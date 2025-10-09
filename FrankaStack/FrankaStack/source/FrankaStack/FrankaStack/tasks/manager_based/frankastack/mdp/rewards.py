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


def object_distance_total(
    env: ManagerBasedRLEnv,
    object_name: str,
    object_cfg: SceneEntityCfg | None = None,
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    tool_rightfinger_cfg: SceneEntityCfg = SceneEntityCfg("tool_rightfinger_frame"),
    tool_leftfinger_cfg: SceneEntityCfg = SceneEntityCfg("tool_leftfinger_frame"),
) -> torch.Tensor:
    
    object_cfg = SceneEntityCfg(object_name)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[end_effector_cfg.name]
    tool_rightfinger_frame: FrameTransformer = env.scene[tool_rightfinger_cfg.name]
    tool_leftfinger_frame: FrameTransformer = env.scene[tool_leftfinger_cfg.name]

    object_position_world = object.data.root_pos_w
    ee_position_world = ee_frame.data.target_pos_w[:, 0, :]
    tool_rightfinger_position_world = tool_rightfinger_frame.data.target_pos_w[:, 0, :]
    tool_leftfinger_position_world = tool_leftfinger_frame.data.target_pos_w[:, 0, :]


    object_ee_distance = torch.norm(object_position_world - ee_position_world, dim=-1)
    object_tool_rightfinger_distance = torch.norm(object_position_world - tool_rightfinger_position_world, dim=-1)
    object_tool_leftfinger_distance = torch.norm(object_position_world - tool_leftfinger_position_world, dim=-1)

    mean_distance = (object_ee_distance + object_tool_rightfinger_distance + object_tool_leftfinger_distance)/3.0

    return torch.maximum(object_align(env), 1 - torch.tanh(10.0 * mean_distance)) * (1 - stack_finish(env))


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_name: str,
    object_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    object_cfg = SceneEntityCfg(object_name)
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0) * (1 - stack_finish(env))

def object_align(
    env: ManagerBasedRLEnv,
    red_cube_size: float = 0.0466*0.8,
    green_cube_size: float = 0.0466,
    red_cube_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    green_cube_cfg: SceneEntityCfg = SceneEntityCfg("green_cube"),
) -> torch.Tensor:
    red_cube: RigidObject = env.scene[red_cube_cfg.name]
    green_cube: RigidObject = env.scene[green_cube_cfg.name]

    red_pos = red_cube.data.root_pos_w
    green_pos = green_cube.data.root_pos_w
    offset = (red_cube_size + green_cube_size) / 2.0

    distance = green_pos - red_pos
    distance[:,2] += offset

    distance = torch.norm(distance, dim=-1)

    return (1 - torch.tanh(10.0 * distance)) * object_is_lifted(env, minimal_height=0.025, object_name="red_cube") * (1 - stack_finish(env))

def stack_finish(
    env: ManagerBasedRLEnv,
    red_cube_size: float = 0.0466*0.8,
    green_cube_size: float = 0.0466,
    red_cube_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    green_cube_cfg: SceneEntityCfg = SceneEntityCfg("green_cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    red_cube: RigidObject = env.scene[red_cube_cfg.name]
    green_cube: RigidObject = env.scene[green_cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    red_pos = red_cube.data.root_pos_w[:, :2]
    green_pos = green_cube.data.root_pos_w[:, :2]
    distance_error = torch.norm(green_pos - red_pos, dim=-1)

    red_height = red_cube.data.root_pos_w[:, 2]
    height_error = torch.abs(red_height - (green_cube_size + red_cube_size/2.0))

    # print(f"[DEBUG] red_height[0]: {red_height[0].item():.2f}, height_error[0]: {height_error[0].item():.2f}")

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    ee_distance = torch.norm(ee_pos - red_cube.data.root_pos_w, dim=-1)

    return torch.where((distance_error < 0.005) & (height_error < 0.005) & (ee_distance > 0.1), 1.0, 0.0)
