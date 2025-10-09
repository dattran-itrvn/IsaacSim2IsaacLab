# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    ee_distance = torch.norm(ee_pos - red_cube.data.root_pos_w, dim=-1)

    return (distance_error < 0.005) & (height_error < 0.005) & (ee_distance > 0.1)
