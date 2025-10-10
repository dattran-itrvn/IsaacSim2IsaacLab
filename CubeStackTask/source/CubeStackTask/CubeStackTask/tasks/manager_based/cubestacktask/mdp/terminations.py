
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_red_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    object_blue_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    red_cube_size: float = 0.05 * 0.8,
    green_cube_size: float = 0.05 * 1.2,
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[object_blue_cfg.name]
    cube_2: RigidObject = env.scene[object_red_cfg.name]

    height_diff = green_cube_size + red_cube_size/2.0
    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
  
    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)

    # Check cube positions
    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, h_dist_c12 - height_diff < height_threshold)
    stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)

    # Check gripper positions
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

        stacked = torch.logical_and(
            torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[0]],
                torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                atol=atol,
                rtol=rtol,
            ),
            stacked,
        )
        stacked = torch.logical_and(
            torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[1]],
                torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                atol=atol,
                rtol=rtol,
            ),
            stacked,
        )
    else:
        raise ValueError("No gripper_joint_names found in environment config")

    return stacked
