from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    object_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object_cfg = SceneEntityCfg(object_name)
    object: RigidObject = env.scene[object_cfg.name]
    
    robot_pose_world = robot.data.root_pose_w
    object_pose_world = object.data.root_pose_w

    robot_position_world = robot_pose_world[:, :3]
    robot_orientation_world = robot_pose_world[:, 3:7]
    object_position_world = object_pose_world[:, :3]
    object_orientation_world = object_pose_world[:, 3:7]

    object_position_robot, object_orientation_robot = subtract_frame_transforms(
        robot_position_world, robot_orientation_world,
        object_position_world, object_orientation_world
        
    )

    # print("Object orientation (env 0):", object_orientation_robot[0].cpu().numpy())
    # (7)
    return torch.cat([object_position_robot, object_orientation_robot], dim=-1)

def objectA_objectB_distance(
    env: ManagerBasedRLEnv,
    objectA_name: str,
    objectB_name: str,
    objectA_cfg: SceneEntityCfg | None = None,
    objectB_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    objectA_cfg = SceneEntityCfg(objectA_name)
    objectB_cfg = SceneEntityCfg(objectB_name)
    objectA: RigidObject = env.scene[objectA_cfg.name]
    objectB: RigidObject = env.scene[objectB_cfg.name]

    objectA_position_world = objectA.data.root_pose_w[:, :3]
    objectB_position_world = objectB.data.root_pose_w[:, :3]
    # (3)
    return objectB_position_world - objectA_position_world

def end_effector_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    robot_pose_world = robot.data.root_pose_w

    robot_position_world = robot_pose_world[:, :3]
    robot_orientation_world = robot_pose_world[:, 3:7]

    ee_frame_position_world = ee_frame.data.target_pos_w[:,0,:]
    ee_frame_orientation_world = ee_frame.data.target_quat_w[:,0,:]

    ee_frame_position_robot, ee_frame_orientation_robot = subtract_frame_transforms(
        robot_position_world, robot_orientation_world,
        ee_frame_position_world, ee_frame_orientation_world
        
    )
    # (7)
    return torch.cat([ee_frame_position_robot, ee_frame_orientation_robot], dim=-1)
    

def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
        finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
        finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
        return torch.cat((finger_joint_1, finger_joint_2), dim=1)
    else:
        raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")
    