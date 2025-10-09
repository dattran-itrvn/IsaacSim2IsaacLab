import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


from . import mdp

# Import robotic asset
from .franka import FRANKA_CFG

# Nucleus server paths
import carb
NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
"""Path to the root directory on the Nucleus Server."""
NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""
ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

# Import frame marker for visualization
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

##
# Scene definition
##


@configclass
class FrankastackSceneCfg(InteractiveSceneCfg):

    def __post_init__(self):
        super().__post_init__()
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_instanceable/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_instanceable/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        self.tool_rightfinger_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_instanceable/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_instanceable/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        self.tool_leftfinger_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_instanceable/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_instanceable/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        
        self.red_cube_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_instanceable/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Red_cube/Cube",
                    name="object",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robot
    robot = FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # object
    green_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Green_cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, -0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
    
    red_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Red_cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["panda_joint.*"],
        scale=0.5, 
        use_default_offset=True, 
        debug_vis=True
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", 
        joint_names=["panda_finger_joint.*"],
        debug_vis=True,
        open_command_expr= {"panda_finger_joint.*": 0.0},
        close_command_expr= {"panda_finger_joint.*": 0.04},
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        object_pose_in_robot_root_frame = ObsTerm(
            func=mdp.object_pose_in_robot_root_frame,
            params={"object_name": "red_cube"},
        )
        red_cube_green_cube_distance = ObsTerm(
            func=mdp.objectA_objectB_distance,
            params={"objectA_name": "red_cube", "objectB_name": "green_cube"},
        )
        end_effector_pose_in_robot_root_frame = ObsTerm(
            func=mdp.end_effector_pose_in_robot_root_frame,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_green_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (-0.1, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("green_cube", body_names="Cube"),
        },
    )

    reset_red_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.0), "y": (0.0, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("red_cube", body_names="Cube"),
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    object_distance_total = RewTerm(
        func=mdp.object_distance_total,
        params={"object_name": "red_cube"},
        weight=0.1,
    )

    object_is_lifted = RewTerm(
        func=mdp.object_is_lifted,
        params={"object_name": "red_cube", "minimal_height": 0.025},
        weight=1.5,
    )

    object_align = RewTerm(
        func=mdp.object_align,
        weight=2.0,
    )

    stack_finish = RewTerm(func=mdp.stack_finish, weight=1600.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-1)
 
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    red_cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("red_cube")}
    )

    green_cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("green_cube")}
    )

    # stack_finish = DoneTerm(func=mdp.stack_finish)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.2, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.2, "num_steps": 10000}
    )

##
# Environment configuration
##

@configclass
class FrankastackEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: FrankastackSceneCfg = FrankastackSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

@configclass
class FrankastackEnvCfg_PLAY(FrankastackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False