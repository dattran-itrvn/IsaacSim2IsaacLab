from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_ultils
from isaaclab.actuators import ImplicitActuatorCfg

UR_GRIPPER_CFG = ArticulationCfg(
    spawn= sim_ultils.UsdFileCfg(
        usd_path=f"/home/sim2/Khang/ur_gripper.usd",
        rigid_props=sim_ultils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_ultils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.00005,
            stabilization_threshold=0.00001 
        )    
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "left_inner_finger_pad_joint": 0.7853,
            "right_outer_knuckle_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_pad_joint": 0.7853,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=["left_outer_finger_joint",
            "left_inner_finger_joint",
            "left_inner_finger_pad_joint",
            "right_outer_finger_joint",
            "right_inner_finger_joint",
            "right_inner_finger_pad_joint"],
            effort_limit=50.0,
            velocity_limit_sim=10.0,
            stiffness=100,
            damping=0.001,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint", "right_outer_knuckle_joint"],
            effort_limit=1650.0,
            velocity_limit_sim=10.0,
            stiffness=170,
            damping=0.2,
        ),
    }
)