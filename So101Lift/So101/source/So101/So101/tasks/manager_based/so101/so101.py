from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_ultils
from isaaclab.actuators import ImplicitActuatorCfg
from math import radians

SO101_CFG = ArticulationCfg(
    spawn=sim_ultils.UsdFileCfg(
        usd_path=f"/home/sim2/Khang/so101_new_calib/so101_new_calib.usd",
        rigid_props=sim_ultils.RigidBodyPropertiesCfg(
            rigid_body_enabled= True,
            disable_gravity= False,
            max_depenetration_velocity= 5.0,
        ),
        articulation_props=sim_ultils.ArticulationRootPropertiesCfg(
            enabled_self_collisions= True,
            solver_position_iteration_count= 8,
            solver_velocity_iteration_count= 0,
            sleep_threshold= 0.00005,
            stabilization_threshold= 0.00001
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos = {
            "shoulder_pan": 0.035,
            "shoulder_lift": -0.35,
            "elbow_flex": 0.17,
            "wrist_flex": 1.65,
            "wrist_roll": radians(-90.0),
            "gripper": 0.78,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim= 5.0,
            velocity_limit_sim = 1.5,
            stiffness={
                "shoulder_pan": 50.0,
                "shoulder_lift": 100.0,
                "elbow_flex": 10.0,
                "wrist_flex": 3.0,
                "wrist_roll": 3.0,
            },
            damping={
                "shoulder_pan": 5.0,
                "shoulder_lift": 15.0,
                "elbow_flex": 1.0,
                "wrist_flex": 0.3,
                "wrist_roll": 0.3,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim =10.0,
            velocity_limit_sim=5.0,
            stiffness=5.0,
            damping=0.3,
        )
    }
)