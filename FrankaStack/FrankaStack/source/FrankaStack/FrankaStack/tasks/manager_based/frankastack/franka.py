from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_ultils
from isaaclab.actuators import ImplicitActuatorCfg
from math import radians

FRANKA_CFG = ArticulationCfg(
    spawn=sim_ultils.UsdFileCfg(
        usd_path=f"/home/sim2/Khang/FrankaStack/franka_panda.usd",
        activate_contact_sensors=False,
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
            "panda_joint1": 0.0,
            "panda_joint2": radians(-30.0),
            "panda_joint3": 0.0,
            "panda_joint4": radians(-160.0),
            "panda_joint5": 0.0,
            "panda_joint6": radians(180.0),
            "panda_joint7": radians(40.0),
            "panda_finger_joint.*": 0.04,
        }
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)