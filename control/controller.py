
# control/controller.py
import omni.isaac.dynamic_control as dc

def get_robot_articulation(robot_path="/World/CustomRobot"):
    dc_interface = dc._dynamic_control.acquire_dynamic_control_interface()
    robot_art = dc_interface.get_articulation(robot_path)
    return dc_interface, robot_art

def command_differential_drive(dc_interface, robot_art, v_lin, v_ang, wheel_params):
    # wheel_params is a dict with keys: left_joint_name, right_joint_name, axle_length, wheel_radius
    left_joint = dc_interface.find_articulation_dof(robot_art, wheel_params["left_joint_name"])
    right_joint = dc_interface.find_articulation_dof(robot_art, wheel_params["right_joint_name"])
    
    L = wheel_params["axle_length"]
    R = wheel_params["wheel_radius"]
    # Convert linear and angular velocity to wheel speeds (rad/s)
    v_left  = (v_lin - (v_ang * L) / 2.0) / R
    v_right = (v_lin + (v_ang * L) / 2.0) / R
    
    dc_interface.set_dof_velocity_target(left_joint, float(v_left))
    dc_interface.set_dof_velocity_target(right_joint, float(v_right))
    print(f"Set left wheel to {v_left} rad/s and right wheel to {v_right} rad/s.")

if __name__ == "__main__":
    # For testing, assume differential drive with example parameters.
    wheel_params = {
        "left_joint_name": "left_wheel_joint",
        "right_joint_name": "right_wheel_joint",
        "axle_length": 0.5,
        "wheel_radius": 0.1
    }
    dc_interface, robot_art = get_robot_articulation()
    # Command forward motion at 1.0 m/s and 0.0 rad/s turn rate.
    command_differential_drive(dc_interface, robot_art, 1.0, 0.0, wheel_params)
