# simulation/robot_loader.py
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.prims as prim_utils

def load_robot(usd_path, prim_path="/World/CustomRobot", translation=(0, 0, 0), orientation=(0, 0, 0, 1)):
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    prim_utils.set_transform(prim_path, translation=translation, orientation=orientation)
    print(f"Robot loaded at {prim_path} with translation {translation} and orientation {orientation}.")

if __name__ == "__main__":
    # Example: load a robot located at "assets/robot.usd"
    load_robot("assets/robot.usd", translation=(1, 0, 0))
