#!/usr/bin/env python
# create_test_robot.py
"""
This script creates a simple differential drive robot USD file for testing.
It uses Isaac Sim to create a basic robot with two wheels and a cube body.
"""

import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import isaaclab
from isaaclab.app import AppLauncher

def parse_args():
    parser = argparse.ArgumentParser(description="Create a test robot USD file")
    
    # Add AppLauncher arguments first
    AppLauncher.add_app_launcher_args(parser)
    
    # Add custom arguments
    parser.add_argument("--output", type=str, default="assets/robot.usd",
                       help="Output path for the robot USD file")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Launch Isaac Sim
    logger.info(f"Launching Isaac Sim to create robot USD at {args.output}...")
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app
    
    # Import required modules (must be done after launching simulation)
    import omni.usd
    import omni.kit.commands
    from pxr import UsdGeom, Gf, Sdf, UsdPhysics
    
    # Create a new stage
    omni.kit.commands.execute("CreateNewStage")
    stage = omni.usd.get_context().get_stage()
    
    # Set up stage with physics
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
    
    # Create the robot base (root prim)
    robot_prim = UsdGeom.Xform.Define(stage, "/robot")
    
    # Create the robot body
    body_prim_path = "/robot/body"
    body = UsdGeom.Cube.Define(stage, body_prim_path)
    body.CreateSizeAttr(0.5)  # 0.5m cube
    body.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.25))  # Lift body above ground
    
    # Add physics to body
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(body_prim_path))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(body_prim_path))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(body_prim_path))
    body_collider = UsdPhysics.BoxCollisionAPI.Apply(stage.GetPrimAtPath(body_prim_path))
    body_collider.CreateSizeAttr(Gf.Vec3f(0.5, 0.5, 0.5))
    
    # Create left wheel
    left_wheel_path = "/robot/left_wheel"
    left_wheel = UsdGeom.Cylinder.Define(stage, left_wheel_path)
    left_wheel.CreateRadiusAttr(0.1)  # 0.1m radius
    left_wheel.CreateHeightAttr(0.05)  # 0.05m width
    left_wheel.AddTranslateOp().Set(Gf.Vec3f(0, 0.25, 0.1))  # Position on left side
    left_wheel.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))  # Orient for rolling
    
    # Add physics to left wheel
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(left_wheel_path))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(left_wheel_path))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(left_wheel_path))
    left_wheel_collider = UsdPhysics.CylinderCollisionAPI.Apply(stage.GetPrimAtPath(left_wheel_path))
    left_wheel_collider.CreateRadiusAttr(0.1)
    left_wheel_collider.CreateHeightAttr(0.05)
    
    # Create right wheel
    right_wheel_path = "/robot/right_wheel"
    right_wheel = UsdGeom.Cylinder.Define(stage, right_wheel_path)
    right_wheel.CreateRadiusAttr(0.1)  # 0.1m radius
    right_wheel.CreateHeightAttr(0.05)  # 0.05m width
    right_wheel.AddTranslateOp().Set(Gf.Vec3f(0, -0.25, 0.1))  # Position on right side
    right_wheel.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))  # Orient for rolling
    
    # Add physics to right wheel
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(right_wheel_path))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(right_wheel_path))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(right_wheel_path))
    right_wheel_collider = UsdPhysics.CylinderCollisionAPI.Apply(stage.GetPrimAtPath(right_wheel_path))
    right_wheel_collider.CreateRadiusAttr(0.1)
    right_wheel_collider.CreateHeightAttr(0.05)
    
    # Create front caster wheel (to balance the robot)
    caster_path = "/robot/caster"
    caster = UsdGeom.Sphere.Define(stage, caster_path)
    caster.CreateRadiusAttr(0.05)  # 0.05m radius
    caster.AddTranslateOp().Set(Gf.Vec3f(0.2, 0, 0.05))  # Position at front
    
    # Add physics to caster
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(caster_path))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(caster_path))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(caster_path))
    caster_collider = UsdPhysics.SphereCollisionAPI.Apply(stage.GetPrimAtPath(caster_path))
    caster_collider.CreateRadiusAttr(0.05)
    
    # Create joints
    # Left wheel joint
    left_joint_path = "/robot/left_wheel_joint"
    left_joint = UsdPhysics.RevoluteJoint.Define(stage, left_joint_path)
    left_joint.CreateBody0Rel().SetTargets([body_prim_path])
    left_joint.CreateBody1Rel().SetTargets([left_wheel_path])
    left_joint.CreateAxisAttr().Set(Gf.Vec3f(0, 1, 0))  # Rotate around Y axis
    left_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0.25, 0.1))  # Position relative to body
    left_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))  # Position relative to wheel
    
    # Right wheel joint
    right_joint_path = "/robot/right_wheel_joint"
    right_joint = UsdPhysics.RevoluteJoint.Define(stage, right_joint_path)
    right_joint.CreateBody0Rel().SetTargets([body_prim_path])
    right_joint.CreateBody1Rel().SetTargets([right_wheel_path])
    right_joint.CreateAxisAttr().Set(Gf.Vec3f(0, 1, 0))  # Rotate around Y axis
    right_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, -0.25, 0.1))  # Position relative to body
    right_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))  # Position relative to wheel
    
    # Caster joint (fixed)
    caster_joint_path = "/robot/caster_joint"
    caster_joint = UsdPhysics.FixedJoint.Define(stage, caster_joint_path)
    caster_joint.CreateBody0Rel().SetTargets([body_prim_path])
    caster_joint.CreateBody1Rel().SetTargets([caster_path])
    caster_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.2, 0, 0.05))  # Position relative to body
    caster_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))  # Position relative to caster
    
    # Define default prim
    stage.SetDefaultPrim(stage.GetPrimAtPath("/robot"))
    
    # Save the stage
    logger.info(f"Saving USD file to {args.output}...")
    omni.usd.get_context().save_as_stage(args.output, None)
    
    # Give time for save to complete
    import time
    time.sleep(2)
    
    # Close the simulation
    logger.info("Robot USD created successfully!")
    sim_app.close()

if __name__ == "__main__":
    main() 