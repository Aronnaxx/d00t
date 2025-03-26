Awesome. I’ll put together a full project scaffold using the latest Isaac Lab main branch with UV package management. It'll cover launching the Warehouse environment, loading your robot, integrating real-time ONNX inference, adding camera streaming for VLM input, and logging all the telemetry (trajectories, IMU, camera) for training improvements—built in a modular, clean, and maintainable way.

I’ll circle back once it’s all mapped out and ready for you to dive in.

# Integrating a Custom Robot into NVIDIA Isaac Lab (Using UV Package Manager)

**Goal:** Create a UV-managed Python project to integrate a custom robot (URDF/MJCF/USD) into NVIDIA Isaac Lab (latest main branch) and enable simulation with advanced features. We will cover installation, launching a Warehouse environment, loading a custom robot, external control, camera streaming to a vision-language model (MoonDream-V2), real-time ONNX model inference for autonomy, and comprehensive telemetry logging.

## 1. Project Setup: Installing Isaac Lab with UV

NVIDIA Isaac Lab is a GPU-accelerated, open-source robotics framework built on NVIDIA Isaac Sim ([GitHub - isaac-sim/IsaacLab: Unified framework for robot learning built on NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacLab#:~:text=Isaac%20Lab%20is%20a%20GPU,real%20transfer%20in%20robotics)). We use the **UV package manager** to set up an isolated project environment with Isaac Lab.

- **Initialize a UV Project**: In a terminal, create a new project folder and initialize it with UV. Ensure you have Python 3.10 (Isaac Lab requires 3.10) ([Installing Isaac Lab through Pip — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html#:~:text=,10)). For example: 

  ```bash
  $ uv init my_robot_project
  Initialized project `my_robot_project` at `/path/to/my_robot_project`
  $ cd my_robot_project
  ```

  This creates a `pyproject.toml`, a virtual environment (`.venv`), and basic files (like `README.md`) ([Python UV: The Ultimate Guide to the Fastest Python Package Manager | DataCamp](https://www.datacamp.com/tutorial/python-uv#:~:text=%24%20cd%20explore,toml)).

- **Add Isaac Lab Dependency**: Use UV to add the Isaac Lab package (which will pull in Isaac Sim). Isaac Lab is available on NVIDIA’s PyPI index. Run:

  ```bash
  $ uv pip install isaaclab[isaacsim,all]==2.0.2 --extra-index-url https://pypi.nvidia.com
  ```
  This will create the virtual env (if not already) and install Isaac Lab 2.0.2 and Isaac Sim. (The `isaaclab[isaacsim,all]` extra ensures Isaac Sim and all extensions are included ([Installing Isaac Lab through Pip — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html#:~:text=,will%20also%20install%20Isaac%20Sim)).)

- **Verify Installation**: After installation, you can verify Isaac Sim launches correctly. For example, run `isaacsim --help` to see available options ([Installing Isaac Lab through Pip — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html#:~:text=,isaacsim)). The first run may download extensions (can take ~10 minutes) ([Installing Isaac Lab through Pip — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html#:~:text=Attention)) and prompt you to accept the Omniverse EULA ([Installing Isaac Lab through Pip — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html#:~:text=The%20first%20run%20will%20prompt,prompted%20with%20the%20below%20message)).

- **Project Structure**: Create a modular project structure for clarity and portability. For example:

  ```text
  my_robot_project/
  ├── pyproject.toml                # UV project config with dependencies
  ├── main.py                       # Main script to run the simulation
  ├── simulation/                   # Package for simulation setup
  │   ├── environment.py            # Launch Isaac Sim and environment
  │   └── robot_loader.py           # Load & validate the custom robot
  ├── control/                      # Package for robot control logic
  │   └── controller.py             # External Python controller for the robot
  ├── perception/                   # Package for vision and AI integration
  │   ├── camera.py                 # Camera sensor setup and streaming
  │   └── model_inference.py        # ONNX model loading and inference
  └── logging/                      # Package for telemetry logging
      └── telemetry.py              # Utilities to record trajectory, IMU, etc.
  ```
  Each module is separated for clarity: **simulation** for environment/robot setup, **control** for motion commands, **perception** for vision and AI, and **logging** for data capture. This makes the project easy to maintain and extend.

## 2. Launching the Warehouse Environment

Next, we launch Isaac Lab and load a sample **Warehouse** environment to simulate a realistic scene. We’ll use Isaac Sim’s built-in Warehouse USD stage.

- **Start Isaac Sim Programmatically**: Isaac Lab (via Isaac Sim) can be launched in-headless or GUI mode from Python. In `simulation/environment.py`, initialize the simulator:

  ```python
  # simulation/environment.py
  from omni.isaac.kit import SimulationApp
  config = {"headless": False}  # set True for headless mode
  simulation_app = SimulationApp(config)  # Launch Isaac Sim
  ```

  This starts the Isaac Sim application. (Ensure this import is done **before** any other Isaac modules.)

- **Open the Warehouse Stage**: Isaac Sim provides a Warehouse scene USD. According to NVIDIA’s assets, the warehouse USD is located at:  
  `omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse.usd` ([Environment Assets — Isaac Sim 4.2.0 (OLD)](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_environments.html#:~:text=Assets%20Path%3A%20)) (the “Simple Warehouse” with a shelf). Use the USD API to open this stage:

  ```python
  import omni.usd
  # Path to the Warehouse USD (could also use ISAAC_NUCLEUS_DIR env variable as in docs)
  warehouse_usd = "omniverse://localhost/NVIDIA/Assets/Isaac/Environments/Simple_Warehouse/warehouse.usd"
  omni.usd.get_context().open_stage(warehouse_usd, None)
  ```

  This loads the warehouse scene into the simulation. You may use alternative paths if you have the USD asset locally or on Nucleus. The stage will contain the environment (walls, shelves, etc.). Optionally, you can add a ground plane if needed via Isaac Sim utils (e.g., `world.scene.add_default_ground_plane()` if using the World API).

- **Initialize Simulation World**: After loading the stage, you might integrate with Isaac Lab’s world or environment manager. For example, using the Isaac Core `World` abstraction:
  
  ```python
  from omni.isaac.core import World
  world = World(stage_units_in_meters=1.0)  # Create a simulation world
  world.initialize_simulation()             # Prepare the simulation
  ```
  
  This isn’t strictly required for basic usage, but using `World` can manage physics stepping and entities. The Warehouse environment is now ready to populate with our robot.

## 3. Loading and Validating the Custom Robot

Now load your **custom robot** model into the scene. Isaac Lab supports URDF/MJCF import or direct USD. We assume the robot is **pre-validated** (i.e., its URDF/MJCF has been converted to USD and tested for correct physics).

- **Convert URDF/MJCF to USD (if needed)**: If you only have a URDF or MJCF, use Isaac Lab’s conversion tool. Isaac Lab provides a `convert_urdf.py` script to generate a USD from a URDF ([Importing a New Asset — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html#:~:text=Using%20URDF%20Importer)). For example, to convert an ANYmal robot URDF:

  ```bash
  $ ./isaaclab.sh -p scripts/tools/convert_urdf.py \
        ~/path/to/robot.urdf \
        source/isaaclab_assets/data/Robots/MyRobot/my_robot.usd \
        --merge-joints
  ``` 
  (The `--merge-joints` flag merges fixed joints for efficiency ([Importing a New Asset — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html#:~:text=.%2Fisaaclab.sh%20,joints)).) After conversion, open the USD in Isaac Sim to verify geometry, joints, and physical properties are correct (this is the "validation step", ensuring the robot behaves as expected in simulation).

- **Load Robot USD into the Scene**: With a validated `.usd` for your robot, you can add it to the Warehouse environment. Use the Stage API to **reference** the robot USD into the world (so it appears as an instance in our scene). In `simulation/robot_loader.py`:

  ```python
  # simulation/robot_loader.py
  from omni.isaac.core.utils.stage import add_reference_to_stage
  ROBOT_USD = "/path/to/my_robot.usd"
  ROBOT_PRIM = "/World/CustomRobot"  # desired path in the scene
  add_reference_to_stage(usd_path=ROBOT_USD, prim_path=ROBOT_PRIM)
  ```

  This command brings the robot’s USD into the `/World` of the current stage as a prim named "CustomRobot" ([Adding multiple robots in a scene - Isaac Sim - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/adding-multiple-robots-in-a-scene/249603#:~:text=from%20omni,manipulators%20import%20SingleManipulator)). The robot (all its links and joints) is now part of the simulation scene at the specified location. You can adjust its initial position/orientation if needed. For example, if using the Isaac Lab `World` API, you could offset the robot prim after adding: 
  ```python
  import omni.isaac.core.utils.prims as prim_utils
  prim_utils.set_transform(ROBOT_PRIM, translation=(0,0,0), orientation=(0,0,0,1))
  ```
  to place it at the origin (or any desired pose).

- **Verification**: Ensure the robot appears correctly. At this point, if you run the simulation (e.g., stepping a few frames), the robot should spawn in the Warehouse. If the robot has motors or sensors defined, they are now part of the simulation. (If Isaac Lab provided a specific robot class for it, you could alternatively use `world.scene.add(...)` with that class, but for a custom robot, referencing the USD as above is straightforward.)

## 4. Enabling External Python Control of the Robot

With the robot in the simulation, we enable **external Python control** to move it. Isaac Lab/Sim allows programmatic control of robot joints and movement. We will retrieve the robot’s articulation and command its joints each simulation step.

- **Access the Robot Articulation**: Isaac Sim’s Dynamic Control interface lets us get a handle to the robot’s joints. For example, using the `omni.isaac.dynamic_control` extension:

  ```python
  import omni.isaac.dynamic_control as dc
  dc_interface = dc._dynamic_control.acquire_dynamic_control_interface()
  robot_art = dc_interface.get_articulation("/World/CustomRobot")
  ```

  Here, `robot_art` is a handle to the robot’s articulated body in the physics simulation. We can now query or command its DOFs (Degrees of Freedom).

- **Command Joints or Wheels**: Suppose our custom robot is a wheeled mobile base with two drive wheels named `"left_wheel_joint"` and `"right_wheel_joint"`. We can set velocity targets for these wheel joints to drive the robot:

  ```python
  # control/controller.py
  left_wheel = dc_interface.find_articulation_dof(robot_art, "left_wheel_joint")
  right_wheel = dc_interface.find_articulation_dof(robot_art, "right_wheel_joint")
  # Example command: set both wheels to drive forward at 1.0 rad/s
  dc_interface.set_dof_velocity_target(left_wheel, 1.0)
  dc_interface.set_dof_velocity_target(right_wheel, 1.0)
  ```

  This will cause the robot to move forward. For a manipulator robot, you could similarly use `set_dof_position_target` or `set_dof_effort` on each joint DOF to move arms or grippers. For example, to set a joint named "joint1" to 90 degrees: 
  ```python
  joint1 = dc_interface.find_articulation_dof(robot_art, "joint1")
  dc_interface.set_dof_position_target(joint1, 1.5708)  # 1.5708 rad ≈ 90°
  ```

- **Control Loop**: Typically, you would integrate these commands inside the simulation loop (e.g., every frame or at a fixed control rate). We will see below how to combine this with sensor input and AI model output. The key is that **external Python code can read sensor data and send commands to the robot in real time** – effectively a simulation-based control loop, akin to how you’d control a real robot via a Python API.

## 5. Camera Streaming Integration (Vision-Language Model)

To enable vision-based AI, we attach a **camera sensor** to the robot and stream its feed into a vision-language model (like *MoonDream-V2*). NVIDIA Isaac Sim supports realistic cameras using RTX rendering. We will create an RGB camera on the robot and retrieve images each tick.

- **Attach a Camera to the Robot**: In `perception/camera.py`, use the Isaac Sim `Camera` class from the `omni.isaac.sensor` module to create a camera sensor. For example, to mount a camera on the robot’s base:

  ```python
  from omni.isaac.sensor import Camera
  import numpy as np

  # Create a camera at /World/CustomRobot/CameraSensor
  camera = Camera(
      prim_path="/World/CustomRobot/CameraSensor", 
      name="robot_camera",
      resolution=(640, 480),
      position=np.array([0.0, 0.0, 1.0])    # 1.0m above the robot base
  )
  camera.initialize()  # Important: initialize the camera sensor ([Accessing camera image from python - Isaac Sim - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/accessing-camera-image-from-python/293762#:~:text=Hello%2C%20have%20you%20already%20tried,the%20documentation%20here%20if%20not)) ([Accessing camera image from python - Isaac Sim - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/accessing-camera-image-from-python/293762#:~:text=Yes%2C%20that%20worked,initialize))
  ```

  This will create a new camera prim under the robot (as a child of the `CustomRobot` prim) if none exists ([Isaac Sensor Extension [omni.isaac.sensor] — isaac_sim 4.2.0-rc.17 documentation](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.sensor/docs/index.html#module-omni.isaac.sensor.scripts.camera#:~:text=Provides%20high%20level%20functions%20to,prim%20path%20will%20be%20created)). The camera is placed at the given position relative to the robot (here 1.0m up along Z-axis). By parenting it to the robot, it will move with the robot. The resolution is set to 640×480. You can adjust parameters like focal length or use higher resolution if needed (with performance trade-offs ([Camera — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#:~:text=Rendered%20images%20are%20unique%20among,challenges%20in%20the%20rendering%20pipeline)) ([Camera — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#:~:text=the%20inherently%20large%20bandwidth%20requirements,challenges%20in%20the%20rendering%20pipeline))).

- **Retrieve Camera Frames**: Once the simulation is running and the camera is initialized, you can grab images each frame. The `Camera` object provides methods to get images as NumPy arrays. For example:

  ```python
  rgba_image = camera.get_rgba()  # Get latest RGBA frame as a numpy array ([Isaac Sensor Extension [omni.isaac.sensor] — isaac_sim 4.2.0-rc.17 documentation](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.sensor/docs/index.html#module-omni.isaac.sensor.scripts.camera#:~:text=get_rgb%28%29%20%E2%86%92%20numpy))
  rgb_image = rgba_image[..., :3]  # Drop alpha channel if not needed
  ```

  Each call fetches the latest rendered frame from the simulation. You might call this in sync with the simulation step (e.g., after stepping physics).

- **Feed to Vision-Language Model**: The retrieved `rgb_image` can now be passed to a vision-language model like MoonDream-V2. This model (assuming you have it loaded, e.g., via HuggingFace or local API) can process the image and possibly a text query to produce a response. *MoonDream-V2* is a multimodal model requiring less than 6GB GPU memory according to its project page ([This VLM can be your MultiModal AI with less than 6GB Memory!!!](https://www.youtube.com/watch?v=_BzWviKLtxg#:~:text=Memory%21%21%21%20www,4K)). For instance:

  ```python
  # Pseudocode: using MoonDream-V2 to caption the scene or detect objects
  result = moondream_v2_model.infer_image(rgb_image)
  print("VLM output:", result)
  ```
  
  The specifics will depend on the model’s API, but the key point is that our simulation camera feed is now available for AI inference in real time. This could enable, for example, a language model to describe what the robot “sees” or to follow high-level instructions based on vision.

*(Note: Ensure the simulation is stepping/rendering while capturing images. If you get blank images at first, step the simulator a few frames before using the data, as camera sensors may need a frame to populate their buffers.)*

## 6. Real-Time ONNX Model Inference for Autonomous Behavior

We integrate a pre-trained `.onnx` model to drive the robot autonomously. This model could be a learned policy or an AI that takes the camera image (and possibly other sensor data) as input and outputs movement commands. We will use **ONNX Runtime** for fast inference on the model.

- **Load the ONNX Model**: In `perception/model_inference.py`, set up the ONNX runtime session:

  ```python
  import onnxruntime as ort
  # Load the ONNX model (assumes model.onnx is in project directory)
  ort_session = ort.InferenceSession("model.onnx")
  # Get the model's expected input name and shape
  input_name = ort_session.get_inputs()[0].name
  input_shape = ort_session.get_inputs()[0].shape
  ```

  If the model was exported from Isaac Lab training, it might expect an image observation or other sensor data (some Isaac Lab RL examples export policies to ONNX ([Policy Inference in USD Environment — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/policy_inference_in_usd.html#:~:text=.%2Fisaaclab.sh%20,checkpoint%20logs%2Frsl_rl%2Fh1_rough%2FEXPERIMENT_NAME%2FPOLICY_FILE.pt))). Make sure the input format matches what we feed (e.g., an image tensor of shape `(1,3, H, W)` for a CNN policy).

- **Preprocess Input**: Before inference, preprocess the camera frame as required by the model – e.g., resize or normalize pixel values, and add batch dimension:

  ```python
  import cv2
  # Assume model expects 224x224 RGB
  frame = cv2.resize(rgb_image, (224, 224))
  frame = frame.astype('float32') / 255.0  # normalize if needed
  frame = frame.transpose(2,0,1)[None, ...]  # shape to (1,C,H,W)
  ```

  (Adjust these steps based on your model’s requirements.)

- **Run Inference Each Step**: In the main simulation loop (e.g., in `main.py`), run the model on each new image frame to get robot actions:

  ```python
  # main.py (inside simulation loop)
  obs_image = camera.get_rgba()
  rgb = obs_image[...,:3]
  # preprocess as above...
  model_input = preprocess_image(rgb)
  output = ort_session.run(None, {input_name: model_input})
  actions = output[0]  # e.g., model outputs an array of actions
  ```

  The `actions` might represent desired motion – for instance, `[v, ω]` for linear and angular velocity if it's a navigation policy, or target joint angles if it's a manipulation task. This is specific to your model. If it’s a vision-language model guiding the robot, the output could even be high-level commands that you parse. In our example, assume it outputs a driving command.

- **Apply Model Output to Robot**: Convert the model’s output to actual robot controls. For a differential drive robot, suppose the model outputs `[v_lin, v_ang]` (linear and angular velocity commands for the base). We can translate that into wheel speeds:

  ```python
  v_lin, v_ang = actions  # model inference result
  # Convert to individual wheel velocities (simple differential drive kinematics)
  L = 0.5  # axle length (m)
  R = 0.1  # wheel radius (m)
  v_left  = (v_lin - (v_ang * L)/2.0) / R
  v_right = (v_lin + (v_ang * L)/2.0) / R
  dc_interface.set_dof_velocity_target(left_wheel, float(v_left))
  dc_interface.set_dof_velocity_target(right_wheel, float(v_right))
  ```

  This applies the model’s decision to the robot by setting wheel joint targets accordingly. In the case of an arm, the model might output joint angles – you would then call `set_dof_position_target` for each relevant joint.

- **Looping**: Continue this cycle in real time: capture image -> inference -> apply action -> step simulation. Isaac Lab’s simulation can be stepped by calling, for example, `world.step(render=True)` in each loop iteration, or by allowing the simulation app to run continuously and using callbacks. Ensure your loop runs at an appropriate frequency (the simulation should ideally run faster or equal to the control frequency for real-time behavior).

By doing this, your custom robot is effectively controlled by the ONNX model’s decisions, enabling autonomous movement and interaction in the Warehouse environment.

## 7. Full Telemetry Logging for Analysis and Retraining

Finally, we implement comprehensive **telemetry logging**. This includes recording the robot’s trajectory, sensor readings, and actions, as well as saving camera images. Such data is invaluable for debugging, analysis, and retraining (much like how Disney Research’s BDX droids likely log extensive data for their developers).

- **Log Robot State (Trajectory)**: At each simulation step, record the robot’s pose and any other relevant state. Isaac Sim allows querying an articulation’s state. For example:

  ```python
  from omni.isaac.core.utils.types import ArticulationAction
  import time
  log = []  # list to accumulate log entries

  # During each loop iteration:
  current_time = simulation_app.get_time()  # or time.time() for wall-clock
  # Get robot base position and orientation:
  base_transform = dc_interface.get_articulation_root_pose(robot_art)  # returns pos (x,y,z) and orientation (quat)
  lin_vel = dc_interface.get_articulation_velocity(robot_art)         # linear velocity
  ang_vel = dc_interface.get_articulation_angular_velocity(robot_art) # angular velocity

  log_entry = {
      "time": current_time,
      "position": list(base_transform.p),   # (x, y, z)
      "orientation": list(base_transform.r),# quaternion (w, x, y, z)
      "linear_vel": list(lin_vel),
      "angular_vel": list(ang_vel),
      "command": [float(v_lin), float(v_ang)]  # commanded linear & angular velocity
  }
  log.append(log_entry)
  ```
  
  Here we gather timestamp, pose, velocities, and the last command issued. You could also log individual joint states if needed (e.g., wheel speeds or arm joint angles via `dc_interface.get_dof_state()`).

- **Log IMU Data**: If your robot has an IMU sensor (Isaac Sim has an IMU sensor extension ([Isaac Sensor Extension [omni.isaac.sensor] — isaac_sim 4.2.0-rc.17 documentation](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.sensor/docs/index.html#module-omni.isaac.sensor.scripts.camera#:~:text=,16))), you can attach it similarly to the camera and record its readings (linear acceleration, angular velocity). For simplicity, you might derive IMU-like data from physics (e.g., the change in velocity to approximate acceleration).

- **Log Camera Feed**: Storing every frame can be heavy (as noted, a 800x600 image at 60Hz is ~120MB/s ([Camera — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#:~:text=Rendered%20images%20are%20unique%20among,challenges%20in%20the%20rendering%20pipeline))). Depending on your needs, you can log at a lower frame rate or only when certain events happen. Two common approaches:
  - **Save to disk**: Use OpenCV or PIL to write image files:
    ```python
    import cv2
    cv2.imwrite(f"logs/frame_{frame_count}.png", rgb_image)
    ```
    This saves each frame as an image file. Ensure `logs/` directory exists. For a long run, consider saving every Nth frame to reduce data.
  - **Video stream**: Use a video writer:
    ```python
    if frame_count == 0:
        video_writer = cv2.VideoWriter("logs/run.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640,480))
    video_writer.write(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    ```
    and close the writer at the end. This produces a video of the run.

- **Export Logs**: After the simulation, export the collected telemetry. For example, save the `log` list to a JSON or CSV for analysis:

  ```python
  import json
  with open("logs/telemetry.json", "w") as f:
      json.dump(log, f, indent=4)
  ```
  
  This JSON would contain the timeline of robot states and commands, which you can later use for plotting or retraining machine learning models. If using CSV, you could write headings and rows via Python’s `csv` module.

By logging **trajectory, IMU, control commands, and visual data**, you create a rich dataset. This is in the spirit of data-driven robotics systems like Disney Research’s BDX, where all sensor and motion data are recorded for offline analysis. Such logs enable you to replay scenarios, improve your model (e.g., using recorded camera frames to retrain the vision model), and verify system performance over time.

---

**Next Steps:** With this scaffold in place, you can iterate on your robot and model. For example, you might integrate a gamepad or keyboard control as an override, add additional sensors (LIDAR, depth camera), or incorporate ROS2 bridges if needed. The modular structure ensures you can expand each component (simulation, control, perception, logging) independently. Happy building with Isaac Lab!

**Sources:** 

- NVIDIA Isaac Lab documentation (installation and usage) ([Installing Isaac Lab through Pip — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html#:~:text=,will%20also%20install%20Isaac%20Sim)) ([GitHub - isaac-sim/IsaacLab: Unified framework for robot learning built on NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacLab#:~:text=Isaac%20Lab%20is%20a%20GPU,real%20transfer%20in%20robotics))  
- Isaac Sim asset library (Warehouse environment) ([Environment Assets — Isaac Sim 4.2.0 (OLD)](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_environments.html#:~:text=Assets%20Path%3A%20))  
- NVIDIA Isaac Sim forum and docs (URDF import, multi-robot loading, camera usage) ([Importing a New Asset — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html#:~:text=.%2Fisaaclab.sh%20,joints)) ([Adding multiple robots in a scene - Isaac Sim - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/adding-multiple-robots-in-a-scene/249603#:~:text=from%20omni,manipulators%20import%20SingleManipulator)) ([Accessing camera image from python - Isaac Sim - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/accessing-camera-image-from-python/293762#:~:text=Hello%2C%20have%20you%20already%20tried,the%20documentation%20here%20if%20not))  
- Tech community examples for UV and ONNX integration ([Python UV: The Ultimate Guide to the Fastest Python Package Manager | DataCamp](https://www.datacamp.com/tutorial/python-uv#:~:text=Adding%20initial%20dependencies%20to%20the,project)) ([Policy Inference in USD Environment — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/policy_inference_in_usd.html#:~:text=.%2Fisaaclab.sh%20,checkpoint%20logs%2Frsl_rl%2Fh1_rough%2FEXPERIMENT_NAME%2FPOLICY_FILE.pt)) (various).