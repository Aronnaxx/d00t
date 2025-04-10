# This is a motion controller for the duck_vla project

# If we are running in simulation mode, then we will pass arrow keys as incoming commands
# if we are running in real mode, then we will pass the actual commands as an xbox controller joystick

# For simulator mode we should reference submodules/open_duck_playground/playground/open_duck_mini_v2/mujoco_infer.py for how
# the keybinds are used and for what (e.g. arrow keys for forward, backward, left, right)

# For the real version we should reference submodules/open_duck_mini_runtime/scripts/v2_rl_walk_mujoco.py
# and pass the commands to the runtime instead of using the joystick / xbox controller