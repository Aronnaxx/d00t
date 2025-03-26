# simulation/environment.py
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from isaaclab.app import AppLauncher

def launch_simulation(headless=False, enable_cameras=True, device="cuda:0"):
    """Launch the Isaac Sim application with the given configuration.
    
    Args:
        headless (bool): Whether to run in headless mode (no GUI)
        enable_cameras (bool): Whether to enable camera rendering
        device (str): Device to use (cpu or cuda:N)
        
    Returns:
        SimulationApp: The simulation app instance
    """
    logger.info(f"Launching simulation with headless={headless}, enable_cameras={enable_cameras}, device={device}")
    
    # Create parser for AppLauncher
    parser = argparse.ArgumentParser(description="Isaac Sim App")
    AppLauncher.add_app_launcher_args(parser)
    
    # Setup arguments
    args = parser.parse_args([])  # Empty list to avoid using command line args
    args.headless = headless
    args.enable_cameras = enable_cameras
    args.device = device
    
    # Launch the app
    app_launcher = AppLauncher(args)
    return app_launcher.app

def load_warehouse_stage():
    """Load the Warehouse USD stage from NVIDIA's asset library."""
    # Import modules after simulation is initialized
    import omni.usd
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    
    logger.info("Loading warehouse stage")
    
    # Path to the Warehouse USD using Isaac Lab's nucleus directory
    warehouse_usd = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
    try:
        omni.usd.get_context().open_stage(warehouse_usd, None)
        logger.info(f"Warehouse environment loaded from {warehouse_usd}")
    except Exception as e:
        logger.error(f"Failed to load warehouse: {e}")
        logger.info("If using a different Nucleus server or local assets, adjust the path accordingly.")

def initialize_world():
    """Initialize the simulation world with physics setup.
    
    Returns:
        World: The simulation world object
    """
    # Import modules after simulation is initialized
    from omni.isaac.core import World
    
    logger.info("Initializing simulation world")
    
    # Create a simulation world with meter units
    world = World(stage_units_in_meters=1.0)
    
    # Initialize physics simulation
    world.initialize_simulation()
    
    logger.info("Simulation world initialized with physics")
    return world

if __name__ == "__main__":
    # For testing this module independently
    import time
    
    # Launch simulation
    app = launch_simulation(headless=False, enable_cameras=True)
    
    # Load stage and initialize world (this will import omni modules)
    load_warehouse_stage()
    world = initialize_world()
    
    # Keep the app running for a few seconds to verify
    for _ in range(100):
        world.step(render=True)
        time.sleep(0.01)
    
    app.close()
    logger.info("Test complete.")
