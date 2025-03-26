# simulation/environment.py
from omni.isaac.kit import SimulationApp

def launch_simulation(headless=False):
    config = {"headless": headless}
    simulation_app = SimulationApp(config)
    return simulation_app

def load_warehouse_stage():
    import omni.usd
    # Define the USD path to the Warehouse environment.
    warehouse_usd = "omniverse://localhost/NVIDIA/Assets/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    omni.usd.get_context().open_stage(warehouse_usd, None)
    print("Warehouse environment loaded.")

def initialize_world():
    from omni.isaac.core import World
    world = World(stage_units_in_meters=1.0)
    world.initialize_simulation()
    return world

if __name__ == "__main__":
    app = launch_simulation(headless=False)
    load_warehouse_stage()
    world = initialize_world()
    print("Simulation and world initialized.")
