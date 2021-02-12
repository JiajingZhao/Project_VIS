# %%
"""Kinematic Object Placement"""
import math
import os
import random
import sys

import git
import magnum as mn
import numpy as np

import habitat_sim
from habitat_sim.utils import viz_utils as vut
# %%
file_path = "/home/habitat/habitat/habitat-sim/"
os.chdir(file_path)
%set_env DISPLAY=:0
data_path = os.path.join(file_path, "data")
output_path = os.path.join("/home/habitat/habitat/examples/video/")

# %%
def remove_all_objects(sim):
    for id_ in sim.get_existing_object_ids():
        sim.remove_object(id_)


def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.7, 1.0]
    agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = os.path.join(
        data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb"
    )
    assert os.path.exists(backend_cfg.scene_id)
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [544, 720]
    sensors = {
        "rgba_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "rgba_camera_3rdperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 1.0, 0.3],
            "orientation": [-45, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.SensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_specs.append(sensor_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())

    return observations
# %%
# create the simulators AND resets the simulator

cfg = make_configuration()
try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)
agent_transform = place_agent(sim)

# get the primitive assets attributes manager
prim_templates_mgr = sim.get_asset_template_manager()

# get the physics object attributes manager
obj_templates_mgr = sim.get_object_template_manager()

# show handles
handles = prim_templates_mgr.get_template_handles()
for handle in handles:
    print(handle)

make_video = True
show_video = False
get_frames = True


# %%
observations = []
iconsphere_template = prim_templates_mgr.get_default_icosphere_template(is_wireframe = False)
iconsphere_template_handle = iconsphere_template.handle
id_1 = sim.add_object_by_handle(iconsphere_template_handle)
sim.set_translation(np.array([2.4, -0.64, 0]), id_1)

# set one object to kinematic
sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, id_1)
#habitat_sim.physics.MotionType.KINEMATIC/STATIC/DYNAMIC

# drop some dynamic objects
id_2 = sim.add_object_by_handle(iconsphere_template_handle)
sim.set_translation(np.array([2.4, -0.64, 0.28]), id_2)
sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, id_2)

id_3 = sim.add_object_by_handle(iconsphere_template_handle)
sim.set_translation(np.array([2.4, -0.64, -0.28]), id_3)
sim.set_object_motion_type(habitat_sim.physics.MotionType.DYNAMIC, id_3)
id_4 = sim.add_object_by_handle(iconsphere_template_handle)
sim.set_translation(np.array([2.4, -0.3, 0]), id_4)

id_5 = sim.add_object_by_handle(iconsphere_template_handle)
sim.set_translation(np.array([2.4,-0.10,-0.28]), id_5)

# simulate
observations = simulate(sim, dt=2, get_frames=True)

if make_video:
    vut.make_video(
        observations,
        "rgba_camera_1stperson",
        "color",
        output_path + "2_kinematic_interactions_3STATIC",
        open_vid=show_video
    )

# %%

