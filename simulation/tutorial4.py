# %%
"""Embodied Agents"""
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
        data_path, "scene_datasets/habitat-test-scenes/ZMojNkEp431.glb")
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

# %%
# load the lobot_merged asset
#locobot_template_id = obj_templates_mgr.load_configs(str(os.path.join(data_path, "objects/locobot_merged")))[0]
remove_all_objects(sim)
locobot_template = prim_templates_mgr.get_default_cylinder_template(is_wireframe = False)
locobot_template_id = locobot_template.ID
locobot_template_handle = locobot_template.handle
# add robot object to the scene with the agent/camera SceneNode attached
id_1 = sim.add_object(locobot_template_id, sim.agents[0].scene_node)
sim.set_translation(np.array([1.75, -1.02, 0.4]), id_1)

vel_control = sim.get_object_velocity_control(id_1)
vel_control.linear_velocity = np.array([0, 0, -1.0])
vel_control.angular_velocity = np.array([0.0, 2.0, 0])

# simulate robot dropping into place
observations = simulate(sim, dt=1.5, get_frames=make_video)

vel_control.controlling_lin_vel = True
vel_control.controlling_ang_vel = True
vel_control.lin_vel_is_local = True
vel_control.ang_vel_is_local = True

# simulate forward and turn
observations += simulate(sim, dt=1.0, get_frames=make_video)

vel_control.controlling_lin_vel = False
vel_control.angular_velocity = np.array([0.0, 1.0, 0])

# simulate turn only
observations += simulate(sim, dt=1.5, get_frames=make_video)

vel_control.angular_velocity = np.array([0.0, 0.0, 0])
vel_control.controlling_lin_vel = True
vel_control.controlling_ang_vel = True

# simulate forward only with damped angular velocity (reset angular velocity to 0 after each step)
observations += simulate(sim, dt=1.0, get_frames=make_video)

vel_control.angular_velocity = np.array([0.0, -1.25, 0])

# simulate forward and turn
observations += simulate(sim, dt=2.0, get_frames=make_video)

vel_control.controlling_ang_vel = False
vel_control.controlling_lin_vel = False

# simulate settling
observations += simulate(sim, dt=3.0, get_frames=make_video)

# remove the agent's body while preserving the SceneNode
sim.remove_object(id_1, False)

# video rendering with embedded 1st person view
if make_video:
    vut.make_video(
        observations,
        "rgba_camera_1stperson",
        "color",
        output_path + "4_robot_control_grass",
        open_vid=show_video
    )

# %%