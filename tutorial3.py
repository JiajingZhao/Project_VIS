# %%
"""Kinematic Velocity Control"""
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
        data_path, "scene_datasets/habitat-test-scenes/van-gogh-room.glb")
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
        "semantic_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
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

remove_all_objects(sim)
observations = []
capsule_template = prim_templates_mgr.get_default_capsule_template(is_wireframe = False)
capsule_template_handle = capsule_template.handle
id_1 = sim.add_object_by_handle(capsule_template_handle)
sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, id_1)
sim.set_translation(np.array([0.8, 0, 0.5]), id_1)

start_time = sim.get_world_time()
dt = 2
while sim.get_world_time() < start_time + dt:
    # manually control the object's kinematic state
    sim.set_translation(sim.get_translation(id_1) + np.array([0, 0, 0.01]), id_1)
    sim.set_rotation(
        mn.Quaternion.rotation(mn.Rad(0.025), np.array([0, 0, -1.0]))
        * sim.get_rotation(id_1),
        id_1,
    )
    sim.step_physics(1.0 / 60.0)
    observations.append(sim.get_sensor_observations())

if make_video:
    vut.make_video(
        observations,
        "semantic_camera_1stperson",
        "semantic",
        output_path + "3_kinematic_update_z",
        open_vid=show_video,
    )
# %%
# get object VelocityControl structure and setup control
vel_control = sim.get_object_velocity_control(id_1)
vel_control.linear_velocity = np.array([0, 0, -1.0])
vel_control.angular_velocity = np.array([4.0, 0, 0])
vel_control.controlling_lin_vel = True
vel_control.controlling_ang_vel = True

observations = simulate(sim, dt=2.0, get_frames=True)

# reverse linear direction
vel_control.linear_velocity = np.array([0, 0, 1.0])

observations += simulate(sim, dt=1.0, get_frames=True)

if make_video:
    vut.make_video(
        observations,
        "semantic_camera_3rdperson",
        "semantic",
        output_path + "3_velocity_control",
        open_vid=show_video,
    )
# %%
vel_control.linear_velocity = np.array([0, 0, 2.0])
vel_control.angular_velocity = np.array([-4.0, 0.0, 0])
vel_control.lin_vel_is_local = True
vel_control.ang_vel_is_local = True

observations = simulate(sim, dt=1.5, get_frames=True)

# video rendering
if make_video:
    vut.make_video(
        observations,
        "rgba_camera_1stperson",
        "color",
        output_path + "3_local_velocity_control",
        open_vid=show_video,
    )

# %%
