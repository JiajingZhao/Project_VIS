# %%
"""Continuous Control on NavMesh"""
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
        data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb")
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
            "position": [0.0, 1, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, 1, 0.0],
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

# %%

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
"""————————————————locobot————————————————"""
# load the lobot_merged asset
#locobot_template_id = obj_templates_mgr.load_configs(str(os.path.join(data_path, "objects/locobot_merged")))[0]

locobot_template = prim_templates_mgr.get_default_cylinder_template(is_wireframe = False)
locobot_template.half_length = 5
locobot_template_id = locobot_template.ID
locobot_template_handle = locobot_template.handle
# add robot object to the scene with the agent/camera SceneNode attached
id_5 = sim.add_object(locobot_template_id, sim.agents[0].scene_node)
initial_rotation = sim.get_rotation(id_5)

# set the agent's body to kinematic since we will be updating position manually
sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, id_5)

# create and configure a new VelocityControl structure
# Note: this is NOT the object's VelocityControl, so it will not be consumed automatically in sim.step_physics
vel_control = habitat_sim.physics.VelocityControl()
vel_control.controlling_lin_vel = True
vel_control.lin_vel_is_local = True
vel_control.controlling_ang_vel = True
vel_control.ang_vel_is_local = True
vel_control.linear_velocity = np.array([0, 0, -1.0])

# try 2 variations of the control experiment
for iteration in range(2):
    # reset observations and robot state
    observations = []
    sim.set_translation(np.array([1.75, -1.02, 0.4]), id_5)
    sim.set_rotation(initial_rotation, id_5)
    vel_control.angular_velocity = np.array([0.0, 0, 0])

    video_prefix = "robot_control_sliding"
    # turn sliding off for the 2nd pass
    if iteration == 1:
        sim.config.sim_cfg.allow_sliding = False
        video_prefix = "robot_control_no_sliding"

    # manually control the object's kinematic state via velocity integration
    start_time = sim.get_world_time()
    last_velocity_set = 0
    dt = 8.0
    time_step = 1.0 / 60.0
    while sim.get_world_time() < start_time + dt:
        previous_rigid_state = sim.get_rigid_state(id_5)

        # manually integrate the rigid state
        target_rigid_state = vel_control.integrate_transform(
            time_step, previous_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        end_pos = sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )
        sim.set_translation(end_pos, id_5)
        sim.set_rotation(target_rigid_state.rotation, id_5)

        # Check if a collision occured
        dist_moved_before_filter = (target_rigid_state.translation - previous_rigid_state.translation).dot()
        dist_moved_after_filter = (end_pos - previous_rigid_state.translation).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

        # run any dynamics simulation
        sim.step_physics(time_step)

        # render observation
        observations.append(sim.get_sensor_observations())

        # randomize angular velocity
        last_velocity_set += time_step
        if last_velocity_set >= 1.0:
            vel_control.angular_velocity = np.array(
                [0, (random.random() - 0.5) * 2.0, 0]
            )
            last_velocity_set = 0

    # video rendering with embedded 1st person views
    if make_video:
        sensor_dims = (
            sim.get_agent(0).agent_config.sensor_specifications[0].resolution
        )
        overlay_dims = (int(sensor_dims[1] / 4), int(sensor_dims[0] / 4))
        overlay_settings = [
            {
                "obs": "rgba_camera_1stperson",
                "type": "color",
                "dims": overlay_dims,
                "pos": (10, 10),
                "border": 2,
            },
            {
                "obs": "depth_camera_1stperson",
                "type": "depth",
                "dims": overlay_dims,
                "pos": (10, 30 + overlay_dims[1]),
                "border": 2,
            },
        ]

        vut.make_video(
            observations=observations,
            primary_obs="rgba_camera_3rdperson",
            primary_obs_type="color",
            video_file=output_path + video_prefix,
            fps=60,
            open_vid=show_video,
            overlay_settings=overlay_settings,
            depth_clip=10.0,
        )
        vut.make_video(
            observations=observations,
            primary_obs="semantic_camera_1stperson",
            primary_obs_type="semantic",
            video_file=output_path + video_prefix + "semantic",
            fps=60,
            open_vid=show_video,
            overlay_settings=overlay_settings,
            depth_clip=10.0,
        )

# %%
