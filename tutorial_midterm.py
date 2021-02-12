# %%
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
observations = []

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


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = os.path.join(
        data_path, "scene_datasets/habitat-test-scenes/ZMojNkEp431.glb"
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

make_video = True
show_video = False
# %%

"""——————————————————————Translation——————————————————————"""
sphere_template= prim_templates_mgr.get_default_UVsphere_template(is_wireframe = False)
# add an object to the scene
id_1 = sim.add_object(sphere_template.ID)
sim.set_translation(np.array([4.50, 1, 0.2]), id_1)

# %%
"""——————————————————————Forces and torques——————————————————————"""
#add sphere template
sphere_template= prim_templates_mgr.get_default_UVsphere_template(is_wireframe = False)
sphere_template_handle = sphere_template.handle

#add 5 cubes 
cheezit_template = prim_templates_mgr.get_default_cube_template(is_wireframe = False)
cheezit_template_handle = cheezit_template.handle
cheezit_template.half_length = 2.5

box_positions = [
    np.array([2.39, -0.37, 0]),
    np.array([2.39, -0.64, 0]),
    np.array([2.39, -0.91, 0]),
    np.array([2.39, -0.64, -0.22]),
    np.array([2.39, -0.64, 0.22]),
]
box_orientation = mn.Quaternion.rotation(
    mn.Rad(math.pi / 2.0), np.array([-1.0, 0, 0]))
print("box_orientation: ",box_orientation)
# instance and place the boxes
box_ids = []
for b in range(5):
    box_ids.append(sim.add_object_by_handle(cheezit_template_handle))
    sim.set_translation(box_positions[b], box_ids[b])
    sim.set_rotation(box_orientation, box_ids[b])
    """——————————————————————Kinematic Object Placement——————————————————————"""
    if b == 2 or b == 3:
        sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, box_ids[b])

# get the object's initialization attributes (all boxes initialized with same mass)
object_init_template = sim.get_object_initialization_template(box_ids[0])
# anti-gravity force f=m(-g)
anti_grav_force = -1.0 * sim.get_gravity() * object_init_template.mass

# throw a sphere at the boxes from the agent position
sphere_template_id = sphere_template.ID
sphere_template = obj_templates_mgr.get_template_by_ID(sphere_template_id)
sphere_template.scale = np.array([0.25, 0.25, 0.25])
obj_templates_mgr.register_template(sphere_template)

sphere_id = sim.add_object(sphere_template_id)
sim.set_translation(
    sim.agents[0].get_state().position + np.array([0, 1.0, 0]), sphere_id
)
# get the vector from the sphere to a box
target_direction = sim.get_translation(box_ids[0]) - sim.get_translation(sphere_id)
print("target_direction: ",target_direction)

# apply an initial velocity for one step
sim.set_linear_velocity(target_direction * 5, sphere_id)
sim.set_angular_velocity(np.array([0, -1.0, 0]), sphere_id)

# %%
"""——————————————————————Kinematic Velocity Control——————————————————————"""
capsule_template = prim_templates_mgr.get_default_capsule_template(is_wireframe = False)
capsule_template_handle = capsule_template.handle
id_2 = sim.add_object_by_handle(capsule_template_handle)
sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, id_2)
sim.set_translation(np.array([0.8, 0, 0.5]), id_2)



"""——————————————————————Embodied Agents——————————————————————"""

"""——————————————————————Continuous Control on NavMesh——————————————————————"""

# %%
"""——————————————————————Simulation——————————————————————"""
start_time = sim.get_world_time()
dt = 10
while sim.get_world_time() < start_time + dt:
    #forces/torques
    for box_id in box_ids:
        sim.apply_force(anti_grav_force, np.array([0, 0.0, 0]), box_id)
        sim.apply_torque(np.array([0, 0.01, 0]), box_id)
    #Kinematic Velocity Control
    sim.set_translation(sim.get_translation(id_2) + np.array([0, 0, 0.01]), id_2)
    sim.set_rotation(
        mn.Quaternion.rotation(mn.Rad(0.0125), np.array([0, 0, -1.0]))
        * sim.get_rotation(id_2),
        id_2,
    )
    
    #
    sim.step_physics(1.0 / 60.0)
    observations.append(sim.get_sensor_observations())

if make_video:
    vut.make_video(
        observations,
        "rgba_camera_1stperson",
        "color",
        output_path + "mid-term_1_rgb",
        open_vid=show_video,
    )

# %%