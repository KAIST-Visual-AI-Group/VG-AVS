import atexit
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, Optional, Callable, Tuple
import math

import numpy as np
import torch
from PIL import Image

import habitat_sim
from habitat_sim.agent import AgentState
from habitat_sim.utils.common import quat_from_angle_axis
from pathlib import Path

logger = logging.getLogger(__name__)


_HABITAT_SIMULATOR: Optional[habitat_sim.Simulator] = None
_HABITAT_DATASET = None  # Cache HM3D dataset

DEBUG_MODE = str(os.getenv("DEBUG_MODE", "0")) == "1"


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor, a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_habitat_controller(
    cfg, scene_path: Optional[str] = None
) -> habitat_sim.Simulator:
    """Return a lazily-initialized Habitat-Sim Simulator bound to the current local GPU.

    - Binds gpuDevice to LOCAL_RANK (override with HABITAT_GPU_DEVICE)
    - Uses headless rendering for multi-GPU scenarios
    - Registers atexit cleanup to stop the simulator when the process exits
    - Checks simulator health and recreates if connection is broken

    Args:
        headless: Use headless rendering
        width: Render width
        height: Render height
        force_new: Force create a new simulator even if one exists
    """
    global _HABITAT_SIMULATOR

    if _HABITAT_SIMULATOR is not None:
        _HABITAT_SIMULATOR.close()
        _HABITAT_SIMULATOR = None

    scene_dataset_cfg = Path(cfg.scene_dataset_config).resolve()
    # Create simulator configuration
    sim_settings = {
            "scene": str(scene_path) if scene_path is not None else 'default_scene',
            "scene_dataset": str(scene_dataset_cfg),
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": cfg.img_width,
            "height": cfg.img_height,
            "hfov": cfg.hfov,
            "color_sensor": cfg.rgb_sensor,
            "depth_sensor": cfg.depth_sensor,
            "semantic_sensor": cfg.semantic_sensor,
        }

    sim_cfg = make_simple_cfg(sim_settings)
    simulator = habitat_sim.Simulator(sim_cfg)
    pathfinder = simulator.pathfinder
    pathfinder.seed(cfg.seed)

    if scene_path is not None:
        navmesh_path = Path(str(scene_path).replace(".glb", ".navmesh"))
        pathfinder.load_nav_mesh(str(navmesh_path))

    # Create simulator
    _HABITAT_SIMULATOR = simulator
    
    def _cleanup():
        try:
            if _HABITAT_SIMULATOR is not None:
                _HABITAT_SIMULATOR.close()
        except Exception:
            pass

    atexit.register(_cleanup)
    #if DEBUG_MODE:
        #print(
        #    f"[habitat] Simulator initialized (gpuDevice={gpu_device}, src={source}, LOCAL_RANK={rank_info.get('local_rank')}, headless={headless})"
        #)
    return _HABITAT_SIMULATOR


def quaternion_to_euler_degrees(quat: Any) -> Dict[str, float]:
    """Convert quaternion (w, x, y, z) to Euler angles (x, y, z) in degrees.

    Accepts:
      - a sequence [w, x, y, z]
      - a dict {"w": w, "x": x, "y": y, "z": z}
      - an object with attributes .w, .x, .y, .z (e.g., Habitat/Magnum quaternion)

    Returns a dict with keys "x", "y", "z" representing rotations (degrees).
    Convention: yaw (heading) is around Y-up axis and returned in key "y".
    """
    # Extract components
    if hasattr(quat, "w") and hasattr(quat, "x") and hasattr(quat, "y") and hasattr(quat, "z"):
        w = float(quat.w); x = float(quat.x); y = float(quat.y); z = float(quat.z)
    elif isinstance(quat, dict) and all(k in quat for k in ("w", "x", "y", "z")):
        w = float(quat["w"]); x = float(quat["x"]); y = float(quat["y"]); z = float(quat["z"])
    elif isinstance(quat, (list, tuple)) and len(quat) == 4:
        w = float(quat[0]); x = float(quat[1]); y = float(quat[2]); z = float(quat[3])
    else:
        raise ValueError(f"Unsupported quaternion format: {type(quat)}")

    # Normalize to avoid drift
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Tait–Bryan angles for Y-up system
    # Heading (yaw) around Y, pitch around X, roll around Z
    # Reference formulas adapted for Y-up:
    # yaw_y = atan2(2*(w*y + z*x), 1 - 2*(y*y + x*x))
    # pitch_x = asin(2*(w*x - y*z))
    # roll_z = atan2(2*(w*z + x*y), 1 - 2*(z*z + x*x))
    yaw_y = math.atan2(2.0 * (w * y + z * x), 1.0 - 2.0 * (y * y + x * x))

    # Clamp asin argument to [-1, 1]
    t2 = 2.0 * (w * x - y * z)
    t2 = max(-1.0, min(1.0, t2))
    pitch_x = math.asin(t2)

    roll_z = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (z * z + x * x))

    to_deg = 180.0 / math.pi
    return {
        "x": (pitch_x * to_deg),
        "y": (yaw_y * to_deg) % 360.0,
        "z": (roll_z * to_deg),
    }


def quaternion_rotate_vector(quat: Any, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by quaternion (w, x, y, z). Returns np.ndarray shape (3,).

    Accepts quat as [w,x,y,z], dict, or object with .w/.x/.y/.z. Vector is array-like.
    """
    if hasattr(quat, "w") and hasattr(quat, "x") and hasattr(quat, "y") and hasattr(quat, "z"):
        w = float(quat.w); x = float(quat.x); y = float(quat.y); z = float(quat.z)
    elif isinstance(quat, dict) and all(k in quat for k in ("w", "x", "y", "z")):
        w = float(quat["w"]); x = float(quat["x"]); y = float(quat["y"]); z = float(quat["z"])
    elif isinstance(quat, (list, tuple)) and len(quat) == 4:
        w = float(quat[0]); x = float(quat[1]); y = float(quat[2]); z = float(quat[3])
    else:
        raise ValueError(f"Unsupported quaternion format: {type(quat)}")

    # Normalize
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Rotation matrix from quaternion
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    R = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)      ],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)      ],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)

    v = np.asarray(vec, dtype=np.float32).reshape(3)
    return R @ v


def _apply_actions_with_set_state(
    controller: habitat_sim.Simulator,
    current_position: Dict[str, float],
    current_rotation: Dict[str, float],
    rot1_deg: float,
    forward_cm: float,
    rot2_deg: float,
    trans_scale: float = 100.0,
):
    """Apply [rot1, forward_cm, rot2] by computing new AgentState and setting it.

    - Rotations are applied about Y-up (yaw) in degrees.
    - Forward direction is agent's -Z after rot1; movement in meters.
    Returns (observations, final_position_dict, final_quaternion)
    """
    agent = controller.get_agent(0)

    # Build initial quaternion from yaw (degrees)
    curr_yaw_deg = float(current_rotation.get("y", 0.0)) 
    curr_quat = quat_from_angle_axis(np.radians(curr_yaw_deg), np.array([0.0, 1.0, 0.0]))

    # Apply first rotation (right-multiply to match previous behavior)
    rot1_quat = quat_from_angle_axis(np.radians(-float(rot1_deg)), np.array([0.0, 1.0, 0.0])) # use negative (+ : right, - : left)
    quat_after_rot1 = curr_quat * rot1_quat

    # Move forward in agent's facing direction (-Z) after rot1
    forward_meters = float(forward_cm) / float(trans_scale)
    start_pos = np.array([current_position["x"], current_position["y"], current_position["z"]], dtype=np.float32)
    if forward_meters > 0.0:
        forward_vec = quaternion_rotate_vector(quat_after_rot1, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        target_pos = start_pos + forward_vec * forward_meters
    else:
        target_pos = start_pos

    # Apply second rotation
    rot2_quat = quat_from_angle_axis(np.radians(float(-rot2_deg)), np.array([0.0, 1.0, 0.0])) # use negative (+ : right, - : left)
    final_quat = quat_after_rot1 * rot2_quat

    # Set final state
    agent_state = AgentState()
    if controller.pathfinder.is_navigable(target_pos) == False:
        # applying FallBack processing 
        for num_discretized_steps in reversed(range(20, 1, -1)):
            fallbacked_forward_meters = forward_meters * num_discretized_steps / 20
            target_pos = start_pos + forward_vec * fallbacked_forward_meters
            if controller.pathfinder.is_navigable(target_pos):
                print(f"[Fallback] Used fallback position: {target_pos} ({num_discretized_steps}/20)")
                break

        if num_discretized_steps == 0:
            print(f"[Error] Target position is not navigable, use original position")
            return None, None, None

    agent_state.position = target_pos
    agent_state.rotation = final_quat
    agent.set_state(agent_state)

    observations = controller.get_sensor_observations()
    final_position_dict = {"x": float(target_pos[0]), "y": float(target_pos[1]), "z": float(target_pos[2])}
    return observations, final_position_dict, final_quat


def build_additional_view(
    controller: habitat_sim.Simulator,
    actions: list[int],
    render_metadata: dict[str, Any] | None = None,
):
    """Build an additional view by updating the 3D scene camera with predicted action parameters.

    This function takes the model's predicted rotation, forward distance, and look angle,
    then applies them to the current camera position in the 3D simulator to render a new view.

    Args:
        controller: Habitat-Sim Simulator instance for 3D scene rendering
        actions: List of three integers [rotation_angle_degrees, forward_distance_cm, final_look_angle_degrees]
                 - rotation_angle_degrees: How much to rotate the camera from current orientation [0, 360)
                 - forward_distance_cm: How far to move forward in centimeters after rotating
                 - final_look_angle_degrees: Additional rotation after reaching new position [0, 360)
        render_metadata: Dictionary containing current camera state
                        - position: Current camera position {x, y, z}
                        - rotation: Current camera rotation {x, y, z}
                        - scene_index: Scene identifier
                        - trans_scale: Scale factor to convert centimeters to simulator units
        question: Natural language question to extract object name from
        pixel_threshold: Minimum pixel count for object to be considered visible

    Returns:
        Tuple[PIL.Image, dict]: Rendered view and metadata dict containing:
            - used_fallback: True if target was unreachable and fallback was used
            - target_position: Original predicted target position (optional)
            - actual_position: Actual reached position (optional)
            - last_event: Last observations from simulator (on failure, for debugging)
        Returns (None, dict_with_last_obs) on failure
    """

    if actions is None or render_metadata is None:
        if DEBUG_MODE:
            print("[!] Failed to load render_metadata")
        return None, {"last_event": None, "error": "Missing actions or render_metadata"}

    current_position = render_metadata.get("position")
    current_rotation = render_metadata.get("rotation")
    trans_scale = float(
        render_metadata.get("trans_scale", 100.0)
    )  # default: 100 (cm to meters conversion)
    if not current_position or not current_rotation:
        raise ValueError("Current position or rotation is not set")

    rot1, trans_cm, rot2 = (actions + [0, 0, 0])[:3]


    obs, final_position_dict, final_rot_quat = _apply_actions_with_set_state(
        controller,
        current_position,
        current_rotation,
        rot1,
        trans_cm,
        rot2,
        trans_scale=trans_scale,
    )
    
    if obs is None:
        print(f"[Fallback:] Failed to apply actions")
        return None, {"last_event": None, "error": "Target position is not navigable"}
    # Convert quaternion to Euler angles (simplified - just extract yaw)
    # For simplicity, we'll just track the yaw component
    final_rot = {
        "x": 0.0,
        "y": 0.0,  # Would need quat_to_euler conversion
        "z": 0.0,
    }

    curr_x = current_position["x"]
    curr_y = current_position["y"]
    curr_z = current_position["z"]
    curr_rot = current_rotation["y"]
    final_x = float(final_position_dict["x"])
    final_y = float(final_position_dict["y"])
    final_z = float(final_position_dict["z"])

    if DEBUG_MODE:
        print(
            f"[Pred]: rot1: {rot1}, trans_cm: {trans_cm}, rot2: {rot2} | [Action] ({curr_x:.2f}, {curr_y:.2f}, {curr_z:.2f}, {curr_rot:.2f}) -> ({final_x:.2f}, {final_y:.2f}, {final_z:.2f}, {final_rot['y']:.2f})"
        )

    # Get the final rendered frame from observations
    if obs is None:
        obs = controller.get_sensor_observations()
    frame = obs.get("rgb")
    if frame is None:
        # Fallback to color_sensor if present
        frame = obs.get("color_sensor")
        if frame is None:
            if DEBUG_MODE:
                print(f"[Error] No frame returned")
            return None, {"last_event": obs, "error": "No frame returned", "event_metadata": obs}

    result_metadata = {
        "answer": None,
        "used_fallback": False,
        "target_position": None,
        "actual_position": {"x": final_x, "y": final_y, "z": final_z},
    }


    return Image.fromarray(frame), result_metadata
