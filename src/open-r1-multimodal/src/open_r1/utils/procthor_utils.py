import atexit
import gzip
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, Optional, Callable, Tuple

import numpy as np
import torch
import prior
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ControllerFailureError(Exception):
    """Raised when controller operations fail after maximum retries.
    
    This exception is raised when a controller operation fails after the maximum
    number of retries (default: 3). Training should be terminated when this occurs.
    """
    pass


_PROC_CONTROLLER: Optional[Controller] = None
_PROC_DATASET = None  # Cache ProcTHOR dataset

DEBUG_MODE = str(os.getenv("DEBUG_MODE", "0")) == "1"

def _get_local_rank():
    """Get local rank for debug output control."""
    return int(os.getenv("LOCAL_RANK", "0"))


def _get_rank_info():
    """Infer local rank and world size from deepspeed/torch.distributed envs.

    Returns a dict: {"local_rank": int | None, "world_size": int | None}
    """
    local_rank = None
    world_size = None
    try:
        # Preferred: LOCAL_RANK provided by deepspeed/torchrun
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])  # 0..n-1 on each node
        # Fallback: derive from RANK and LOCAL_WORLD_SIZE
        elif "RANK" in os.environ and "LOCAL_WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])  # global rank
            lws = int(os.environ["LOCAL_WORLD_SIZE"])  # per-node world size
            local_rank = rank % max(1, lws)
        # Torch distributed introspection as a last resort
        if world_size is None and "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])  # total processes
    except Exception:
        pass
    return {"local_rank": local_rank, "world_size": world_size}


def get_procthor_controller(
    headless: bool = True, width: int = 512, height: int = 512, force_new: bool = False
) -> Controller:
    """Return a lazily-initialized ProcTHOR Controller bound to the current local GPU.

    - Binds gpuDevice to LOCAL_RANK (override with AITHOR_GPU_DEVICE)
    - Uses CloudRendering for headless multi-GPU rendering
    - Registers atexit cleanup to stop the controller when the process exits
    - Checks controller health and recreates if connection is broken

    Args:
        headless: Use CloudRendering for headless mode
        width: Render width
        height: Render height
        force_new: Force create a new controller even if one exists
    """
    global _PROC_CONTROLLER

    # Check if existing controller is still alive
    if not force_new and _PROC_CONTROLLER is not None:
        if _is_controller_alive(_PROC_CONTROLLER):
            return _PROC_CONTROLLER
        else:
            print("[procthor] Existing controller is dead, recreating...")
            _reset_controller()

    rank_info = _get_rank_info()
    # Resolve GPU device in the following priority:
    # 1) AITHOR_GPU_DEVICE env (explicit override)
    # 2) LOCAL_RANK (torchrun/deepspeed)
    # 3) torch.cuda.current_device() if available (respects CUDA_VISIBLE_DEVICES)
    # 4) Derive from RANK/LOCAL_WORLD_SIZE
    # 5) Fallback: 0
    explicit = os.getenv("AITHOR_GPU_DEVICE")
    if explicit is not None and explicit != "":
        gpu_device = int(explicit)
        source = "AITHOR_GPU_DEVICE"
    elif rank_info.get("local_rank") is not None:
        gpu_device = int(rank_info["local_rank"])
        source = "LOCAL_RANK"
    elif torch.cuda.is_available():
        try:
            gpu_device = int(torch.cuda.current_device())
            source = "torch.cuda.current_device"
        except Exception:
            gpu_device = 0
            source = "fallback"
    else:
        # final fallback
        gpu_device = 0
        source = "fallback"


    

    # GPU 0 -> 127.0.0.1, GPU 1 -> 127.0.0.2, etc.
    # For GPU >= 127, use a different pattern to avoid invalid IPs
    if gpu_device < 127:
        resolved_host = f"127.0.0.{gpu_device + 1}"
    else:
        # For higher GPU numbers, use 127.1.0.X pattern
        resolved_host = f"127.1.{gpu_device // 256}.{gpu_device % 256}"

    try:
        _PROC_CONTROLLER = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            gridSize=0.25,
            snapToGrid=False,
            rotateStepDegrees=90,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            width=width,
            height=height,
            fieldOfView=90,
            #platform=CloudRendering if headless else None,
            gpuDevice=gpu_device,
            host=resolved_host,
        )
        _PROC_CONTROLLER.step(action="PausePhysicsAutoSim") 
    except Exception as e:
        print(f"[procthor] Error creating controller: {e}")
        raise

    def _cleanup():
        try:
            if _PROC_CONTROLLER is not None:
                _PROC_CONTROLLER.stop()
        except Exception:
            pass

    atexit.register(_cleanup)
    if DEBUG_MODE and _get_local_rank() == 0:
        print(
            f"[procthor] Controller initialized (gpuDevice={gpu_device}, src={source}, LOCAL_RANK={rank_info.get('local_rank')}, headless={headless}, host={resolved_host})"
        )
    return _PROC_CONTROLLER


def get_procthor_dataset(custom_house_path: Optional[Dict[str, Any]] = None, split="train"):
    """Load and cache ProcTHOR-10k dataset."""
    global _PROC_DATASET
    if _PROC_DATASET is None:
        if custom_house_path is not None:
            _PROC_DATASET = {}  # Initialize as dict first
            # for split in ["train", "val", "test"]:
            if DEBUG_MODE and _get_local_rank() == 0:
                print(
                    f"[procthor] Loading custom house path for {split}: {custom_house_path}"
                )
            _PROC_DATASET[split] = []
            house_path = custom_house_path#[split]
            if not house_path or not os.path.exists(house_path):
                if DEBUG_MODE and _get_local_rank() == 0:
                    print(f"No custom house path for {split}")
            with gzip.open(house_path, "rt", encoding="utf-8") as f:
                for line in f:
                    json_line = json.loads(line)
                    _PROC_DATASET[split].append(json_line)
            return _PROC_DATASET
        else:
            print("[procthor] Loading ProcTHOR-10k dataset...")
            _PROC_DATASET = prior.load_dataset("procthor-10k")
        print(f"[procthor] Dataset loaded: {len(_PROC_DATASET[split])} houses")
    return _PROC_DATASET


def get_procthor_house(custom_house_path: Dict[str, Any], house_index: int, split: str = "train"):
    """Get ProcTHOR house object by index."""
    dataset = get_procthor_dataset(custom_house_path=custom_house_path, split=split)
    if house_index < 0 or house_index >= len(dataset[split]):
        print(f"[procthor] Warning: house_index {house_index} out of range, using 0")
        house_index = 0
    return dataset[split][house_index]


# LEGACY
def customize_scene(controller: Controller, modified_metadata: dict):
    logger.warning("`customize_scene()` is deprecated.")
    """Customize scene by loading the custom house path."""
    # remove object
    removed_object_ids = modified_metadata["removed_object_ids"]
    injected_object_ids = modified_metadata["injected_objects"]
    injected_object_positions = modified_metadata["injected_object_positions"]

    for oid in removed_object_ids:
        event, success = _safe_controller_step(controller, action="DisableObject", objectId=oid)
        if not success:
            print(f"[Customize Scene] Failed to disable object {oid}")
            continue
        event, success = _safe_controller_step(controller, action="Pass")
        if not success:
            print(f"[Customize Scene] Failed to pass after disabling object {oid}")
            continue
        if DEBUG_MODE and _get_local_rank() == 0:
            print(f"[Customize Scene] Removed object {oid}")

    for oid, position in zip(injected_object_ids, injected_object_positions):
        event, success = _safe_controller_step(controller, action="EnableObject", objectId=oid)
        if not success:
            print(f"[Customize Scene] Failed to enable object {oid}")
            continue
        event, success = _safe_controller_step(controller, action="PlaceObjectAtPoint", objectId=oid, position=position)
        if not success:
            print(f"[Customize Scene] Failed to place object {oid} at {position}")
            continue
        event, success = _safe_controller_step(controller, action="Pass")
        if not success:
            print(f"[Customize Scene] Failed to pass after placing object {oid}")
            continue
        if DEBUG_MODE and _get_local_rank() == 0:
            print(f"[Customize Scene] Added object {oid} at {position}")


def _is_controller_alive(controller: Controller) -> bool:
    """Check if the controller is still responsive."""
    if controller is None:
        return False
    try:
        # Try a simple operation to check if connection is alive
        _ = controller.last_event
        return True
    except (BrokenPipeError, EOFError, ConnectionError, AttributeError) as e:
        print(f"[procthor] Controller health check failed: {type(e).__name__}")
        return False
    except Exception as e:
        print(f"[procthor] Controller health check unexpected error: {e}")
        return False


def _reset_controller():
    """Force reset the global controller (used when connection is broken)."""
    global _PROC_CONTROLLER
    if _PROC_CONTROLLER is not None:
        # try:
        #     _PROC_CONTROLLER.stop()
        # except Exception:
        #     pass
        _PROC_CONTROLLER = None


def _safe_controller_operation(
    operation_name: str,
    operation_func: Callable,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    recreate_controller_on_failure: bool = True,
) -> Tuple[Any, bool]:
    """Safely execute a controller operation with retry logic.
    
    Args:
        operation_name: Name of the operation for logging (e.g., "reset", "step")
        operation_func: Function to execute (should return the result)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        recreate_controller_on_failure: Whether to recreate controller on failure
    
    Returns:
        Tuple[result, success]: The operation result and success flag (always True if returned)
    
    Raises:
        ControllerFailureError: If controller operation fails after maximum retries.
            Training will be terminated when this exception is raised.
    """
    global _PROC_CONTROLLER
    last_error = None
    for attempt in range(max_retries):
        try:
            result = operation_func()
            return result, True
        except Exception as e:
            last_error = e
            error_type = type(e).__name__
            print(f"[procthor] {operation_name} failed ({error_type}): {e}, attempt {attempt + 1}/{max_retries}")
            print(f"[procthor] Traceback:")
            traceback.print_exc()
            if recreate_controller_on_failure:
                print(f"[procthor] Recreating controller and retrying...")
                _reset_controller()
                _PROC_CONTROLLER = get_procthor_controller(force_new=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    print(f"[procthor] {operation_name} failed after {max_retries} attempts. Last error: {last_error}")
    print(f"[procthor] Controller failed after {max_retries} retries. Terminating training.")
    if last_error is not None:
        print(f"[procthor] Final error traceback:")
        traceback.print_exception(type(last_error), last_error, last_error.__traceback__)
    raise ControllerFailureError(
        f"Controller {operation_name} failed after {max_retries} retry attempts. Last error: {last_error}"
    )


def _safe_controller_reset(controller: Controller, scene=None) -> bool:
    """Safely reset the controller with retry logic.
    
    Args:
        controller: Controller instance
        scene: Scene to reset to (optional)
    
    Returns:
        bool: True if successful
    
    Raises:
        ControllerFailureError: If controller operation fails after maximum retries
    """
    def reset_operation():
        # Use global controller if it was recreated, otherwise use the passed one
        current_controller = _PROC_CONTROLLER if _PROC_CONTROLLER is not None else controller
        if scene is not None:
            current_controller.reset(scene=scene)
            current_controller.step(action="PausePhysicsAutoSim")
        else:
            current_controller.reset()
            current_controller.step(action="PausePhysicsAutoSim")
        return True
    
    
    result, success = _safe_controller_operation("reset", reset_operation)
    return success


def _safe_controller_step(controller: Controller, **kwargs) -> Tuple[Any, bool]:
    """Safely execute controller.step with retry logic.
    
    Args:
        controller: Controller instance
        **kwargs: Arguments to pass to controller.step
    
    Returns:
        Tuple[event, success]: The step event and success flag (True if successful)
    
    Raises:
        ControllerFailureError: If controller operation fails after maximum retries
    """
    def step_operation():
        # Use global controller if it was recreated, otherwise use the passed one
        current_controller = _PROC_CONTROLLER if _PROC_CONTROLLER is not None else controller
        return current_controller.step(**kwargs)
    
    event, success = _safe_controller_operation("step", step_operation)
    return event, success

def set_physical_state(controller: Controller, render_metadata: dict):
    """Set the physical state of the controller."""
    print('[set_physical_state]', render_metadata.get("state"))
    state = render_metadata.get("state")
    object_id = render_metadata.get("object_id")

    if state == "OpenObject":
        controller.step(action="OpenObject", openness=0.7, objectId=object_id, forceAction=True)
    elif state == "FillObjectWithLiquid":
        controller.step(action="FillObjectWithLiquid", fillLiquid='coffee', objectId=object_id, forceAction=True)
    elif state in ['CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', 'EmptyLiquidFromObject', 'UseUpObject']:
        controller.step(action=state, objectId=object_id, forceAction=True)
    else:
        raise ValueError(f"Invalid state: {state}")



def build_additional_view(
    controller: Controller,
    actions: list[int],
    render_metadata: dict,
    # split: str = "train",
):
    """Build an additional view by updating the 3D scene camera with predicted action parameters.

    This function takes the model's predicted rotation, forward distance, and look angle,
    then applies them to the current camera position in the 3D simulator to render a new view.

    Args:
        controller: AI2-THOR controller instance for 3D scene rendering
        actions: List of three integers [rotation_angle_degrees, forward_distance_cm, final_look_angle_degrees]
                 - rotation_angle_degrees: How much to rotate the camera from current orientation (-180, 180]. Positive = right, negative = left
                 - forward_distance_cm: How far to move forward in centimeters after rotating
                 - final_look_angle_degrees: Additional rotation after reaching new position (-180, 180]. Positive = right, negative = left
        render_metadata: Dictionary containing current camera state
                        - position: Current camera position {x, y, z}
                        - rotation: Current camera rotation {x, y, z}
                        - scene_index: Scene identifier
                        - trans_scale: Scale factor to convert centimeters to simulator units

    Returns:
        Tuple[PIL.Image, dict]: Rendered view and metadata dict containing:
            - used_fallback: True if target was unreachable and fallback was used
            - target_position: Original predicted target position (optional)
            - actual_position: Actual reached position (optional)
            - last_event: Last event from controller (on failure, for debugging)
        Returns (None, dict_with_last_event) on failure
    """

    if actions is None or render_metadata is None:
        if DEBUG_MODE and _get_local_rank() == 0:
            print("[!] Failed to load render_metadata")
        return None, {"last_event": None, "error": "Missing actions or render_metadata"}

    scene_index = render_metadata.get("scene_index")
    custom_house_path = render_metadata.get("custom_house_path", None)
    # print('[build_additional_view]', custom_house_path)
    current_position = render_metadata.get("position")
    current_rotation = render_metadata.get("rotation")
    split = render_metadata.get("data_split", "train")
    trans_scale = float(
        render_metadata.get("trans_scale", 100.0)
    )  # default: 100 (cm to meters conversion)
    use_fallback = render_metadata.get(
        "use_fallback", True
    )  # Default: use fallback for invalid actions
    num_grids = int(render_metadata.get("num_grids", 20))  # Default: 20 grid points
    # modified_metadata = render_metadata.get("modified_metadata", None)

    if not current_position or not current_rotation:
        raise ValueError("Current position or rotation is not set")

    rot1, trans_cm, rot2 = (actions + [0, 0, 0])[:3]
    trans_meters = (
        float(trans_cm) / trans_scale
    )  # Convert cm to meters (trans_cm / 100.0)

    if not _is_controller_alive(controller):
        controller = get_procthor_controller()

    # Reset/ensure scene is active
    last_event = None
    if scene_index is not None:
        house = get_procthor_house(
            custom_house_path=custom_house_path, house_index=scene_index, split=split
        )
        success = _safe_controller_reset(controller, scene=house)
        if not success:
            print(f"[procthor] Failed to reset controller with scene_index={scene_index}")
            return None, {"last_event": None, "error": "Failed to reset controller", "scene_index": scene_index}

    if render_metadata.get("state", None):
        set_physical_state(controller, render_metadata)

    if custom_house_path:
        print(f"[procthor] tick 5 seconds to stabilize the scene ...")
        for _ in range(100):
            controller.step(action="AdvancePhysicsStep", timeStep=0.05)
        
    # Check if target is reachable, and get fallback if not
    is_feasible, target_pos, fallback_pos = _check_action_feasibility_fast(
        controller,
        current_position,
        current_rotation,
        rot1,
        trans_cm,
        trans_scale,
        find_fallback=use_fallback,
        num_grids=num_grids,
    )

    used_fallback = False
    actual_target_pos = target_pos

    # If action is infeasible and fallback is disabled, return early with error
    if not is_feasible and not use_fallback:
        if DEBUG_MODE and _get_local_rank() == 0:
            print(
                "[Feasibility] Action infeasible and fallback disabled. Returning None with error."
            )
        return None, {
            "last_event": None,
            "error": "Action infeasible (fallback disabled)",
            "used_fallback": False,
            "target_position": target_pos,
        }

    # If target is not feasible but we have a fallback, use it
    if not is_feasible and fallback_pos is not None and use_fallback:
        used_fallback = True
        actual_target_pos = fallback_pos
        if DEBUG_MODE and _get_local_rank() == 0:
            print(f"[Fallback] Target unreachable, using fallback position")
            print(
                f"  Predicted: ({target_pos['x']:.2f}, {target_pos['z']:.2f}) -> ({fallback_pos['x']:.2f}, {fallback_pos['z']:.2f})"
            )

    # Step 0: Teleport to initial position with initial rotation
    event, success = _safe_controller_step(
        controller,
        action="Teleport",
        position=current_position,
        rotation=current_rotation,
        forceAction=True,
    )
    last_event = event
    if not success or event is None:
        print(f"[procthor] Failed to teleport to initial position")
        return None, {"last_event": last_event, "error": "Failed to teleport to initial position", "action": "Teleport", "position": current_position}

    # Step 1: Rotate by rot1 degrees (rot1 is in (-180, 180] range: positive = right, negative = left)
    if rot1 != 0:
        # Convert (-180, 180] to [0, 360) for internal calculation
        rot1_normalized = rot1 if rot1 > 0 else rot1 + 360
        # Use RotateRight for positive values, RotateLeft for negative values
        if rot1 > 0:
            action_name = "RotateRight"
            degrees = float(rot1)
        else:
            action_name = "RotateLeft"
            degrees = float(-rot1)  # RotateLeft takes positive value
        event, success = _safe_controller_step(
            controller,
            action=action_name,
            degrees=degrees,
        )
        last_event = event
        if not success or event is None:
            print(f"[procthor] Failed to rotate by {rot1} degrees")
            return None, {"last_event": last_event, "error": f"Failed to rotate by {rot1} degrees", "action": action_name, "degrees": degrees}

    # Step 2: Move forward - use Teleport to target position
    if trans_meters > 0:
        # Calculate intermediate rotation (after rot1, before movement)
        # Convert rot1 from (-180, 180] to [0, 360) for internal calculation
        rot1_normalized = rot1 if rot1 >= 0 else rot1 + 360
        intermediate_rotation = {
            "x": current_rotation["x"],
            "y": (current_rotation["y"] + rot1_normalized) % 360,
            "z": current_rotation["z"],
        }
        # Teleport to target position
        event, success = _safe_controller_step(
            controller,
            action="Teleport",
            position=actual_target_pos,
            rotation=intermediate_rotation,
            forceAction=True,
        )
        last_event = event
        if not success or event is None:
            print(f"[procthor] Failed to teleport to target position")
            return None, {"last_event": last_event, "error": f"Failed to teleport to target position", "action": "Teleport", "position": actual_target_pos}

    # Step 3: Rotate by rot2 degrees (rot2 is in (-180, 180] range: positive = right, negative = left)
    if rot2 != 0:
        # Convert (-180, 180] to [0, 360) for internal calculation
        rot2_normalized = rot2 if rot2 >= 0 else rot2 + 360
        # Use RotateRight for positive values, RotateLeft for negative values
        if rot2 > 0:
            action_name = "RotateRight"
            degrees = float(rot2)
        else:
            action_name = "RotateLeft"
            degrees = float(-rot2)  # RotateLeft takes positive value
        event, success = _safe_controller_step(
            controller,
            action=action_name,
            degrees=degrees,
        )
        last_event = event
        if not success or event is None:
            print(f"[procthor] Failed to rotate by {rot2} degrees")
            return None, {"last_event": last_event, "error": f"Failed to rotate by {rot2} degrees", "action": action_name, "degrees": degrees}

    if not event.metadata.get("lastActionSuccess", False):
        if DEBUG_MODE and _get_local_rank() == 0:
            print(f"[Error] Action failed")
        return None, {"last_event": event, "error": "lastActionSuccess=False", "event_metadata": event.metadata if event else None}

    # Final state
    agent = event.metadata.get("agent", {})
    final_pos = agent.get("position")
    final_rot = agent.get("rotation")

    curr_x = current_position["x"]
    curr_y = current_position["y"]
    curr_z = current_position["z"]
    curr_rot = current_rotation["y"]
    final_x = final_pos["x"]
    final_y = final_pos["y"]
    final_z = final_pos["z"]
    final_rot = final_rot["y"]

    if DEBUG_MODE and _get_local_rank() == 0:
        print(
            f"[Pred]: rot1: {rot1}, trans_cm: {trans_cm}, rot2: {rot2} | [Action] ({curr_x:.2f}, {curr_y:.2f}, {curr_z:.2f}, {curr_rot:.2f}) -> ({final_x:.2f}, {final_y:.2f}, {final_z:.2f}, {final_rot:.2f})"
        )

    # Get the final rendered frame
    frame = getattr(event, "frame", None)

    result_metadata = {
        "answer": None,
        "used_fallback": used_fallback,
        "target_position": target_pos,
        "actual_position": final_pos,
    }
    return Image.fromarray(frame), result_metadata


def _check_action_feasibility_fast(
    controller: Controller,
    current_position: dict,
    current_rotation: dict,
    rot1: float,
    trans_cm: float,
    trans_scale: float = 100.0,
    find_fallback: bool = False,
    num_grids: int = 8,
) -> tuple[bool, dict, dict | None]:
    """
    Fast check if action is feasible without full rendering.
    Uses Teleport with forceAction=False to test positions directly.

    Args:
        controller: AI2-THOR controller
        current_position: Current agent position {x, y, z}
        current_rotation: Current agent rotation {x, y, z}
        rot1: First rotation angle (degrees) in (-180, 180] range (positive = right, negative = left)
        trans_cm: Forward movement distance (centimeters)
        trans_scale: Scale factor for cm to meters conversion
        find_fallback: If True and target is unreachable, find the furthest reachable position on the line
        num_grids: Number of grid points to check along the line (default: 8)

    Returns:
        Tuple[bool, dict, dict | None]: (is_feasible, target_position, fallback_position)
            - is_feasible: True if target is reachable
            - target_position: Original predicted target position
            - fallback_position: Valid position on the line to target (only if find_fallback=True and target unreachable)
    """
    # Calculate target position after rotation
    # Convert rot1 from (-180, 180] to [0, 360) for internal calculation
    rot1_normalized = rot1 if rot1 >= 0 else rot1 + 360
    trans_meters = float(trans_cm) / trans_scale
    current_yaw = current_rotation["y"]
    intermediate_yaw = (current_yaw + rot1_normalized) % 360
    intermediate_yaw_rad = np.radians(intermediate_yaw)

    dx = trans_meters * np.sin(intermediate_yaw_rad)
    dz = trans_meters * np.cos(intermediate_yaw_rad)

    target_position = {
        "x": current_position["x"] + dx,
        "y": current_position["y"],
        "z": current_position["z"] + dz,
    }

    # Calculate intermediate rotation (after rot1, before movement)
    intermediate_rotation = {
        "x": current_rotation["x"],
        "y": intermediate_yaw,
        "z": current_rotation["z"],
    }

    if not find_fallback:
        # Simple case: just test target
        is_reachable = _test_position_reachable(
            controller, target_position, intermediate_rotation
        )
        return is_reachable, target_position, None

    # Find furthest reachable position (includes testing target first)
    reachable_position = _find_fallback_position_on_line(
        controller,
        current_position,
        intermediate_rotation,
        target_position,
        num_grids,
        include_target=True,
    )

    if reachable_position is None:
        # No position found, stay at current
        return False, target_position, current_position

    # Check if we found the target itself or a fallback
    # Threshold is HARD-CODED.
    is_target = (
        abs(reachable_position["x"] - target_position["x"]) < 0.01
        and abs(reachable_position["z"] - target_position["z"]) < 0.01
    )

    if is_target:
        return True, target_position, None
    else:
        return False, target_position, reachable_position




def _find_fallback_position_on_line(
    controller: Controller,
    current_position: dict,
    current_rotation: dict,
    target_position: dict,
    num_grids: int = 8,
    include_target: bool = True,
) -> dict | None:
    """
    Find the furthest reachable position on the line from current to target.
    Uses Teleport with forceAction=False to test each grid point directly.

    Args:
        controller: AI2-THOR controller
        current_position: Starting position {x, y, z}
        current_rotation: Current rotation {x, y, z} (preserved during check)
        target_position: Desired target position {x, y, z}
        num_grids: Number of grid points to check along the line (default: 8)
        include_target: If True, test target first (α=1.0) before grid points

    Returns:
        Furthest reachable position along the line (could be target itself), or None if no valid position found
    """
    # Direction vector from current to target
    dx = target_position["x"] - current_position["x"]
    dz = target_position["z"] - current_position["z"]
    distance = np.sqrt(dx**2 + dz**2)

    if distance < 0.01:  # Too small movement
        if DEBUG_MODE and _get_local_rank() == 0:
            print(
                f"  Target {target_position} is too close to current position {current_position}, skipping"
            )
        return target_position

    # First, test target itself if requested (α=1.0)
    if include_target:
        if _test_position_reachable(controller, target_position, current_rotation):
            if DEBUG_MODE and _get_local_rank() == 0:
                print(f"  Target {target_position} is reachable, returning")
            return target_position

    # Check grid points from furthest to closest (excluding target which is α=1.0)
    # Test: α = (num_grids-1)/num_grids, (num_grids-2)/num_grids, ..., 1/num_grids
    for i in range(num_grids - 1, 0, -1):
        alpha = i / num_grids

        grid_point = {
            "x": current_position["x"] + alpha * dx,
            "y": current_position["y"],
            "z": current_position["z"] + alpha * dz,
        }

        if _test_position_reachable(controller, grid_point, current_rotation):
            if DEBUG_MODE and _get_local_rank() == 0:
                print(f"Grid point {grid_point} is reachable, returning")
            return grid_point

    # No reachable position found on the line
    if DEBUG_MODE and _get_local_rank() == 0:
        print(f"No reachable position found on the line, returning None")
    return None


def _test_position_reachable(
    controller: Controller, position: dict, rotation: dict
) -> bool:
    """
    Test if a position is reachable using Teleport with forceAction=False.

    Args:
        controller: AI2-THOR controller
        position: Position to test {x, y, z}
        rotation: Rotation to use {x, y, z}

    Returns:
        True if position is reachable, False otherwise
    """
    test_event, success = _safe_controller_step(
        controller,
        action="Teleport",
        position=position,
        rotation=rotation,
        forceAction=False,  # Will fail if unreachable
    )
    if not success or test_event is None:
        return False
    return test_event.metadata.get("lastActionSuccess", False)

