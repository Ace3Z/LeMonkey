#!/usr/bin/env python3
"""HG-DAgger recorder for SO-101 + SmolVLA.

Press SPACE to TOGGLE teleop ON/OFF. While ON, the leader arm controls the
follower with anchored-delta semantics: when you press SPACE the script
captures the leader and follower poses at that instant, and from then on
the follower target is `follower_anchor + (leader_now - leader_anchor)`.
That means you don't have to physically align the leader with the follower
first — your hand motion just *adds* to wherever the policy left the
follower. Press SPACE again to hand control back to the policy.

Frames are logged to a LeRobotDataset with `is_intervention=1` on the
teleop frames, so during fine-tuning you can either upweight or filter
for the corrective demonstrations.

This implements Human-Gated DAgger (Kelly et al. 2019, "HG-DAgger: Interactive
Imitation Learning with Human Experts") on top of LeRobot v3 datasets.

Why not just use lerobot-record:
  lerobot-record's main loop dispatches to either the policy *or* the teleop,
  not both (lerobot_record.py:379-453). Hybrid action arbitration requires
  a custom main loop, which is what this script provides.

Usage:
    dagger_record.py \\
      --policy-path /home/lemonkey/LeMonkey/eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model \\
      --dataset-root /home/lemonkey/LeMonkey/datasets/eval1_dagger/blue \\
      --dataset-repo-id HBOrtiz/so101_eval1_dagger_blue \\
      --task "Put the banana in the blue colored bowl." \\
      --num-episodes 5 \\
      --episode-time-s 30
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path
from threading import Event

import numpy as np
import torch
from pynput import keyboard

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.control_utils import predict_action


# ─── Args ────────────────────────────────────────────────────────────────────

DEFAULT_POLICY = "/home/lemonkey/LeMonkey/eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model"

p = argparse.ArgumentParser()
p.add_argument("--policy-path", default=DEFAULT_POLICY)
p.add_argument("--dataset-root", required=True)
p.add_argument("--dataset-repo-id", required=True)
p.add_argument("--task", required=True, help="Single task string for the prompt")
p.add_argument("--follower-port", default="/dev/ttyACM1")
p.add_argument("--leader-port",   default="/dev/ttyACM0")
p.add_argument("--follower-id",   default="my_follower")
p.add_argument("--leader-id",     default="my_leader")
p.add_argument("--cam-path",      default="/dev/video0")
p.add_argument("--cam-width",     type=int, default=640)
p.add_argument("--cam-height",    type=int, default=480)
p.add_argument("--fps",           type=int, default=30)
p.add_argument("--num-episodes",  type=int, default=5)
p.add_argument("--episode-time-s", type=float, default=30)
p.add_argument("--reset-time-s",  type=float, default=10)
p.add_argument("--device",        default="cuda")
args = p.parse_args()


# ─── Keyboard listener for SPACEBAR-as-intervene ─────────────────────────────

intervene    = Event()         # True = teleop active; toggled by SPACE
quit_flag    = Event()
rest_request = Event()         # set when user presses 'r' — triggers manual home reset

# Edge-detection state for SPACE so key-repeat doesn't re-toggle every frame
_space_down = False


def on_press(key):
    global _space_down
    if key == keyboard.Key.space:
        # Only toggle on the press-down edge, not while holding
        if not _space_down:
            if intervene.is_set():
                intervene.clear()
            else:
                intervene.set()
        _space_down = True
    elif key == keyboard.Key.esc:
        quit_flag.set()
    else:
        # Character keys (e.g. 'r')
        try:
            if key.char and key.char.lower() == "r":
                rest_request.set()
        except AttributeError:
            pass


def on_release(key):
    global _space_down
    if key == keyboard.Key.space:
        _space_down = False


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# ─── Build follower + leader + camera ────────────────────────────────────────

print("Connecting follower / leader / camera ...")
cam_cfg = OpenCVCameraConfig(
    index_or_path=args.cam_path,
    width=args.cam_width,
    height=args.cam_height,
    fps=args.fps,
)
follower_cfg = SOFollowerRobotConfig(
    port=args.follower_port,
    id=args.follower_id,
    cameras={"camera1": cam_cfg},  # match policy's expected camera key
)
follower = SOFollower(follower_cfg)
follower.connect()

leader_cfg = SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id)
leader = SOLeader(leader_cfg)
leader.connect()

print("OK robot+teleop connected.\n")


# ─── Clean shutdown: disable follower torque on Ctrl+C / exit ────────────────

_shutdown_done = False
def _shutdown(reason: str = ""):
    """Release follower torque so the arm can be moved by hand back to home."""
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True
    print(f"\n[shutdown] {reason}")
    try:
        # Disable torque so the user can manually move the follower home
        if follower.is_connected:
            follower.bus.disable_torque()
            print("[shutdown] follower torque DISABLED — you can now move the arm by hand.")
    except Exception as e:
        print(f"[shutdown] follower torque release failed: {e}")
    try:
        if leader.is_connected:
            leader.bus.disable_torque()  # already disabled in normal use, but be safe
    except Exception:
        pass
    try:
        listener.stop()
    except Exception:
        pass


def _sigint_handler(signum, frame):
    _shutdown("Ctrl+C received")
    sys.exit(130)


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


def rest_arms_interactive():
    """Release follower torque, wait for user to manually home both arms, re-engage."""
    rest_request.clear()
    intervene.clear()
    print("\n  🛌 REST mode — disabling follower torque ...")
    try:
        follower.bus.disable_torque()
        # Leader torque is already off by design; nothing to do there
    except Exception as e:
        print(f"     (could not disable torque: {e})")
    print("     Move BOTH arms to the rest / home position.")
    try:
        input("     Press ENTER when done ... ")
    except (EOFError, KeyboardInterrupt):
        pass
    print("     Re-engaging follower torque at the new pose ...")
    try:
        follower.bus.enable_torque()
    except Exception as e:
        print(f"     (could not re-enable torque: {e})")
    rest_request.clear()  # in case more keypresses queued during input()
    print("     ✓ done — resuming\n")


# ─── Load policy + preprocessor ──────────────────────────────────────────────

print(f"Loading policy from {args.policy_path} ...")
policy = SmolVLAPolicy.from_pretrained(args.policy_path).eval().to(args.device)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=args.policy_path,
    preprocessor_overrides={
        "device_processor": {"device": args.device},
        # No rename_map: the camera is already keyed as 'camera1'
    },
)
print("Policy loaded.\n")


# ─── Build dataset ───────────────────────────────────────────────────────────

# Define features matching what we'll write per frame.  Use the same shapes as
# the original training dataset so the merge will work later.
features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action":            {"dtype": "float32", "shape": (6,), "names": None},
    "observation.images.camera1": {
        "dtype": "video",
        "shape": (args.cam_height, args.cam_width, 3),
        "names": ["height", "width", "channels"],
    },
    "is_intervention":   {"dtype": "int64",   "shape": (1,), "names": None},  # 0 or 1
}

root = Path(args.dataset_root)
if root.exists():
    # Distinguish "real dataset already here" (refuse) from "stale empty
    # directory from a previously crashed init" (safe to remove).
    has_episodes = any(root.glob("data/**/*.parquet")) or any(root.glob("videos/**/*.mp4"))
    if has_episodes:
        print(f"ERROR: {root} already contains episode data. Refusing to overwrite.")
        print(f"       Move/delete it manually if you really want a fresh dataset.")
        raise SystemExit(1)
    import shutil
    print(f"NOTE: removing stale empty dir from a previously-crashed init at {root}")
    shutil.rmtree(root)

dataset = LeRobotDataset.create(
    repo_id=args.dataset_repo_id,
    fps=args.fps,
    root=root,
    robot_type=follower.name,
    features=features,
    use_videos=True,
    image_writer_processes=0,
    image_writer_threads=4,
)
print(f"Dataset created at {root}\n")


# ─── Helpers ─────────────────────────────────────────────────────────────────

ACTION_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def action_dict_to_array(act: dict) -> np.ndarray:
    """Pack a teleop/robot action dict into a (6,) array in canonical SO-101 order."""
    return np.array([float(act[k]) for k in ACTION_KEYS], dtype=np.float32)


def array_to_action_dict(arr) -> dict:
    if torch.is_tensor(arr):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr).reshape(-1)
    return {k: float(v) for k, v in zip(ACTION_KEYS, arr)}


def get_obs_for_dataset() -> dict:
    """Return obs dict in the format `predict_action` and the dataset writer expect."""
    obs = follower.get_observation()
    # follower.get_observation() returns a flat dict with joint positions
    # (e.g. "shoulder_pan.pos", ...) and one entry per camera (key = the name
    # we gave it in the cameras dict, here "camera1").

    # State: pack the 6 joint positions into a (6,) float32 array
    state_arr = np.array(
        [float(obs[k]) for k in ACTION_KEYS], dtype=np.float32
    )

    # Image: try the configured camera key first, then the namespaced fallback
    if "camera1" in obs:
        img = obs["camera1"]
    elif "observation.images.camera1" in obs:
        img = obs["observation.images.camera1"]
    else:
        raise KeyError(f"No camera image found. obs keys: {list(obs.keys())}")

    return {
        "observation.state": state_arr,
        "observation.images.camera1": img,  # HxWx3 uint8 array
    }


# ─── Main per-episode loop ───────────────────────────────────────────────────

print("=" * 60)
print(f"  HG-DAgger recorder")
print(f"  policy   : {args.policy_path}")
print(f"  dataset  : {root}")
print(f"  task     : \"{args.task}\"")
print(f"  episodes : {args.num_episodes}")
print(f"  per-ep s : {args.episode_time_s}")
print()
print("  PRESS SPACE    toggle teleop ON / OFF")
print("                   - ON  : leader controls follower from current pose")
print("                           (anchored delta — leader can stay wherever it is)")
print("                   - OFF : policy resumes from current follower pose")
print("  PRESS 'r'      release torque + manually home both arms (rest mode)")
print("                   - mid-episode: discards in-progress episode and redoes it")
print("                   - between episodes: just rests, then continues")
print("  ESC            quit between episodes")
print("=" * 60)
print()

dt = 1.0 / args.fps

ep_idx = 0
while ep_idx < args.num_episodes:
    if quit_flag.is_set():
        print("Quit requested; stopping.")
        break

    # If user pressed 'r' while we were between episodes, handle it now
    if rest_request.is_set():
        rest_arms_interactive()

    print(f"── Episode {ep_idx + 1} / {args.num_episodes} ──")
    inp = input("Place banana, press ENTER to start  (or type 'r' + ENTER for rest first): ")
    if inp.strip().lower() == "r":
        rest_arms_interactive()
        continue  # redo this episode from the top
    print(f"  recording for {args.episode_time_s}s — press SPACE to toggle teleop, 'r' to rest\n")

    policy.reset()
    n_frames = 0
    n_interventions = 0
    t_start = time.time()

    # Two-state toggle: SPACE flips between "policy" and "teleop".
    # When toggling INTO teleop, capture leader+follower anchors so the leader's
    # subsequent motion adds incrementally to the follower's current pose.
    # The leader can stay physically wherever it is — only its *delta* matters.
    teleop_on = False
    leader_anchor = None        # np.ndarray (6,)
    follower_anchor = None      # np.ndarray (6,)

    rest_during_episode = False

    while time.time() - t_start < args.episode_time_s:
        loop_start = time.time()

        # ─ Mid-episode rest request: discard partial episode and redo it ─
        if rest_request.is_set():
            rest_during_episode = True
            print("\n  🛌 'r' pressed — discarding in-progress episode and entering rest mode.")
            break

        # ─ Read observation (state + image) ─
        obs = get_obs_for_dataset()
        follower_state_arr = obs["observation.state"]  # (6,) current joint pos

        # ─ Compute policy action ─
        policy_act_tensor = predict_action(
            obs, policy, torch.device(args.device), preprocessor, postprocessor,
            use_amp=False, task=args.task, robot_type=follower.robot_type,
        )
        policy_act_arr = policy_act_tensor.cpu().numpy().reshape(-1)

        # ─ Read leader's current pose ─
        leader_act_dict = leader.get_action()
        leader_act_arr = action_dict_to_array(leader_act_dict)

        # ─ Toggle transitions ─
        space_active = intervene.is_set()
        if space_active and not teleop_on:
            # Just toggled on → snapshot anchors at the policy-induced state
            teleop_on = True
            leader_anchor = leader_act_arr.copy()
            follower_anchor = follower_state_arr.copy()
            print(f"\n  ▶  TELEOP ON — leader controls follower from current pose. "
                  f"(press SPACE again to return to policy)")
        elif not space_active and teleop_on:
            # Toggled off → policy resumes from current follower pose
            teleop_on = False
            leader_anchor = None
            follower_anchor = None
            print(f"  ◀  TELEOP OFF — policy resumes.")

        # ─ Choose action ─
        if teleop_on:
            # Anchored delta: target = follower_at_toggle + (leader_now - leader_at_toggle)
            delta = leader_act_arr - leader_anchor
            chosen_arr = follower_anchor + delta
            is_int = True
            n_interventions += 1
        else:
            chosen_arr = policy_act_arr
            is_int = False

        chosen_dict = array_to_action_dict(chosen_arr)
        follower.send_action(chosen_dict)

        # ─ Save frame ─
        frame = {
            "observation.state": obs["observation.state"],
            "observation.images.camera1": obs["observation.images.camera1"],
            "action": chosen_arr.astype(np.float32),
            "is_intervention": np.array([1 if is_int else 0], dtype=np.int64),
            "task": args.task,
        }
        dataset.add_frame(frame)
        n_frames += 1

        # ─ Pace to fps ─
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # End-of-episode handling
    if rest_during_episode:
        # Discard the partial episode buffer and redo this episode
        try:
            dataset.clear_episode_buffer()
        except Exception as e:
            print(f"  (clear_episode_buffer failed: {e})")
        rest_arms_interactive()
        # Don't increment ep_idx — repeat this episode index
        continue

    pct = 100 * n_interventions / max(n_frames, 1)
    print(f"  episode done: {n_frames} frames, "
          f"{n_interventions} intervention frames ({pct:.0f}%)")
    dataset.save_episode()
    ep_idx += 1

    # Reset time (interruptible by 'r')
    if ep_idx < args.num_episodes:
        print(f"  resetting for {args.reset_time_s}s — reposition arm + banana ('r' for rest) ...")
        sleep_until = time.time() + args.reset_time_s
        while time.time() < sleep_until:
            if rest_request.is_set() or quit_flag.is_set():
                break
            time.sleep(0.1)


# ─── Cleanup ─────────────────────────────────────────────────────────────────

print("\nFinalizing dataset ...")
dataset.finalize()
_shutdown("normal end of recording")
try:
    follower.disconnect()
except Exception:
    pass
try:
    leader.disconnect()
except Exception:
    pass
print("Done.")
