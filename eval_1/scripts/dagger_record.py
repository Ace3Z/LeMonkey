#!/usr/bin/env python3
"""HG-DAgger recorder for SO-101 + SmolVLA.

Hold SPACEBAR to take over via the leader arm; release to return to policy
control. Frames are logged to a LeRobotDataset with `is_intervention=True`
on the takeover frames, so during fine-tuning you can either upweight or
filter for the corrective demonstrations.

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
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
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

intervene = Event()
quit_flag = Event()


def on_press(key):
    if key == keyboard.Key.space:
        intervene.set()
    elif key == keyboard.Key.esc:
        quit_flag.set()


def on_release(key):
    if key == keyboard.Key.space:
        intervene.clear()


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
follower_cfg = SOFollowerConfig(
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
    print(f"WARN: root {root} exists; LeRobotDataset.create requires it not to.")
    print("       If you want to resume / append, use lerobot-record's --resume flag separately.")
    raise SystemExit(1)

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
    # follower.get_observation returns a dict with state + images.  Keys depend
    # on the camera dict — we used "camera1" → key is observation.images.camera1.
    state_arr = action_dict_to_array(
        {k.replace(".pos", ".pos"): v for k, v in obs.items() if k in ACTION_KEYS}
    ) if all(k in obs for k in ACTION_KEYS) else None
    if state_arr is None:
        # Some lerobot versions nest state differently; fall back to known keys
        state_arr = np.array(
            [float(obs[k]) for k in ACTION_KEYS], dtype=np.float32
        )
    img = obs.get("camera1") or obs.get("observation.images.camera1")
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
print("  HOLD SPACEBAR to take over the leader arm")
print("  RELEASE     to return to policy control")
print("  ESC         to quit between episodes")
print("=" * 60)
print()

dt = 1.0 / args.fps

for ep_idx in range(args.num_episodes):
    if quit_flag.is_set():
        print("Quit requested; stopping.")
        break

    print(f"── Episode {ep_idx + 1} / {args.num_episodes} ──")
    input("Place banana in target failure zone, press ENTER to start the episode ... ")
    print(f"  recording for {args.episode_time_s}s — hold SPACE to intervene\n")

    policy.reset()
    n_frames = 0
    n_interventions = 0
    t_start = time.time()

    while time.time() - t_start < args.episode_time_s:
        loop_start = time.time()

        # ─ Read observation (state + image) ─
        obs = get_obs_for_dataset()

        # ─ Compute policy action and read leader's current pose ─
        policy_act_tensor = predict_action(
            obs, policy, torch.device(args.device), preprocessor, postprocessor,
            use_amp=False, task=args.task, robot_type=follower.robot_type,
        )
        policy_act_arr = policy_act_tensor.cpu().numpy().reshape(-1)

        leader_act_dict = leader.get_action()
        leader_act_arr = action_dict_to_array(leader_act_dict)

        # ─ Decide which action to send ─
        if intervene.is_set():
            chosen_arr = leader_act_arr
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

    # End-of-episode
    pct = 100 * n_interventions / max(n_frames, 1)
    print(f"  episode done: {n_frames} frames, "
          f"{n_interventions} intervention frames ({pct:.0f}%)")
    dataset.save_episode()

    # Reset time
    if ep_idx < args.num_episodes - 1:
        print(f"  resetting for {args.reset_time_s}s — reposition arm + banana ...")
        time.sleep(args.reset_time_s)


# ─── Cleanup ─────────────────────────────────────────────────────────────────

print("\nFinalizing dataset ...")
dataset.finalize()
follower.disconnect()
leader.disconnect()
listener.stop()
print("Done.")
