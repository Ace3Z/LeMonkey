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
import logging
import signal
import sys
import time
from pathlib import Path
from threading import Event

import numpy as np
import torch
from pynput import keyboard


# Silence the noisy 'Relative goal position magnitude had to be clamped' warning
# from lerobot.robots.utils.ensure_safe_goal_position. We're intentionally
# clamping via max_relative_target — it's expected, not a problem to flag every frame.
class _DropClampWarning(logging.Filter):
    def filter(self, record):
        return "Relative goal position magnitude had to be clamped" not in record.getMessage()


logging.getLogger().addFilter(_DropClampWarning())
for _h in logging.getLogger().handlers:
    _h.addFilter(_DropClampWarning())

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
p.add_argument("--follower-port", default="/dev/so101-follower")
p.add_argument("--leader-port",   default="/dev/so101-leader")
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
p.add_argument("--max-relative-target", type=float, default=8.0,
               help="Per-frame joint motion cap (degrees). Smooths sudden "
                    "policy-output jumps when transitioning back from teleop. "
                    "Set to 0 (or negative) to disable.")
args = p.parse_args()


# ─── Keyboard listener for SPACEBAR-as-intervene ─────────────────────────────

intervene          = Event()   # True = teleop active; toggled by SPACE
quit_flag          = Event()
rest_request       = Event()   # 'r' — manual home reset
next_ep_request    = Event()   # 'n' — end episode early, save, advance
delete_last_req    = Event()   # 'd' — mark last saved episode for deletion at end of run
delete_all_req     = Event()   # 'a' — mark ALL saved episodes for deletion at end of run

pending_deletes    = []        # episode indices queued for deletion

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
        # Character keys (e.g. 'r', 'n', 'd', 'a')
        try:
            ch = key.char.lower() if key.char else ""
            if ch == "r":
                rest_request.set()
            elif ch == "n":
                next_ep_request.set()
            elif ch == "d":
                delete_last_req.set()
            elif ch == "a":
                delete_all_req.set()
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
_max_rel = args.max_relative_target if args.max_relative_target > 0 else None
follower_cfg = SOFollowerRobotConfig(
    port=args.follower_port,
    id=args.follower_id,
    cameras={"camera1": cam_cfg},  # match policy's expected camera key
    max_relative_target=_max_rel,
)
if _max_rel is not None:
    print(f"  follower max_relative_target = {_max_rel}° / frame "
          f"(clamps sudden policy-output jumps)")
follower = SOFollower(follower_cfg)
follower.connect()

leader_cfg = SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id)
leader = SOLeader(leader_cfg)
leader.connect()

# Bilateral teleop: torque the leader so we can drive it to mirror the follower
# while the policy is in control. Disabled while user takes over (SPACE on).
leader.bus.enable_torque()

print("OK robot+teleop connected (bilateral mode: leader will mirror follower).\n")


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
MOTOR_NAMES = [k.removesuffix(".pos") for k in ACTION_KEYS]


def action_dict_to_array(act: dict) -> np.ndarray:
    """Pack a teleop/robot action dict into a (6,) array in canonical SO-101 order."""
    return np.array([float(act[k]) for k in ACTION_KEYS], dtype=np.float32)


def drive_leader_to(state_arr: np.ndarray) -> None:
    """Send Goal_Position to the leader's motors so it tracks the follower.

    Used in bilateral teleop: while the policy drives the follower, we
    actively command the leader's motors to mirror the follower's joint
    angles. This gives haptic feedback (you can feel the policy's plan
    in the leader) and means that when you press SPACE to take over, the
    leader is already aligned with the follower — no anchored-delta needed.

    `state_arr`: (6,) follower joint positions in canonical order.
    """
    target = {motor: float(state_arr[i]) for i, motor in enumerate(MOTOR_NAMES)}
    leader.bus.sync_write("Goal_Position", target)


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
print("                   - OFF (default): policy drives follower; leader is")
print("                     actively driven to mirror the follower (haptic feedback)")
print("                   - ON: leader torque released; you drive the follower")
print("                     anchored from current pose")
print("  PRESS 'n'      end this episode NOW, save it, advance")
print("                   - use after a successful pick+place to skip dead time")
print("  PRESS 'r'      release torque + manually home both arms (rest mode)")
print("                   - mid-episode: discards in-progress episode and redoes it")
print("                   - between episodes: just rests, then continues")
print("  PRESS 'd'      mark the LAST saved episode for deletion (applied at run end)")
print("  PRESS 'a'      mark ALL saved episodes for deletion (applied at run end)")
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

    # Clear any stray key events that happened during the calibration
    # prompts or the input() above. Without this, an accidental SPACE press
    # would put us in TELEOP from frame 0 of the next episode.
    intervene.clear()
    rest_request.clear()
    next_ep_request.clear()
    delete_last_req.clear()
    delete_all_req.clear()
    print(f"  recording for {args.episode_time_s}s — SPACE teleop, 'n' end-now, 'r' rest, 'd'/'a' del-last/del-all\n")

    policy.reset()
    n_frames = 0
    n_interventions = 0
    t_start = time.time()

    # Bilateral teleop with anchored-delta in teleop mode:
    #   policy mode: leader torqued, mirrors follower (haptic feedback)
    #   teleop mode: leader torque off; follower target =
    #     follower_anchor + (leader_now - leader_anchor)
    #     This keeps the follower steady at the toggle moment regardless of
    #     where the user's hand happens to be on the leader; only their
    #     deliberate motion translates to the follower.
    teleop_on = False
    leader_anchor = None
    follower_anchor = None
    rest_during_episode = False
    end_early = False

    while time.time() - t_start < args.episode_time_s:
        loop_start = time.time()

        # ─ Mid-episode rest request: discard partial episode and redo it ─
        if rest_request.is_set():
            rest_during_episode = True
            print("\n  🛌 'r' pressed — discarding in-progress episode and entering rest mode.")
            break

        # ─ Mid-episode end-now request: save what we have and advance ─
        if next_ep_request.is_set():
            next_ep_request.clear()
            end_early = True
            print(f"\n  ⏭   'n' pressed — ending episode early, saving and advancing.")
            break

        # ─ Read observation (state + image) ─
        obs = get_obs_for_dataset()
        follower_state_arr = obs["observation.state"]  # (6,) current joint pos

        # ─ Read leader's current pose ─
        leader_act_dict = leader.get_action()
        leader_act_arr = action_dict_to_array(leader_act_dict)

        # ─ Compute policy action ONLY when in policy mode (saves ~50ms / frame in teleop) ─
        policy_act_arr = None
        if not intervene.is_set():
            policy_act_tensor = predict_action(
                obs, policy, torch.device(args.device), preprocessor, postprocessor,
                use_amp=False, task=args.task, robot_type=follower.robot_type,
            )
            policy_act_arr = policy_act_tensor.cpu().numpy().reshape(-1)

        # ─ Toggle transitions ─
        space_active = intervene.is_set()
        if space_active and not teleop_on:
            # Just toggled on → release leader torque so the human can backdrive it.
            # CRITICAL: capture the follower's CURRENT pose as the anchor and
            # the leader's CURRENT pose as the leader anchor BEFORE we read
            # any further leader motion. While teleop is on, the follower's
            # target = follower_anchor + (leader_now - leader_anchor), so the
            # user's hand-on-leader doesn't immediately yank the follower.
            teleop_on = True
            try:
                leader.bus.disable_torque()
            except Exception as e:
                print(f"  (leader torque release failed: {e})")
            follower_anchor = follower_state_arr.copy()
            leader_anchor = leader_act_arr.copy()
            print(f"\n  ▶  TELEOP ON — leader released, follower anchored at current pose. "
                  f"Move the leader to drive the follower from here.")
        elif not space_active and teleop_on:
            # Toggled off → re-engage leader torque, policy resumes.
            # CRITICAL: clear the policy's stale action chunk so it doesn't
            # replay actions queued from before the takeover (those were
            # generated for the pre-intervention observation and are
            # garbage now that the scene has changed).
            teleop_on = False
            try:
                leader.bus.enable_torque()
            except Exception as e:
                print(f"  (leader torque re-engage failed: {e})")
            policy.reset()
            leader_anchor = None
            follower_anchor = None
            print(f"  ◀  TELEOP OFF — policy reset, fresh inference from current state.")

        # ─ Choose action ─
        if teleop_on:
            # Anchored delta: stay at follower_anchor + leader's motion since toggle.
            # When the user isn't actively moving the leader, leader_now ≈ leader_anchor
            # so chosen_arr ≈ follower_anchor → follower holds steady.
            delta = leader_act_arr - leader_anchor
            chosen_arr = follower_anchor + delta
            is_int = True
            n_interventions += 1
        else:
            chosen_arr = policy_act_arr
            is_int = False

        chosen_dict = array_to_action_dict(chosen_arr)
        follower.send_action(chosen_dict)

        # ─ Bilateral feedback: drive leader to follower while in policy mode ─
        if not teleop_on:
            try:
                drive_leader_to(follower_state_arr)
            except Exception as e:
                # Don't crash the episode if leader write fails transiently
                pass

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
    saved_idx = ep_idx
    ep_idx += 1

    # Handle deletion requests for the just-saved episode (or all)
    if delete_all_req.is_set():
        delete_all_req.clear()
        pending_deletes.clear()
        pending_deletes.extend(range(ep_idx))  # mark all 0..ep_idx-1
        print(f"  📝 'a' pressed — marked ALL {ep_idx} saved episodes for deletion at end of run")
    elif delete_last_req.is_set():
        delete_last_req.clear()
        if saved_idx not in pending_deletes:
            pending_deletes.append(saved_idx)
        print(f"  📝 'd' pressed — marked episode {saved_idx} for deletion at end of run "
              f"(total queued: {len(pending_deletes)})")

    # Reset time (interruptible by 'r' / 'd' / 'a')
    if ep_idx < args.num_episodes:
        print(f"  resetting for {args.reset_time_s}s — 'r' rest, 'd' delete-last, 'a' delete-all ...")
        sleep_until = time.time() + args.reset_time_s
        while time.time() < sleep_until:
            if rest_request.is_set() or quit_flag.is_set():
                break
            if delete_last_req.is_set():
                delete_last_req.clear()
                if saved_idx not in pending_deletes:
                    pending_deletes.append(saved_idx)
                print(f"  📝 marked episode {saved_idx} for deletion (queued: {len(pending_deletes)})")
            if delete_all_req.is_set():
                delete_all_req.clear()
                pending_deletes.clear()
                pending_deletes.extend(range(ep_idx))
                print(f"  📝 marked ALL {ep_idx} episodes for deletion")
            time.sleep(0.1)


# ─── Cleanup ─────────────────────────────────────────────────────────────────

print("\nFinalizing dataset ...")
dataset.finalize()
_shutdown("normal end of recording")

# Apply queued deletions via lerobot-edit-dataset (it expects the dataset to be
# closed first, which finalize() handled).
if pending_deletes:
    import subprocess
    eps = sorted(set(pending_deletes))
    print(f"\n=== Deleting {len(eps)} marked episode(s): {eps} ===")
    cmd = [
        "lerobot-edit-dataset",
        f"--repo_id={args.dataset_repo_id}",
        f"--root={args.dataset_root}",
        f"--new_repo_id={args.dataset_repo_id}",
        f"--new_root={args.dataset_root}",
        "--operation.type=delete_episodes",
        f"--operation.episode_indices={list(eps)}",
    ]
    print("  $ " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f"  ✓ deletion complete — {len(eps)} episode(s) removed")
    else:
        print(f"  ✗ deletion failed (rc={rc}). Episodes still on disk; you can run the cmd manually.")
try:
    follower.disconnect()
except Exception:
    pass
try:
    leader.disconnect()
except Exception:
    pass
print("Done.")
