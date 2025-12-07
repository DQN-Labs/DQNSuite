#!/usr/bin/env python3
"""Worker script to visualize or evaluate a saved model in a separate process.
This script is intended to be launched by the GUI as a separate process so
rendering can run on the child process main thread (avoids macOS/Tk/pyglet
native crashes when mixing GUI toolkits in one process).

Usage examples:
  python src/visualize_worker.py --model models/foo_PPO.zip --env CartPole-v1 --max-steps 100000 --render
  python src/visualize_worker.py --model models/foo_PPO.zip --env CartPole-v1 --eval-steps 5000 --render --stats-out /tmp/stats.json

"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

try:
    import gym
except Exception:
    print("Failed to import gym", file=sys.stderr)
    raise

try:
    from stable_baselines3 import PPO, DQN, A2C, TD3, DDPG, SAC
except Exception:
    print("Failed to import stable_baselines3", file=sys.stderr)
    raise

ALGOS = {"PPO": PPO, "DQN": DQN, "A2C": A2C, "TD3": TD3, "DDPG": DDPG, "SAC": SAC}

shutdown_requested = False


def sigterm_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


def detect_algo_from_filename(path: str) -> str | None:
    up = path.upper()
    for a in ALGOS.keys():
        if ("_" + a + ".ZIP") in up or up.endswith("_" + a + ".ZIP"):
            return a
    return None


def load_model(model_path: str):
    algo = detect_algo_from_filename(model_path)
    if algo and algo in ALGOS:
        ModelClass = ALGOS[algo]
        return ModelClass.load(model_path)
    # fallback: try each
    last_err = None
    for ModelClass in ALGOS.values():
        try:
            return ModelClass.load(model_path)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError("Could not load model")


def _atomic_write(path: str, data: dict):
    try:
        tmp = path + ".tmp"
        with open(tmp, 'w') as fh:
            json.dump(data, fh)
        try:
            os.replace(tmp, path)
        except Exception:
            os.rename(tmp, path)
    except Exception:
        pass


def run_visualize(model_path: str, env_id: str, max_steps: int, render: bool, heartbeat: str | None = None):
    model = load_model(model_path)
    env = gym.make(env_id)
    try:
        obs = env.reset()
    except Exception:
        obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs

    step = 0
    while not shutdown_requested and step < max_steps:
        try:
            action, _ = model.predict(obs, deterministic=True)
            out = env.step(action)
        except Exception as e:
            print(f"Runtime error during visualize: {e}", file=sys.stderr)
            break

        if len(out) == 4:
            obs, reward, done, info = out
        else:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated

        if render:
            try:
                env.render()
            except Exception:
                # rendering may not be available
                pass

        if done:
            try:
                obs = env.reset()
            except Exception:
                obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
        step += 1

        # write heartbeat occasionally
        if heartbeat and (step % 10 == 0 or step == 1):
            _atomic_write(heartbeat, {"step": step})
    try:
        env.close()
    except Exception:
        pass


def run_evaluation(model_path: str, env_id: str, eval_steps: int, render: bool, stats_out: str | None, heartbeat: str | None = None):
    model = load_model(model_path)
    env = gym.make(env_id)

    episode_reward = 0.0
    episode_len = 0
    episode_rewards = []
    episode_lengths = []

    try:
        obs = env.reset()
    except Exception:
        obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs

    step = 0
    while not shutdown_requested and step < eval_steps:
        try:
            action, _ = model.predict(obs, deterministic=True)
            out = env.step(action)
        except Exception as e:
            print(f"Runtime error during evaluation: {e}", file=sys.stderr)
            break

        if len(out) == 4:
            obs, reward, done, info = out
        else:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated

        episode_reward += float(reward)
        episode_len += 1

        if render:
            try:
                env.render()
            except Exception:
                pass

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_len)
            episode_reward = 0.0
            episode_len = 0
            try:
                obs = env.reset()
            except Exception:
                obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs

        step += 1

        # write heartbeat occasionally
        if heartbeat and (step % 10 == 0 or step == 1):
            _atomic_write(heartbeat, {"step": step})

    # finalize
    if episode_len > 0:
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_len)

    stats = {}
    if episode_rewards:
        import numpy as _np
        arr = _np.array(episode_rewards)
        stats = {
            "episodes": len(arr),
            "mean_reward": float(_np.mean(arr)),
            "std_reward": float(_np.std(arr)),
            "min_reward": float(_np.min(arr)),
            "max_reward": float(_np.max(arr)),
            "median_reward": float(_np.median(arr)),
            "avg_episode_length": float(_np.mean(episode_lengths)) if episode_lengths else 0.0,
        }
    else:
        stats = {"episodes": 0}

    if stats_out:
        try:
            with open(stats_out, 'w') as fh:
                json.dump(stats, fh)
        except Exception as e:
            print(f"Failed to write stats: {e}", file=sys.stderr)

    # final heartbeat indicating completion
    if heartbeat:
        try:
            _atomic_write(heartbeat, {"step": step, "done": True})
        except Exception:
            pass

    try:
        env.close()
    except Exception:
        pass


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--env', required=True)
    p.add_argument('--max-steps', type=int, default=1000000)
    p.add_argument('--eval-steps', type=int, default=0)
    p.add_argument('--render', action='store_true')
    p.add_argument('--heartbeat', type=str, default=None)
    p.add_argument('--stats-out', type=str, default=None)
    args = p.parse_args(argv)

    if args.eval_steps > 0:
        run_evaluation(args.model, args.env, args.eval_steps, args.render, args.stats_out, args.heartbeat)
    else:
        run_visualize(args.model, args.env, args.max_steps, args.render, args.heartbeat)


if __name__ == '__main__':
    main()
