"""
Watch a trained (or random) policy in visible PyBoy windows.

Usage:
    python watch_live.py --rom PokemonRed.gb --n 2
    python watch_live.py --rom PokemonRed.gb --model runs/final_model.zip --n 2

NOTE: New env has 8 actions (includes Start) and optional 4-frame stacking
in train.py. Old 7-action checkpoints will fail to load — train a fresh model.
"""

import argparse
import time
import sys

import numpy as np
from stable_baselines3 import PPO
from pokemon_env import PokemonRedEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--act-freq", type=int, default=24)
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Must match training (--frame-stack). Use 4 if you trained with stacks.")
    args = parser.parse_args()

    print(f"Opening {args.n} windows...")
    envs = [
        PokemonRedEnv(
            rom_path=args.rom,
            headless=False,
            max_steps=args.max_steps,
            act_freq=args.act_freq,
        )
        for _ in range(args.n)
    ]
    obs_list = [env.reset()[0] for env in envs]
    # Maintain simple frame stacks locally if needed
    stacks = None
    if args.frame_stack > 1:
        stacks = [
            np.repeat(obs, args.frame_stack, axis=-1)
            for obs in obs_list
        ]

    model = None
    if args.model:
        try:
            model = PPO.load(args.model)
        except Exception as e:
            print(f"Failed to load model: {e}", file=sys.stderr)
            for env in envs:
                env.close()
            sys.exit(1)
        n_actions = model.action_space.n
        if n_actions != envs[0].action_space.n:
            print(
                f"Action space mismatch: model={n_actions}, env={envs[0].action_space.n}.\n"
                "Old 7-action checkpoints are incompatible with the new 8-action env.",
                file=sys.stderr,
            )
            for env in envs:
                env.close()
            sys.exit(1)
    else:
        print("No --model given — watching a random policy.")

    try:
        while True:
            for i, env in enumerate(envs):
                if model is not None:
                    if stacks is not None:
                        action, _ = model.predict(stacks[i], deterministic=False)
                    else:
                        action, _ = model.predict(obs_list[i], deterministic=False)
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(int(action))
                obs_list[i] = obs
                if stacks is not None:
                    stacks[i] = np.concatenate([stacks[i][:, :, 1:], obs], axis=-1)
                if terminated or truncated:
                    obs_list[i] = env.reset()[0]
                    if stacks is not None:
                        stacks[i] = np.repeat(obs_list[i], args.frame_stack, axis=-1)
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
