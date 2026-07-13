"""
Train a PPO agent to play Pokemon Red/Blue using Stable-Baselines3.

Usage:
    python train.py --rom PokemonRed.gb --timesteps 2000000 --num-envs 8

Resume (NOTE: action space is now 8 actions including Start — old 7-action
checkpoints will not load; start a fresh run):
    python train.py --rom PokemonRed.gb --model runs/final_model.zip --headless
"""

import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from pokemon_env import PokemonRedEnv
from map_viz import MapVizCallback


def make_env(rom_path, headless, max_steps, act_freq, rank=0):
    def _init():
        env = PokemonRedEnv(
            rom_path=rom_path,
            headless=headless,
            max_steps=max_steps,
            act_freq=act_freq,
        )
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", required=True, help="Path to your Pokemon ROM file")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Parallel emulator instances. Use ~CPU core count for real runs.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--render", dest="headless", action="store_false")
    parser.add_argument("--max-steps", type=int, default=8192, help="Steps per episode")
    parser.add_argument("--act-freq", type=int, default=24, help="Emulator frames per action")
    parser.add_argument("--model", default=None, help="Existing model .zip to continue training")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--map-viz-freq", type=int, default=20_000,
                        help="Save visit heatmaps every N total steps. 0 disables.")
    parser.add_argument("--frame-stack", type=int, default=4,
                        help="Stack last N frames so the policy can see motion. 1 disables.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy bonus — higher = more exploration (helps escape menus/walls).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    state_path = args.rom + ".state"
    if not os.path.exists(state_path):
        print(
            f"WARNING: missing save state '{state_path}'.\n"
            f"  Create one with: python make_save_state.py --rom {args.rom}\n"
            f"  Stand in the overworld (not in battle/menus), then close the window.",
            file=sys.stderr,
        )

    if args.num_envs > 1:
        env = SubprocVecEnv([
            make_env(args.rom, args.headless, args.max_steps, args.act_freq, i)
            for i in range(args.num_envs)
        ])
    else:
        env = DummyVecEnv([make_env(args.rom, args.headless, args.max_steps, args.act_freq)])

    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)

    if args.model:
        try:
            model = PPO.load(args.model, env=env)
        except Exception as e:
            print(
                f"Failed to load {args.model}: {e}\n"
                "If you upgraded from the old 7-action env, start a fresh model "
                "(omit --model). Frame-stack / action-space mismatches also break loads.",
                file=sys.stderr,
            )
            env.close()
            sys.exit(1)
        print(f"Resumed model from {args.model}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=max(2048 // max(args.num_envs, 1), 64),
            batch_size=64,
            learning_rate=2.5e-4,
            gamma=0.998,
            ent_coef=args.ent_coef,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=os.path.join(args.out_dir, "tensorboard"),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(20_000 // max(args.num_envs, 1), 1),
        save_path=args.out_dir,
        name_prefix="pokemon_ppo",
    )

    callbacks = [checkpoint_cb]
    if args.map_viz_freq > 0:
        callbacks.append(MapVizCallback(
            save_freq=max(args.map_viz_freq // max(args.num_envs, 1), 1),
            out_dir=os.path.join(args.out_dir, "maps"),
            verbose=1,
        ))

    reset_num_timesteps = args.model is None
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )
    final_path = os.path.join(args.out_dir, "final_model")
    model.save(final_path)
    print(f"Done. Model saved to {final_path}.zip")
    env.close()


if __name__ == "__main__":
    main()
