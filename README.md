# PokemonRedAgent

Reinforcement learning agent that plays **Pokémon Red** using [PyBoy](https://github.com/Baekalfen/PyBoy) + [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO).

Inspired by [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments). This repo documents an independent training run with reward shaping, map heatmaps, and published checkpoints.

## ⚠️ Legal

**No ROM is included.** You must provide your own legally obtained `PokemonRed.gb` (dump from a cartridge you own).

## Results (this experiment)

- Explored: **Pallet Town → Route 1 → Viridian City → Viridian Forest / gate**
- Heatmaps: see [`maps/`](maps/)
- Checkpoints: see [`models/`](models/) (e.g. `pokemon_ppo_4420000_steps.zip`)
- Training curves: [`tensorboard/`](tensorboard/) (`PPO_7` is the main improved run)
- Did **not** earn gym badges or keep caught Pokémon across episodes (save state = Charmander only)

See [`NOTES.md`](NOTES.md) for more findings and limitations (pixel-only observations vs multi-input RAM features).

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Place your ROM here as PokemonRed.gb
# Optional: use included PokemonRed.gb.state or create one:
python make_save_state.py --rom PokemonRed.gb
```

PyBoy controls for save-state creation: **arrows** = move, **A** = A, **S** = B, **Enter** = Start.

## Train

```bash
python train.py --rom PokemonRed.gb --headless --num-envs 8 --timesteps 5000000
```

## Watch a trained model

```bash
python watch_live.py --rom PokemonRed.gb --model models/pokemon_ppo_4420000_steps.zip --n 2 --frame-stack 4
```

## TensorBoard

```bash
tensorboard --logdir tensorboard
```

## Repo layout

| Path | Description |
|------|-------------|
| `pokemon_env.py` | Gymnasium env + reward shaping |
| `train.py` | PPO training (frame stack, checkpoints, map viz) |
| `map_viz.py` | Exploration heatmaps |
| `watch_live.py` | Live PyBoy windows |
| `maps/` | Heatmap PNGs from training |
| `models/` | Selected PPO checkpoints |
| `tensorboard/` | Event logs from training runs |

## License / credit

Experiment code for research/education. Pokémon is © Nintendo/Creatures/GAME FREAK. ROM not distributed.
