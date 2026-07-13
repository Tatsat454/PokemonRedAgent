# Pokemon Red RL — Experiment Findings Package

## Contents
- `maps/` — exploration heatmaps (where the agent walked)
- `models/` — selected PPO checkpoints (not all 250)
- `tensorboard/` — training curves (PPO_1 … PPO_7)
- `code/` — environment, training, viz scripts
- `PokemonRed.gb.state` — episode start save state (optional for reproduction)
- **No ROM included** — supply your own legally obtained `PokemonRed.gb`

## Highlights
- Agent reached: Pallet Town → Route 1 → Viridian City → Viridian Forest / gate
- No gym badges / no lasting catches observed (starter Charmander only in save state)
- Later runs: reward shaping (menu/stuck/battle), Start button, 4-frame stack
- Limitation vs PWhiddy: pixels-only policy (no Dict/MultiInput RAM features)

## View TensorBoard
```bash
tensorboard --logdir tensorboard
```

## Watch a checkpoint
```bash
# from code/ with ROM + .state beside it
python watch_live.py --rom PokemonRed.gb --model ../models/pokemon_ppo_4400000_steps.zip --n 2 --frame-stack 4
```
