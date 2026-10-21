
# PokemonRedAgent
 
A PPO agent that plays **Pokémon Red** from raw pixels, built on [PyBoy](https://github.com/Baekalfen/PyBoy), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), and [Gymnasium](https://gymnasium.farama.org/).
 
This repo accompanies the paper **"Pixels Are Not Enough: An Empirical Study of Reward Shaping and Policy Collapse in a PPO Agent for Pokémon Red"** (Tatsat Upadhyay, July 2026). It contains the environment, training code, checkpoints, TensorBoard logs, and exploration heatmaps for all seven training runs (~15M environment steps total).
 
Inspired by [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments). The key difference: this agent sees **pixels only** and uses a **per-episode** novelty bonus, whereas that project adds structured RAM features and cross-episode exploration tracking. The contrast between the two results is the central finding.
 
## Legal
 
**No ROM is included.** You must supply your own legally obtained copy of Pokémon Red (US) as `PokemonRed.gb`, dumped from a cartridge you own.
 
## TL;DR of the results
 
- The final configuration (run 7, 4.42M steps) improved steadily: mean episode reward rose from ~126 to ~447.
- It explored **Pallet Town → Route 1 → Viridian City → the edge of Viridian Forest**, and went no further.
- It **never earned a badge** and **never kept a caught Pokémon**. Nearly all of its return came from exploration, movement, and wild-battle rewards. The agent learned to be an efficient tourist, not a trainer.
- Run 6 suffered a complete **policy-entropy collapse** (entropy fell to ~1e-23, mean reward fell to -1637). Entropy was the only early-warning signal; reward alone looked fine until it was too late.
- Why it stalls: the per-episode novelty bonus keeps re-rewarding the already-explored start area, which makes pushing deeper into the maze-like forest a worse bet than staying shallow. Pixels alone also hide the state (badges, party, story flags) that determines long-run return.
See the paper for full details and [`NOTES.md`](NOTES.md) for additional findings and limitations.
- https://docs.google.com/document/d/1HaAOO3b5t3lAIz2GwrsnaVq0HMLWEooRY3ZeUTaRJTY/edit?usp=sharing
 
## Runs
 
| Run | Steps | Mean ep. reward (first → last) | Outcome |
|-----|-------|-------------------------------|---------|
| 1 | 4.1K | -1.5 → 0.6 | Smoke test |
| 2 | 455K | -38.9 → -60.3 | Penalties dominated returns |
| 3 | 18K | n/a | Aborted config check |
| 4 | 5.00M | -57.4 → -52.8 | Flat, no game progress |
| 5 | 625K | -161.8 → -139.9 | Heavier penalties, strongly negative |
| 6 | 4.47M | -158.0 → -1637.2 | **Entropy collapse** |
| 7 | 4.42M | +125.8 → +446.7 | **Final config**, steady improvement |
 
> Reward magnitudes are **not comparable across runs** because the reward function changed between them. Within-run trends are meaningful; cross-run comparisons of absolute reward are not.
 
## Setup
 
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
 
# Place your ROM in the repo root as PokemonRed.gb
# Use the included PokemonRed.gb.state, or create your own:
python make_save_state.py --rom PokemonRed.gb
```
 
PyBoy controls for save-state creation: **arrows** = move, **A** = A, **S** = B, **Enter** = Start.
 
The episode start state is saved just after receiving the Charmander starter, which skips the intro.
 
## Train
 
```bash
python train.py --rom PokemonRed.gb --headless --num-envs 8 --timesteps 5000000
```
 
The final run (run 7) used 8 parallel emulators, 4-frame stacking, and about 4.4M steps. Training was done on a consumer laptop (Apple MacBook Air).
 
## Watch a trained model
 
```bash
python watch_live.py --rom PokemonRed.gb --model models/pokemon_ppo_4420000_steps.zip --n 2 --frame-stack 4
```
 
`pokemon_ppo_4420000_steps.zip` is the final run-7 checkpoint. Checkpoints in `models/` range from 100K to 5M steps.
 
## TensorBoard
 
```bash
tensorboard --logdir tensorboard
```
 
Logs for all seven runs are included (`PPO_1` through `PPO_7`). `PPO_7` is the main improved run and `PPO_6` is the entropy-collapse case study.
 
## Environment summary
 
| Item | Value |
|------|-------|
| Observation | 72x80 grayscale screen (2x downsample), 4 stacked frames |
| Actions | Discrete: no-op, up, down, left, right, A, B, Start (8). Runs 1-6 used 7 actions without Start. |
| Frame skip | Each action is held for 24 emulator frames |
| Episode length | 8192 agent steps (about 55 min of game time) |
| Algorithm | PPO, CNN policy (SB3 defaults), lr 2.5e-4, gamma 0.998, clip 0.2, batch 64, entropy coef 0.01 |
| Reward | Dense, computed from emulator RAM (exploration, levels, badges, battles, plus behavioral penalties) |
 
Per-step reward is clipped to [-15, +60]. The full weight table is in Table 1 of the paper and in `pokemon_env.py`.
 
## Exploration heatmaps
 
Tile-visit heatmaps for each map are in [`maps/`](maps/). They show Pallet Town saturated, a funnel through Route 1, narrow corridors in Viridian City, and a shallow band along the entrance of Viridian Forest, with the forest interior untouched.
 
## Repo layout
 
| Path | Description |
|------|-------------|
| `pokemon_env.py` | Gymnasium environment and reward shaping |
| `train.py` | PPO training (frame stack, checkpoints, map visualization) |
| `map_viz.py` | Exploration heatmaps |
| `watch_live.py` | Live PyBoy windows for trained models |
| `make_save_state.py` | Create the episode-start save state |
| `maps/` | Heatmap PNGs from training |
| `models/` | Selected PPO checkpoints |
| `tensorboard/` | Event logs for all seven runs |
 
## Limitations
 
- One seed per configuration, on one machine. Run 6's collapse may not reproduce exactly, though its mechanism is well understood.
- The runs are sequential debugging iterations, not controlled ablations.
- ~4.4M steps for the final config is far below what comparable projects use for mid-game progress, so longer training might get further.
- The 8192-step episode budget may be too short for the full behavior chain.
## Future work
 
1. Persist novelty across episodes.
2. Add structured RAM features (position, party state, badge bits) to the observation.
3. Use archive- or curiosity-based exploration (Go-Explore, RND).
4. Only then, scale compute.
## Citation
 
```bibtex
@misc{upadhyay2026pixels,
  title  = {Pixels Are Not Enough: An Empirical Study of Reward Shaping and Policy Collapse in a PPO Agent for Pok\'emon Red},
  author = {Upadhyay, Tatsat},
  year   = {2026},
  month  = {July}
}
```
 
## License and credits
 
Code is released for research and education. <!-- TODO: add a LICENSE file (e.g. MIT) and name it here -->
 
Pokémon is © Nintendo / Creatures Inc. / GAME FREAK Inc. This project is not affiliated with or endorsed by them. No ROM is distributed.
