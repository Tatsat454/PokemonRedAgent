"""
Renders per-map visit heatmaps during training.

One PNG per location (e.g. pallet_town.png, route_1.png). Brighter = visited more.
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# Early-game + a few IDs seen in prior training heatmaps
MAP_NAMES = {
    0: "pallet_town",
    1: "viridian_city",
    2: "pewter_city",
    3: "cerulean_city",
    12: "route_1",
    13: "route_2",
    14: "route_3",
    15: "route_4",
    33: "viridian_forest",
    37: "players_house_1f",
    38: "players_house_2f",
    39: "rivals_house",
    40: "oaks_lab",
    41: "viridian_nickname_house",
    42: "viridian_school",
    43: "viridian_gate",
    44: "viridian_gym_entrance",
    50: "pewter_museum_1f",
    51: "pewter_gym",
}


def map_name(map_id):
    return MAP_NAMES.get(map_id, f"map_{map_id}")


def _normalize_entries(entries):
    """Accept [(x,y,count)] or legacy [((x,y), count)] formats."""
    out = []
    for item in entries:
        if len(item) == 3:
            x, y, count = item
            out.append((int(x), int(y), int(count)))
        elif len(item) == 2:
            (x, y), count = item
            out.append((int(x), int(y), int(count)))
    return out


def render_heatmap(visit_data, map_id, out_path, grid_size=256):
    if plt is None:
        raise ImportError("matplotlib is required for map_viz.py")
    if not visit_data:
        return

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for x, y, count in visit_data:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[y, x] += count

    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return
    pad = 3
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, grid_size)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, grid_size)
    cropped = grid[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.log1p(cropped), cmap="inferno", interpolation="nearest")
    ax.set_title(map_name(map_id))
    ax.set_xlabel("x tile")
    ax.set_ylabel("y tile")
    fig.colorbar(im, ax=ax, label="log(visits + 1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


class MapVizCallback(BaseCallback):
    def __init__(self, save_freq=20_000, out_dir="runs/maps", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.out_dir = out_dir

    def _init_callback(self):
        os.makedirs(self.out_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self._render_all_maps()
        return True

    def _render_all_maps(self):
        try:
            per_env_data = self.training_env.env_method("get_map_visit_data")
        except AttributeError:
            if self.verbose:
                print("MapVizCallback: no get_map_visit_data() - skipping")
            return

        merged = {}
        for env_data in per_env_data:
            if not env_data:
                continue
            for map_id, entries in env_data.items():
                bucket = merged.setdefault(map_id, {})
                for x, y, count in _normalize_entries(entries):
                    bucket[(x, y)] = bucket.get((x, y), 0) + count

        for map_id, coord_counts in merged.items():
            visit_data = [(x, y, c) for (x, y), c in coord_counts.items()]
            out_path = os.path.join(self.out_dir, f"{map_name(map_id)}.png")
            try:
                render_heatmap(visit_data, map_id, out_path)
            except Exception as e:
                if self.verbose:
                    print(f"MapVizCallback: failed to render map {map_id}: {e}")

        if self.verbose:
            print(f"MapVizCallback: saved {len(merged)} map heatmaps to {self.out_dir}")
