"""
Gymnasium environment for training an RL agent to play Pokemon Red/Blue
using PyBoy as the Game Boy emulator backend.

Lessons baked in from prior training runs:
- Agents get stuck on the title screen without Start -> Start is an action
- Agents camp the battle party/switch menu -> menu_camp + idle_battle penalties
- Agents wall-grind after local tiles are explored -> stuck penalty + move reward
- Raw per-HP damage rewards were huge and unstable -> damage is fractional + capped
- Heal farming at Pokemon Centers -> heal weight lowered
- map_viz expects [(x, y, count), ...] not Counter.items() 2-tuples

RAM addresses are for Pokemon Red/Blue (US).
"""

from collections import defaultdict, Counter
import os
import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent


# ---- Known RAM addresses (Pokemon Red/Blue, US) ----
ADDR_X_POS = 0xD362
ADDR_Y_POS = 0xD361
ADDR_MAP_N = 0xD35E
ADDR_PARTY_SIZE = 0xD163
ADDR_PARTY_HP_START = 0xD16C
ADDR_PARTY_MAXHP_START = 0xD18D
ADDR_BADGES = 0xD356
ADDR_PARTY_LEVEL_START = 0xD18C
ADDR_EVENT_FLAGS_START = 0xD747
ADDR_EVENT_FLAGS_END = 0xD87E
ADDR_IN_BATTLE = 0xD057          # 0=none, 1=wild, 2=trainer, 0xFF=lost
ADDR_BATTLE_MON_HP = 0xD015
ADDR_ENEMY_MON_HP = 0xCFE6
ADDR_PARTY_MENU_TYPE = 0xD07D    # wPartyMenuTypeOrMessageID
ADDR_ACTIVE_PARTY_IDX = 0xCC2F

PARTY_MENU_BATTLE = 0x02         # "Choose a POKéMON" during battle

# Tuned from observed failure modes (menu camp, wall grind, reward explosion)
DEFAULT_REWARD_WEIGHTS = {
    "new_coord": 0.05,
    "new_map": 8.0,
    "badge": 50.0,
    "level": 1.0,
    "catch": 8.0,
    "heal": 0.5,            # was 2.5 — too easy to farm Pokemon Centers
    "hurt": 2.0,            # was 5.0 — made agent battle-averse
    "event": 2.0,
    "step_penalty": 0.01,
    "menu_camp": 0.12,      # punish unnecessary switch-menu camping
    "idle_battle": 0.05,    # no HP change while fighting / in menus
    "deal_damage": 1.5,     # applied to *fraction* of enemy HP removed, capped
    "win_battle": 4.0,
    "needed_switch": 2.5,   # only when switching after a faint
    "move": 0.008,          # reward actually changing tiles
    "stuck": 0.04,          # wall-bump / stand-still on overworld
    "blackout": 8.0,        # lost a battle / blacked out
}


class PokemonRedEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        rom_path,
        headless=True,
        max_steps=8192,
        act_freq=24,
        reward_weights=None,
        state_path=None,
    ):
        super().__init__()
        self.rom_path = rom_path
        self.state_path = state_path or (rom_path + ".state")
        self.max_steps = max_steps
        self.act_freq = act_freq

        self.w = dict(DEFAULT_REWARD_WEIGHTS)
        if reward_weights:
            self.w.update(reward_weights)

        if not os.path.exists(self.state_path):
            warnings.warn(
                f"No save state at {self.state_path}. Episodes will boot from the "
                "ROM title screen (very slow / agents often get stuck). "
                "Run: python make_save_state.py --rom <rom>",
                UserWarning,
                stacklevel=2,
            )

        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window)
        self.pyboy.set_emulation_speed(0 if headless else 1)

        # 8 discrete actions: noop, U/D/L/R, A, B, Start
        # Start is required for title screens / some menus.
        self.actions = [
            None,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]
        self.release_actions = [
            None,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(72, 80, 1), dtype=np.uint8
        )

        self._reset_episode_state()

    # ---------- helpers ----------
    def _reset_episode_state(self):
        self.step_count = 0
        self.visited_maps = set()
        self.visited_coords = set()
        self.map_visit_counts = getattr(self, "map_visit_counts", None)
        if self.map_visit_counts is None:
            self.map_visit_counts = defaultdict(Counter)
        self.prev_badges = 0
        self.prev_party_size = 0
        self.prev_levels_sum = 0
        self.prev_hp_frac = 1.0
        self.prev_event_bits = 0
        self.prev_in_battle = False
        self.prev_enemy_hp = 0
        self.prev_battle_mon_hp = 0
        self.prev_active_party_idx = 0
        self.idle_battle_steps = 0
        self.prev_coord = None
        self.stuck_steps = 0
        self.prev_battle_flag = 0

    def _read_m(self, addr):
        return self.pyboy.memory[addr]

    def _read_2byte(self, addr):
        return (self._read_m(addr) << 8) + self._read_m(addr + 1)

    def _get_screen(self):
        screen = self.pyboy.screen.ndarray[:, :, 0]
        small = screen[::2, ::2]
        return small[..., None].astype(np.uint8)

    def _badge_count(self):
        return bin(self._read_m(ADDR_BADGES)).count("1")

    def _party_hp_fraction(self):
        size = self._read_m(ADDR_PARTY_SIZE)
        if size == 0:
            return 1.0
        total_hp, total_max = 0, 0
        for i in range(min(size, 6)):
            cur = self._read_2byte(ADDR_PARTY_HP_START + i * 0x2C)
            mx = self._read_2byte(ADDR_PARTY_MAXHP_START + i * 0x2C)
            total_hp += cur
            total_max += max(mx, 1)
        if total_max == 0:
            return 1.0
        return total_hp / total_max

    def _levels_sum(self):
        size = self._read_m(ADDR_PARTY_SIZE)
        total = 0
        for i in range(min(size, 6)):
            total += self._read_m(ADDR_PARTY_LEVEL_START + i * 0x2C)
        return total

    def _event_flag_bits(self):
        total = 0
        for addr in range(ADDR_EVENT_FLAGS_START, ADDR_EVENT_FLAGS_END):
            total += bin(self._read_m(addr)).count("1")
        return total

    def _battle_flag(self):
        return self._read_m(ADDR_IN_BATTLE)

    def _in_battle(self):
        return self._battle_flag() in (1, 2)

    def _lost_battle(self):
        return self._battle_flag() == 0xFF

    def _battle_mon_hp(self):
        return self._read_2byte(ADDR_BATTLE_MON_HP)

    def _enemy_mon_hp(self):
        return self._read_2byte(ADDR_ENEMY_MON_HP)

    def _in_party_menu(self):
        return self._read_m(ADDR_PARTY_MENU_TYPE) == PARTY_MENU_BATTLE

    def _active_party_idx(self):
        return self._read_m(ADDR_ACTIVE_PARTY_IDX)

    def _load_state(self):
        try:
            with open(self.state_path, "rb") as f:
                self.pyboy.load_state(f)
            return True
        except FileNotFoundError:
            return False

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_state()
        self._reset_episode_state()

        map_n = self._read_m(ADDR_MAP_N)
        x, y = self._read_m(ADDR_X_POS), self._read_m(ADDR_Y_POS)
        self.visited_maps = {map_n}
        self.prev_badges = self._badge_count()
        self.prev_party_size = self._read_m(ADDR_PARTY_SIZE)
        self.prev_levels_sum = self._levels_sum()
        self.prev_hp_frac = self._party_hp_fraction()
        self.prev_event_bits = self._event_flag_bits()
        self.prev_in_battle = self._in_battle()
        self.prev_battle_flag = self._battle_flag()
        self.prev_enemy_hp = self._enemy_mon_hp() if self.prev_in_battle else 0
        self.prev_battle_mon_hp = self._battle_mon_hp() if self.prev_in_battle else 0
        self.prev_active_party_idx = self._active_party_idx()
        self.prev_coord = (map_n, x, y)

        return self._get_screen(), {
            "map_id": map_n,
            "has_save_state": os.path.exists(self.state_path),
        }

    def step(self, action_idx):
        action_idx = int(action_idx)
        press = self.actions[action_idx]
        release = self.release_actions[action_idx]
        if press is not None:
            self.pyboy.send_input(press)
        for _ in range(self.act_freq):
            self.pyboy.tick()
        if release is not None:
            self.pyboy.send_input(release)

        self.step_count += 1
        reward, info = self._compute_reward()
        terminated = False
        truncated = self.step_count >= self.max_steps
        return self._get_screen(), float(reward), terminated, truncated, info

    def _compute_reward(self):
        reward = 0.0
        w = self.w
        info = {}

        x, y = self._read_m(ADDR_X_POS), self._read_m(ADDR_Y_POS)
        map_n = self._read_m(ADDR_MAP_N)
        coord = (map_n, x, y)
        self.map_visit_counts[map_n][(x, y)] += 1

        # --- exploration ---
        if map_n not in self.visited_maps:
            self.visited_maps.add(map_n)
            reward += w["new_map"]
            info["new_map"] = map_n
        if coord not in self.visited_coords:
            self.visited_coords.add(coord)
            reward += w["new_coord"]

        in_battle = self._in_battle()
        battle_flag = self._battle_flag()

        # --- overworld movement vs wall-grinding ---
        if not in_battle and not self._lost_battle():
            if self.prev_coord is not None and coord != self.prev_coord:
                reward += w["move"]
                self.stuck_steps = 0
            else:
                self.stuck_steps += 1
                if self.stuck_steps > 6:
                    # Cap ramp so a long stuck streak can't explode the return
                    reward -= w["stuck"] * min((self.stuck_steps - 6) / 4.0, 2.5)
        else:
            self.stuck_steps = 0
        self.prev_coord = coord

        # --- badges ---
        badges = self._badge_count()
        if badges > self.prev_badges:
            reward += w["badge"] * (badges - self.prev_badges)
            info["badge"] = badges
            self.prev_badges = badges

        # --- levels ---
        levels = self._levels_sum()
        if levels > self.prev_levels_sum:
            reward += w["level"] * (levels - self.prev_levels_sum)
            self.prev_levels_sum = levels

        # --- catch ---
        party_size = self._read_m(ADDR_PARTY_SIZE)
        if party_size > self.prev_party_size:
            reward += w["catch"] * (party_size - self.prev_party_size)
            self.prev_party_size = party_size

        # --- party HP heal / hurt (fractional) ---
        hp_frac = self._party_hp_fraction()
        delta_hp = hp_frac - self.prev_hp_frac
        if delta_hp > 0:
            reward += w["heal"] * delta_hp
        elif delta_hp < 0:
            reward += w["hurt"] * delta_hp
        self.prev_hp_frac = hp_frac

        # --- story flags ---
        event_bits = self._event_flag_bits()
        if event_bits > self.prev_event_bits:
            reward += w["event"] * (event_bits - self.prev_event_bits)
            self.prev_event_bits = event_bits

        # --- battle shaping ---
        battle_hp = self._battle_mon_hp() if in_battle else 0
        enemy_hp = self._enemy_mon_hp() if in_battle else 0
        active_idx = self._active_party_idx()
        switch_needed = in_battle and battle_hp == 0

        # Fractional damage (capped) — old per-HP * 0.5 exploded returns
        if in_battle and self.prev_in_battle and enemy_hp < self.prev_enemy_hp:
            prev = max(self.prev_enemy_hp, 1)
            frac = (self.prev_enemy_hp - enemy_hp) / prev
            reward += w["deal_damage"] * min(frac, 1.0)

        if self.prev_in_battle and not in_battle and not self._lost_battle():
            reward += w["win_battle"]
            info["won_battle"] = True

        # Blackout / lost battle
        if self.prev_battle_flag in (1, 2) and battle_flag == 0xFF:
            reward -= w["blackout"]
            info["blackout"] = True

        # Needed switch after faint
        if (
            not switch_needed
            and self.prev_battle_mon_hp == 0
            and self.prev_in_battle
            and in_battle
            and active_idx != self.prev_active_party_idx
            and battle_hp > 0
        ):
            reward += w["needed_switch"]
            info["needed_switch"] = True

        # Unnecessary party-menu camping
        if in_battle and self._in_party_menu() and not switch_needed:
            reward -= w["menu_camp"]
            info["menu_camp"] = True

        # Idle battle (menus / standing around while HP frozen)
        if in_battle and not switch_needed:
            hp_changed = (
                enemy_hp != self.prev_enemy_hp
                or battle_hp != self.prev_battle_mon_hp
            )
            if hp_changed:
                self.idle_battle_steps = 0
            else:
                self.idle_battle_steps += 1
                if self.idle_battle_steps > 10:
                    reward -= w["idle_battle"]
        else:
            self.idle_battle_steps = 0

        self.prev_in_battle = in_battle
        self.prev_battle_flag = battle_flag
        self.prev_enemy_hp = enemy_hp
        self.prev_battle_mon_hp = battle_hp
        self.prev_active_party_idx = active_idx

        reward -= w["step_penalty"]

        # Soft clip keeps PPO value targets stable when big milestones hit
        reward = float(np.clip(reward, -15.0, 60.0))

        info.update({
            "map_id": map_n,
            "x": x,
            "y": y,
            "in_battle": in_battle,
            "switch_needed": switch_needed,
            "stuck_steps": self.stuck_steps,
            "visited_coords": len(self.visited_coords),
            "visited_maps": len(self.visited_maps),
        })
        return reward, info

    def get_map_visit_data(self):
        """Return {map_id: [(x, y, visit_count), ...]} for MapVizCallback."""
        return {
            map_id: [(x, y, count) for (x, y), count in counter.items()]
            for map_id, counter in self.map_visit_counts.items()
        }

    def close(self):
        self.pyboy.stop()
