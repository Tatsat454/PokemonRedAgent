"""
Interactive helper: play the ROM yourself past the intro, then close the
window to save <rom>.state. train.py loads that state on every reset.

IMPORTANT — PyBoy keyboard (not Game Boy labels):
    Arrow keys = D-pad
    A          = A button
    S          = B button
    Enter      = Start
    Backspace  = Select
    Esc        = quit (also triggers save)

Recommended save point: standing in Pallet Town / Route 1 overworld with
your starter, NOT in a battle or menu.

Usage:
    python make_save_state.py --rom PokemonRed.gb
"""

import argparse
from pyboy import PyBoy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", required=True)
    args = parser.parse_args()

    pyboy = PyBoy(args.rom, window="SDL2")
    pyboy.set_emulation_speed(1)

    print("Controls: arrows = move, A = A, S = B, Enter = Start, Esc = quit+save.")
    print("Play to a good overworld start (e.g. Pallet Town with starter),")
    print("then close the window or press Esc — state saves automatically.")

    running = True
    while running:
        running = pyboy.tick()

    state_path = args.rom + ".state"
    with open(state_path, "wb") as f:
        pyboy.save_state(f)
    print(f"Saved state to {state_path}")
    pyboy.stop()


if __name__ == "__main__":
    main()
