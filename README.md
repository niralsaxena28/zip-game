# zip-game
Reinforcement learning + A* solver for a grid-based “Zip Game” puzzle, with terminal and Streamlit web interfaces.
# RL Zip Game – Reinforcement Learning Pathfinding Puzzle

This project implements the **Zip Game**, a grid-based puzzle where an agent must:

- Start at the cell containing number `1`
- Visit all numbered targets `1 → 2 → … → N` in order
- Visit **every cell exactly once**
- End on the final number `N`

The project combines:

- A **Q-Learning** agent (model-free reinforcement learning)
- An **A\*** backup solver (deterministic heuristic search)
- A **human-playable interface** (terminal + Streamlit web UI)

---

## Features

- ✅ Random puzzle generation for grid sizes from **3×3 to 5×5**
- ✅ Q-Learning agent that learns a path using a shaped reward function
- ✅ A\* search as a guaranteed backup solver when RL fails
- ✅ Ensures only **solvable** puzzles are shown to the player
- ✅ Two ways to play:
  - **Terminal UI** (text-based)
  - **Streamlit Web UI** (graphical, in the browser)
- ✅ Comparison between the player’s path and the AI’s optimal path

---

## Project Structure

```text
.
├── zip_game.py                # Environment, Q-Learning agent, A* solver, CLI game loop
├── streamlit_app.py           # Streamlit web UI
├── FULL_CODE_EXPLANATION.md   # Detailed explanation of the code (for learning/report)
├── requirements.txt
└── README.md
