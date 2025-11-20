import streamlit as st
from zip_game import ZipGame, QLearningAgent, AStarSolver

MIN_SIZE = 3
MAX_SIZE = 5


def init_session_state():
    if "game" not in st.session_state:
        st.session_state.game = None
    if "game_ready" not in st.session_state:
        st.session_state.game_ready = False
    if "current_pos" not in st.session_state:
        st.session_state.current_pos = None
    if "path" not in st.session_state:
        st.session_state.path = []
    if "visited_numbers" not in st.session_state:
        st.session_state.visited_numbers = set()
    if "next_target" not in st.session_state:
        st.session_state.next_target = 2
    if "message" not in st.session_state:
        st.session_state.message = ""
    if "message_type" not in st.session_state:
        st.session_state.message_type = "info"  # info | warning | error | success
    if "show_solution" not in st.session_state:
        st.session_state.show_solution = False
    if "celebrated" not in st.session_state:
        st.session_state.celebrated = False


def generate_solvable_game(size: int, num_count: int):
    game = ZipGame(size)
    agent = QLearningAgent(size)

    attempts = 0
    max_attempts = 30
    solution = None

    with st.spinner("Generating solvable game (may take a moment)..."):
        while attempts < max_attempts and not solution:
            attempts += 1
            game.generate_game(num_count)

            # Try RL first
            solution = agent.find_path(game, max_episodes=2000)

            # If RL fails, try A*
            if not solution:
                astar = AStarSolver(game)
                solution = astar.solve()

        if not solution:
            st.error(
                f"Could not generate a solvable game after {max_attempts} attempts. "
                "Try a smaller grid or fewer numbers."
            )
            return

    game.solution_path = solution

    # Initialize interactive state
    start_pos = game.numbers[1]
    st.session_state.game = game
    st.session_state.current_pos = start_pos
    st.session_state.path = [start_pos]
    st.session_state.visited_numbers = {1}
    st.session_state.next_target = 2
    st.session_state.message = ""
    st.session_state.message_type = "info"
    st.session_state.show_solution = False
    st.session_state.celebrated = False
    st.session_state.game_ready = True


def render_grid(game: ZipGame, current_pos, path, visited_numbers):
    size = game.size

    # Progress bar line for numbers
    progress_parts = []
    for num in range(1, len(game.numbers) + 1):
        if num in visited_numbers:
            progress_parts.append(f"[{num}✓]")
        else:
            progress_parts.append(f"[{num}&nbsp;]")
    progress = "".join(progress_parts)

    cells_visited = len(set(path))
    total_cells = size * size

    st.markdown(
        f"<div style='text-align:center; font-size:1.05rem;'>"
        f"<strong>Progress:</strong> {progress} &nbsp;&nbsp; "
        f"<strong>Cells:</strong> {cells_visited}/{total_cells} &nbsp;&nbsp; "
        f"<strong>Position:</strong> {current_pos}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Simple HTML table with basic styling
    table_html = [
        "<style>",
        "table.zipgrid {border-collapse: collapse; margin: 1rem auto;}",
        "table.zipgrid td {border: 1px solid #555; width: 60px; height: 60px; text-align: center; font-family: monospace; font-size: 1.25rem;}",
        ".cell-current {background-color: #ffd54f; font-weight: bold;}",
        ".cell-number {background-color: #bbdefb;}",
        ".cell-number-visited {background-color: #81c784;}",
        ".cell-path {background-color: #e0e0e0;}",
        "</style>",
        "<table class='zipgrid'>",
    ]

    for i in range(size):
        table_html.append("<tr>")
        for j in range(size):
            pos = (i, j)
            classes = []
            text = "&nbsp;"

            if pos == current_pos:
                classes.append("cell-current")
                text = "@"
            elif pos in game.numbers.values():
                # find the number at this position
                num = [k for k, v in game.numbers.items() if v == pos][0]
                if num in visited_numbers:
                    classes.append("cell-number-visited")
                    text = f"{num}✓"
                else:
                    classes.append("cell-number")
                    text = str(num)
            elif pos in path:
                classes.append("cell-path")
                text = "*"

            class_attr = f" class='{' '.join(classes)}'" if classes else ""
            table_html.append(f"<td{class_attr}>{text}</td>")
        table_html.append("</tr>")

    table_html.append("</table>")

    st.markdown("\n".join(table_html), unsafe_allow_html=True)


def move_player(dy: int, dx: int):
    """Move the player by (dy, dx) if the move is valid."""
    game: ZipGame = st.session_state.game
    current_pos = st.session_state.current_pos
    path = st.session_state.path
    visited_numbers = st.session_state.visited_numbers
    next_target = st.session_state.next_target

    new_pos = (current_pos[0] + dy, current_pos[1] + dx)

    # Bounds check
    if new_pos[0] < 0 or new_pos[0] >= game.size or new_pos[1] < 0 or new_pos[1] >= game.size:
        st.session_state.message = "That move would leave the board. Try a different direction."
        st.session_state.message_type = "warning"
        return

    # Already visited check
    if new_pos in path:
        st.session_state.message = "You already visited that cell. Each cell can be used only once."
        st.session_state.message_type = "warning"
        return

    # Apply move
    st.session_state.current_pos = new_pos
    path.append(new_pos)

    # Check if landed on a number
    if new_pos in game.numbers.values():
        num = [k for k, v in game.numbers.items() if v == new_pos][0]
        visited_numbers.add(num)
        if num == next_target:
            st.session_state.next_target += 1
            st.session_state.message = f"Nice! You reached number {num} in the correct order."
            st.session_state.message_type = "success"
        else:
            st.session_state.message = (
                f"You stepped on number {num}, but you're currently aiming for {next_target}. "
                "You can keep going, but the sequence is out of order."
            )
            st.session_state.message_type = "warning"
    else:
        st.session_state.message = "Moved to an empty cell. Keep exploring toward the next number."
        st.session_state.message_type = "info"


def step_back():
    """Undo the last move, similar to the CLI 'B' command."""
    game: ZipGame = st.session_state.game
    path = st.session_state.path

    if len(path) <= 1:
        st.session_state.message = "You're already at the starting position. There's nothing to undo."
        st.session_state.message_type = "warning"
        return

    # Remove last position and update current position
    path.pop()
    st.session_state.current_pos = path[-1]

    # Recompute visited_numbers and next_target from the remaining path
    visited_numbers = {1}
    next_target = 2
    for pos in path:
        if pos in game.numbers.values():
            num = [k for k, v in game.numbers.items() if v == pos][0]
            visited_numbers.add(num)
            if num == next_target:
                next_target += 1

    st.session_state.visited_numbers = visited_numbers
    st.session_state.next_target = next_target
    st.session_state.message = "Step undone. You're back to the previous cell."
    st.session_state.message_type = "info"


def reset_player():
    game: ZipGame = st.session_state.game
    start_pos = game.numbers[1]
    st.session_state.current_pos = start_pos
    st.session_state.path = [start_pos]
    st.session_state.visited_numbers = {1}
    st.session_state.next_target = 2
    st.session_state.message = "Puzzle reset. Fresh start from number 1."
    st.session_state.message_type = "info"
    st.session_state.show_solution = False
    st.session_state.celebrated = False


def check_progress():
    game: ZipGame = st.session_state.game
    visited_numbers = st.session_state.visited_numbers
    path = st.session_state.path
    current_pos = st.session_state.current_pos

    all_numbers_visited = len(visited_numbers) == len(game.numbers)
    all_cells_visited = len(path) == game.size * game.size
    on_last_number = current_pos == game.numbers[len(game.numbers)]

    st.subheader("Check Progress")
    st.write(f"Numbers visited: {len(visited_numbers)}/{len(game.numbers)}")
    st.write(f"Cells visited: {len(path)}/{game.size * game.size}")
    st.write(f"On last number: {on_last_number}")

    if all_numbers_visited and all_cells_visited and on_last_number:
        st.success("Perfect! You have a complete valid path.")
    else:
        if not all_numbers_visited:
            missing = sorted(set(range(1, len(game.numbers) + 1)) - visited_numbers)
            st.warning(f"Still need to visit numbers: {missing}")
        if not all_cells_visited:
            st.warning(f"Still need to visit {game.size * game.size - len(path)} more cells")
        if not on_last_number:
            st.warning(f"You must end on number {len(game.numbers)}")


def show_solution():
    game: ZipGame = st.session_state.game
    solution = game.solution_path

    st.subheader("Optimal Solution")
    st.write(f"Length: {len(solution)} moves")
    st.write("Path as list of coordinates:")
    st.code(repr(solution))


def main():
    init_session_state()

    # Global style for main-area buttons (controls, actions, etc.)
    st.markdown(
        """
        <style>
        /* Style primary buttons in the main area */
        div.block-container div[data-testid="stButton"] > button {
            background: linear-gradient(90deg, #1976d2, #42a5f5);
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.35rem 0.9rem;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.25);
        }
        div.block-container div[data-testid="stButton"] > button:hover {
            background: linear-gradient(90deg, #1565c0, #1e88e5);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("RL Zip Game – Web UI")

    st.sidebar.header("Game Settings")

    size = st.sidebar.slider("Grid size", MIN_SIZE, MAX_SIZE, 4)

    # Recommendations from your original code
    recommendations = {
        3: (2, 5, 3),
        4: (2, 6, 4),
        5: (2, 6, 4),
        6: (2, 6, 4),
    }
    min_nums, max_nums, recommended = recommendations.get(size, (2, size, size // 2))

    st.sidebar.markdown(
        f"**Recommendation for {size}x{size}:**  \n"
        f"- Easy: 2–3 numbers  \n"
        f"- Medium: {recommended} numbers  \n"
        f"- Hard: {max_nums - 1}–{max_nums} numbers"
    )

    num_count = st.sidebar.slider(
        "Number of numbers to place",
        min_value=min_nums,
        max_value=max_nums,
        value=recommended,
    )

    if st.sidebar.button("Generate new puzzle"):
        generate_solvable_game(size, num_count)

    if not st.session_state.game_ready or st.session_state.game is None:
        st.info("Choose settings on the left and click 'Generate new puzzle' to start.")
        return

    # Layout containers: board at top, message under it, controls at the bottom
    board_container = st.container()
    message_container = st.container()
    controls_container = st.container()

    # Controls: arrow buttons + actions (apply moves BEFORE rendering grid)
    with controls_container:
        st.markdown("### Controls")
        left_pad, controls_col, right_pad = st.columns([1, 2, 1])
        with controls_col:
            up_row = st.columns(3)
            with up_row[1]:
                if st.button("Up"):
                    move_player(-1, 0)

            mid_row = st.columns(3)
            with mid_row[0]:
                if st.button("Left"):
                    move_player(0, -1)
            with mid_row[1]:
                if st.button("Back"):
                    step_back()
            with mid_row[2]:
                if st.button("Right"):
                    move_player(0, 1)

            down_row = st.columns(3)
            with down_row[1]:
                if st.button("Down"):
                    move_player(1, 0)

            st.write("")
            if st.button("Reset", use_container_width=True):
                reset_player()

    # After processing any button clicks, read the latest state and render the board and messages
    game: ZipGame = st.session_state.game
    current_pos = st.session_state.current_pos
    path = st.session_state.path
    visited_numbers = st.session_state.visited_numbers

    with board_container:
        st.subheader("Game Board")
        render_grid(game, current_pos, path, visited_numbers)

    with message_container:
        msg = st.session_state.message
        if msg:
            mtype = st.session_state.get("message_type", "info")
            if mtype == "error":
                st.error(msg)
            elif mtype == "warning":
                st.warning(msg)
            elif mtype == "success":
                st.success(msg)
            else:
                st.info(msg)

    st.markdown("---")
    col_check, col_solution = st.columns(2)
    with col_check:
        if st.button("Check progress"):
            check_progress()
    with col_solution:
        if st.button("Show optimal solution"):
            st.session_state.show_solution = True

    # Win condition
    if (
        len(path) == game.size * game.size
        and current_pos == game.numbers[len(game.numbers)]
    ):
        # Trigger balloons once per puzzle completion
        if not st.session_state.get("celebrated", False):
            st.balloons()
            st.session_state.celebrated = True

        st.success("Puzzle complete! You visited every cell and ended on the final number.")
        st.write(f"Your path length: **{len(path)}** moves")
        st.write(f"Optimal path length: **{len(game.solution_path)}** moves")
        if path == game.solution_path:
            st.info("Perfect play – you matched the optimal path exactly.")
        elif len(path) == len(game.solution_path):
            st.info("Great job – you found a different path with the same optimal length.")
        else:
            st.info("Nice work reaching the goal. You can try again to see if you can match the optimal path.")

    if st.session_state.show_solution:
        show_solution()


if __name__ == "__main__":
    main()