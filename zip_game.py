import random
import numpy as np
from heapq import heappush, heappop
import os

class ZipGame:
    def __init__(self, size):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.numbers = {}
        self.solution_path = []
        
    def generate_game(self, num_count):
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.numbers = {}
        
        positions = []
        for i in range(self.size):
            for j in range(self.size):
                positions.append((i, j))
        
        random.shuffle(positions)
        
        for i in range(1, num_count + 1):
            pos = positions[i-1]
            self.numbers[i] = pos
            self.grid[pos[0]][pos[1]] = i
            
    def print_grid(self):
        print("\nCurrent Grid:")
        for row in self.grid:
            print([f"{x:2}" if x != 0 else " ." for x in row])
        print()

class QLearningAgent:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount = 0.9
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
    def get_state(self, pos, visited, target_num):
        return (pos, tuple(sorted(visited)), target_num)
        
    def get_reward(self, old_pos, new_pos, visited, target_num, numbers):
        if new_pos in visited:
            return -10
        if new_pos == numbers.get(target_num):
            return +20
        target_pos = numbers.get(target_num)
        if target_pos:
            old_dist = abs(old_pos[0] - target_pos[0]) + abs(old_pos[1] - target_pos[1])
            new_dist = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])
            if new_dist < old_dist:
                return +5
            else:
                return -5
        return -1
        
    def get_valid_moves(self, pos, grid_size, visited=None):
        moves = []
        x, y = pos
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                if visited is None or (new_x, new_y) not in visited:
                    moves.append((new_x, new_y))
        return moves
        
    def find_path(self, game, max_episodes=3000):
        numbers = game.numbers
        total_numbers = len(numbers)
        
        for episode in range(max_episodes):
            current_pos = numbers[1]
            visited = {current_pos}
            target_num = 2
            path = [current_pos]
            
            while target_num <= total_numbers:
                state = self.get_state(current_pos, visited, target_num)
                valid_moves = self.get_valid_moves(current_pos, game.size, visited)
                
                if not valid_moves:
                    break
                
                if random.random() < self.epsilon:
                    next_pos = random.choice(valid_moves)
                else:
                    q_values = []
                    for move in valid_moves:
                        next_state = self.get_state(move, visited, target_num)
                        q_values.append(self.q_table.get(next_state, 0))
                    best_idx = np.argmax(q_values) if q_values else 0
                    next_pos = valid_moves[best_idx]
                
                reward = self.get_reward(current_pos, next_pos, visited, target_num, numbers)
                
                next_state = self.get_state(next_pos, visited, target_num)
                old_q = self.q_table.get(state, 0)
                next_max_q = max([self.q_table.get(self.get_state(move, visited | {next_pos}, target_num), 0) 
                                for move in self.get_valid_moves(next_pos, game.size, visited | {next_pos})], default=0)
                
                self.q_table[state] = old_q + self.learning_rate * (reward + self.discount * next_max_q - old_q)
                
                current_pos = next_pos
                visited.add(current_pos)
                path.append(current_pos)
                
                if current_pos == numbers.get(target_num):
                    target_num += 1
                    if target_num > total_numbers:
                        if len(visited) == game.size * game.size and current_pos == numbers[total_numbers]:
                            print(f"Solution found in episode {episode + 1}!")
                            return path
                
                if len(path) > game.size * game.size + 10:
                    break
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    
        return None

class AStarSolver:
    def __init__(self, game):
        self.game = game
        self.size = game.size
        self.numbers = game.numbers
        
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, pos, visited):
        neighbors = []
        x, y = pos
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if (new_x, new_y) not in visited:
                    neighbors.append((new_x, new_y))
        return neighbors
    
    def estimate_cost(self, pos, visited, target_num):
        if target_num > len(self.numbers):
            remaining_cells = self.size * self.size - len(visited)
            return remaining_cells
        
        target_pos = self.numbers[target_num]
        return self.manhattan_distance(pos, target_pos)
    
    def solve(self, max_states=100000):
        start_pos = self.numbers[1]
        
        heap = [(0, 0, start_pos, frozenset([start_pos]), 2, [start_pos])]
        visited_states = set()
        states_explored = 0
        
        while heap:
            states_explored += 1
            if states_explored > max_states:
                return None
            
            f_score, g_score, pos, visited, target_num, path = heappop(heap)
            
            state_key = (pos, visited, target_num)
            if state_key in visited_states:
                continue
            visited_states.add(state_key)
            
            if target_num > len(self.numbers) and len(visited) == self.size * self.size and pos == self.numbers[len(self.numbers)]:
                return list(path)
            
            for next_pos in self.get_neighbors(pos, visited):
                new_visited = visited | {next_pos}
                new_path = path + [next_pos]
                new_g_score = g_score + 1
                
                new_target_num = target_num
                if target_num <= len(self.numbers) and next_pos == self.numbers[target_num]:
                    new_target_num = target_num + 1
                
                h_score = self.estimate_cost(next_pos, new_visited, new_target_num)
                new_f_score = new_g_score + h_score
                
                heappush(heap, (new_f_score, new_g_score, next_pos, 
                               frozenset(new_visited), new_target_num, new_path))
        
        return None

def print_interactive_grid(game, current_pos, path, visited_numbers):
    size = game.size
    
    progress = ""
    for num in range(1, len(game.numbers) + 1):
        if num in visited_numbers:
            progress += f"[{num}✓]"
        else:
            progress += f"[{num} ]"
    
    cells_visited = len(set(path))
    total_cells = size * size
    print(f"\nProgress: {progress} | Cells: {cells_visited}/{total_cells} | Position: {current_pos}")
    print()
    
    for i in range(size):
        print("+" + "---+" * size)
        
        row_str = "|"
        for j in range(size):
            pos = (i, j)
            
            if pos == current_pos:
                content = " @ "
            elif pos in game.numbers.values():
                num = [k for k, v in game.numbers.items() if v == pos][0]
                if num in visited_numbers:
                    content = f"{num}✓ " if num < 10 else f"{num}✓"
                else:
                    content = f" {num} "
            elif pos in path:
                content = " * "
            else:
                content = "   "
            
            row_str += content + "|"
        
        print(row_str)
    
    print("+" + "---+" * size)
    print()

def play_interactive_game(game):
    print("\n" + "="*60)
    print("Now it's your turn!")
    print("="*60)
    
    start_pos = game.numbers[1]
    current_pos = start_pos
    path = [current_pos]
    visited_numbers = {1}
    next_target = 2
    message = ""
    
    direction_map = {
        'u': (-1, 0),  # Up
        'd': (1, 0),   # Down
        'l': (0, -1),  # Left
        'r': (0, 1)    # Right
    }
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print_interactive_grid(game, current_pos, path, visited_numbers)
        
        if message:
            print(message)
            print()
        
        if len(path) == game.size * game.size and current_pos == game.numbers[len(game.numbers)]:
            print("✨" * 30)
            print("🎉 CONGRATULATIONS! You completed the puzzle! 🎉")
            print("✨" * 30)
            print(f"\nYour path length: {len(path)}")
            print(f"Optimal path length: {len(game.solution_path)}")
            
            if path == game.solution_path:
                print("⭐ PERFECT! You found the exact optimal solution!")
            elif len(path) == len(game.solution_path):
                print("👏 GREAT! Same length as optimal solution!")
            else:
                print("\nGood effort! The optimal solution was:")
                print(game.solution_path)
            break
        
        print("Legend: @ = You | * = Visited | ✓ = Number reached")
        print("Commands: [U]p [D]own [L]eft [R]ight | [B]ack [T]ry again [C]heck [H]elp [Q]uit")
        
        move = input("Your move: ").strip().lower()
        
        if not move:
            message = ""
            continue
        
        if move == 'h':
            print("\n" + "="*60)
            print("HELP - Available Commands:")
            print("="*60)
            print("  U / u  - Move UP")
            print("  D / d  - Move DOWN")
            print("  L / l  - Move LEFT")
            print("  R / r  - Move RIGHT")
            print("  B / b  - Go BACK one step (undo)")
            print("  T / t  - TRY AGAIN (restart from beginning)")
            print("  C / c  - CHECK if you're done")
            print("  H / h  - Show this HELP")
            print("  Q / q  - QUIT and show solution")
            print("\nGoal: Start at 1, visit all numbers in order, end at {}, visit all cells!".format(len(game.numbers)))
            print("="*60)
            input("\nPress Enter to continue...")
            continue
        
        elif move == 'b':
            if len(path) > 1:
                path.pop()
                current_pos = path[-1]
                visited_numbers = {1}
                next_target = 2
                for pos in path:
                    if pos in game.numbers.values():
                        num = [k for k, v in game.numbers.items() if v == pos][0]
                        visited_numbers.add(num)
                        if num == next_target:
                            next_target += 1
                message = "← Moved back one step"
            else:
                message = "❌ Can't go back - you're at the starting position!"
            continue
        
        elif move == 't':
            current_pos = start_pos
            path = [current_pos]
            visited_numbers = {1}
            next_target = 2
            message = "🔄 Restarted! Back to the beginning."
            continue
        
        elif move == 'c':
            all_numbers_visited = len(visited_numbers) == len(game.numbers)
            all_cells_visited = len(path) == game.size * game.size
            on_last_number = current_pos == game.numbers[len(game.numbers)]
            
            print("\n" + "="*60)
            print("✅ Checking your progress...")
            print("="*60)
            print(f"  Numbers visited: {len(visited_numbers)}/{len(game.numbers)} {'✓' if all_numbers_visited else '✗'}")
            print(f"  Cells visited: {len(path)}/{game.size * game.size} {'✓' if all_cells_visited else '✗'}")
            print(f"  On last number: {'✓' if on_last_number else '✗'}")
            
            if all_numbers_visited and all_cells_visited and on_last_number:
                print("\n✨ Perfect! You can complete the puzzle! ✨")
            else:
                if not all_numbers_visited:
                    missing = set(range(1, len(game.numbers) + 1)) - visited_numbers
                    print(f"\n⚠️  Still need to visit numbers: {sorted(missing)}")
                if not all_cells_visited:
                    print(f"⚠️  Still need to visit {game.size * game.size - len(path)} more cells")
                if not on_last_number:
                    print(f"⚠️  Need to end on number {len(game.numbers)}")
            print("="*60)
            input("\nPress Enter to continue...")
            continue
        
        elif move == 'q':
            print("\n" + "="*60)
            print("🏳️  You quit the game")
            print("="*60)
            print(f"\nYour progress: {len(path)}/{game.size * game.size} cells visited")
            print(f"Numbers reached: {sorted(visited_numbers)}")
            print("\n📖 The optimal solution was:")
            print("="*60)
            print(f"Path: {game.solution_path}")
            print(f"Length: {len(game.solution_path)} moves")
            print("\n🗺️  Step-by-step directions:")
            
            for i in range(len(game.solution_path) - 1):
                curr = game.solution_path[i]
                next_pos = game.solution_path[i + 1]
                
                dy = next_pos[0] - curr[0]
                dx = next_pos[1] - curr[1]
                
                if dy == -1:
                    direction = "UP"
                elif dy == 1:
                    direction = "DOWN"
                elif dx == -1:
                    direction = "LEFT"
                elif dx == 1:
                    direction = "RIGHT"
                
                pos_info = f"{curr}"
                if next_pos in game.numbers.values():
                    num = [k for k, v in game.numbers.items() if v == next_pos][0]
                    pos_info = f"{curr} → {direction} → {next_pos} [Number {num}]"
                else:
                    pos_info = f"{curr} → {direction} → {next_pos}"
                
                print(f"  Step {i+1}: {pos_info}")
            
            print("="*60)
            break
        
        elif move in direction_map:
            dy, dx = direction_map[move]
            new_pos = (current_pos[0] + dy, current_pos[1] + dx)
            
            if new_pos[0] < 0 or new_pos[0] >= game.size or new_pos[1] < 0 or new_pos[1] >= game.size:
                message = "❌ Can't move there - out of bounds!"
                continue
            
            if new_pos in path:
                message = "❌ Can't move there - already visited!"
                continue
            
            current_pos = new_pos
            path.append(current_pos)
            
            if current_pos in game.numbers.values():
                num = [k for k, v in game.numbers.items() if v == current_pos][0]
                visited_numbers.add(num)
                if num == next_target:
                    next_target += 1
                    message = f"✨ Reached number {num}!"
                else:
                    message = f"⚠️  Visited number {num} (out of sequence)"
            else:
                message = ""
        
        else:
            message = f"❌ Unknown command: '{move}'. Type 'h' for help."

def main():
    print("=== RL Zip Game ===")
    
    name = input("Enter your name: ")
    print(f"Hello {name}!")
    
    MIN_SIZE = 3
    MAX_SIZE = 5
    
    print(f"\nGrid size must be between {MIN_SIZE}x{MIN_SIZE} and {MAX_SIZE}x{MAX_SIZE}")
    while True:
        try:
            size = int(input(f"Enter grid size ({MIN_SIZE}-{MAX_SIZE}): "))
            if MIN_SIZE <= size <= MAX_SIZE:
                break
            else:
                print(f"❌ Please enter a size between {MIN_SIZE} and {MAX_SIZE}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    recommendations = {
        3: (2, 5, 3),   # (min, max, recommended)
        4: (2, 6, 4),
        5: (2, 6, 4),   # Keep 5x5 conservative
        6: (2, 6, 4)    # 6x6 with 6+ numbers is very hard
    }
    
    min_nums, max_nums, recommended = recommendations.get(size, (2, size, size // 2))
    
    print(f"\n💡 Recommendation for {size}x{size} grid:")
    print(f"   Easy: 2-3 numbers")
    print(f"   Medium: {recommended} numbers (recommended)")
    print(f"   Hard: {max_nums-1}-{max_nums} numbers")
    
    while True:
        try:
            num_count = int(input(f"\nEnter number of numbers to place ({min_nums}-{max_nums}): "))
            if min_nums <= num_count <= max_nums:
                break
            else:
                print(f"❌ Please enter a number between {min_nums} and {max_nums}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    game = ZipGame(size)
    agent = QLearningAgent(size)
    
    print("\nGenerating solvable game...")
    print("(This may take a moment for harder configurations)\n")
    attempts = 0
    max_attempts = 30
    
    while attempts < max_attempts:
        attempts += 1
        game.generate_game(num_count)
        game.print_grid()
        
        print(f"Attempt {attempts}: Training RL agent to find solution...")
        solution = agent.find_path(game, max_episodes=2000)
        
        if solution:
            print("✓ RL agent found solution!")
            print(f"Path length: {len(solution)}")
            game.solution_path = solution
            break
        else:
            print("  RL couldn't solve it, trying A* algorithm...")
            astar = AStarSolver(game)
            solution = astar.solve()
            
            if solution:
                print("✓ A* found solution!")
                print(f"Path length: {len(solution)}")
                game.solution_path = solution
                break
            else:
                print("✗ No solution exists, generating new game...")
    
    if not game.solution_path:
        print(f"\n❌ Could not generate a solvable game after {max_attempts} attempts.")
        print("\n💡 Suggestions:")
        print("   - Try a smaller grid size")
        print("   - Use fewer numbers")
        print(f"   - For {size}x{size}, we recommend {recommended} numbers or less")
        return
    
    play_interactive_game(game)

if __name__ == "__main__":
    main()