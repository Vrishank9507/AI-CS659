from collections import deque

# -----------------------------
# Rabbit Leap Problem
# -----------------------------

# Initial and goal states
# '>' = east-bound rabbit, '<' = west-bound rabbit, '_' = empty stone
initial_state = ['>', '>', '>', '_', '<', '<', '<']
goal_state = ['<', '<', '<', '_', '>', '>', '>']


def get_next_states(state):
    """Generate all possible next states from the current state."""
    next_states = []
    n = len(state)
    empty = state.index('_')

    # Try all possible moves
    for i in range(n):
        # Move east-bound rabbit '>'
        if state[i] == '>':
            # Move 1 step right
            if i + 1 < n and state[i + 1] == '_':
                new_state = state.copy()
                new_state[i], new_state[i + 1] = new_state[i + 1], new_state[i]
                next_states.append(new_state)

            # Jump over 1 rabbit to an empty stone
            elif i + 2 < n and state[i + 1] in ['<', '>'] and state[i + 2] == '_':
                new_state = state.copy()
                new_state[i], new_state[i + 2] = new_state[i + 2], new_state[i]
                next_states.append(new_state)

        # Move west-bound rabbit '<'
        elif state[i] == '<':
            # Move 1 step left
            if i - 1 >= 0 and state[i - 1] == '_':
                new_state = state.copy()
                new_state[i], new_state[i - 1] = new_state[i - 1], new_state[i]
                next_states.append(new_state)

            # Jump over 1 rabbit to an empty stone
            elif i - 2 >= 0 and state[i - 1] in ['<', '>'] and state[i - 2] == '_':
                new_state = state.copy()
                new_state[i], new_state[i - 2] = new_state[i - 2], new_state[i]
                next_states.append(new_state)

    return next_states


def bfs(start, goal):
    """Breadth-First Search (finds the optimal solution)."""
    queue = deque([[start]])
    visited = set()
    visited.add(tuple(start))

    while queue:
        path = queue.popleft()
        state = path[-1]

        if state == goal:
            return path

        for next_state in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                queue.append(path + [next_state])
    return None


def dfs(start, goal):
    """Depth-First Search (not guaranteed optimal)."""
    stack = [[start]]
    visited = set()
    visited.add(tuple(start))

    while stack:
        path = stack.pop()
        state = path[-1]

        if state == goal:
            return path

        for next_state in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                stack.append(path + [next_state])
    return None


def print_solution(path, method_name):
    print(f"\n{method_name} Solution ({len(path) - 1} steps):")
    for step, state in enumerate(path):
        print(f"Step {step}: {''.join(state)}")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Initial State:", ''.join(initial_state))
    print("Goal State:", ''.join(goal_state))

    # Solve using BFS
    bfs_path = bfs(initial_state, goal_state)
    print_solution(bfs_path, "BFS")

    # Solve using DFS
    dfs_path = dfs(initial_state, goal_state)
    print_solution(dfs_path, "DFS")

    # Comparison
    print("\n--- Comparison ---")
    print(f"BFS found an optimal solution with {len(bfs_path) - 1} steps.")
    print(f"DFS found a (possibly non-optimal) solution with {len(dfs_path) - 1} steps.")
    print("BFS explores nodes level-by-level → higher space usage but guaranteed optimality.")
    print("DFS explores depth-first → lower space usage but may miss optimal paths.")
