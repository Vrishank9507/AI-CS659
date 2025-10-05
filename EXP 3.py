import random

# =============== k-SAT Problem Generator ===============
def generate_k_sat(k, m, n):
    problem = []
    for _ in range(m):
        clause_vars = random.sample(range(1, n+1), k)
        clause = []
        for var in clause_vars:
            if random.choice([True, False]):
                clause.append(var)
            else:
                clause.append(-var)
        problem.append(clause)
    return problem

# =============== Heuristic Functions ===============
def heuristic1(problem, assignment):
    unsat = 0
    for clause in problem:
        if not any((lit > 0 and assignment[abs(lit)-1]) or (lit < 0 and not assignment[abs(lit)-1]) for lit in clause):
            unsat += 1
    return unsat

def heuristic2(problem, assignment):
    score = 0
    for clause in problem:
        if not any((lit > 0 and assignment[abs(lit)-1]) or (lit < 0 and not assignment[abs(lit)-1]) for lit in clause):
            score += len(clause)
    return score

# =============== Hill-Climbing ===============
def hill_climb(problem, heuristic_fn, n, max_steps=1000):
    assignment = [random.choice([True, False]) for _ in range(n)]
    current_score = heuristic_fn(problem, assignment)
    steps = 0
    while steps < max_steps:
        neighbors = []
        for i in range(n):
            neighbor = assignment[:]
            neighbor[i] = not neighbor[i]
            neighbors.append(neighbor)
        best_neighbor = min(neighbors, key=lambda a: heuristic_fn(problem, a))
        best_score = heuristic_fn(problem, best_neighbor)
        if best_score < current_score:
            assignment, current_score = best_neighbor, best_score
        else:
            break
        steps += 1
    return assignment, current_score

# =============== Beam Search ===============
def beam_search(problem, heuristic_fn, n, beam_width=3, max_steps=1000):
    beams
