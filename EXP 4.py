import random
import math

# Example city coordinates (x, y) - replace with actual Rajasthan tourist site coordinates or distance matrix
cities = [
    (28.7041, 77.1025), (26.9124, 75.7873), (27.0238, 74.2179), (26.2389, 73.0243), 
    (27.9881, 78.0421), (25.4481, 74.6399), (26.4499, 73.0601), (28.3587, 74.6162),
    (26.4499, 74.6399), (27.2038, 76.4491), (27.7897, 75.0094), (27.6649, 76.2219),
    (26.9124, 75.7873), (29.0578, 75.1576), (27.0238, 74.2179), (28.1305, 75.7993),
    (29.9457, 74.8797), (27.1767, 78.0081), (26.4499, 74.6399), (26.2389, 73.0243)
]

def distance(city1, city2):
    # Euclidean distance
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def total_distance(tour):
    dist = 0
    for i in range(len(tour)):
        dist += distance(cities[tour[i]], cities[tour[(i+1) % len(tour)]])
    return dist

def swap_cities(tour):
    # Generate neighbor by swapping two cities
    new_tour = tour[:]
    i, j = random.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp(-(new_cost - old_cost) / temperature)

def simulated_annealing(cities, initial_temp=10000, cooling_rate=0.995, stopping_temp=0.01):
    current_tour = list(range(len(cities)))
    random.shuffle(current_tour)
    current_cost = total_distance(current_tour)
    temperature = initial_temp
    best_tour = current_tour
    best_cost = current_cost
    
    while temperature > stopping_temp:
        candidate_tour = swap_cities(current_tour)
        candidate_cost = total_distance(candidate_tour)
        if acceptance_probability(current_cost, candidate_cost, temperature) > random.random():
            current_tour = candidate_tour
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_tour = current_tour
                best_cost = current_cost
        temperature *= cooling_rate
    
    return best_tour, best_cost

if __name__ == "__main__":
    best_tour, best_cost = simulated_annealing(cities)
    print("Best tour found:")
    print(best_tour)
    print("Tour cost (distance):", best_cost)
