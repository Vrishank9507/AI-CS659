import heapq
import string
import re

# --------------------------
# Text Preprocessing
# --------------------------
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into sentences
    sentences = re.split(r'\.\s+', text.strip())
    return [s for s in sentences if s]

# --------------------------
# Levenshtein Distance
# --------------------------
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                   dp[i][j-1],    # insertion
                                   dp[i-1][j-1])  # substitution
    return dp[m][n]

# --------------------------
# A* Search Algorithm
# --------------------------
class State:
    def __init__(self, i, j, cost, path):
        self.i = i  # index in doc1
        self.j = j  # index in doc2
        self.cost = cost  # g(n)
        self.path = path  # alignment so far
    
    def __lt__(self, other):
        return self.cost < other.cost

def heuristic(doc1, doc2, i, j):
    # Estimate: assume remaining sentences align perfectly (minimum cost 0)
    remaining_cost = 0
    while i < len(doc1) and j < len(doc2):
        remaining_cost += 0
        i += 1
        j += 1
    return remaining_cost

def a_star(doc1, doc2):
    start = State(0, 0, 0, [])
    heap = [(start.cost + heuristic(doc1, doc2, 0, 0), start)]
    visited = set()
    
    while heap:
        _, current = heapq.heappop(heap)
        if (current.i, current.j) in visited:
            continue
        visited.add((current.i, current.j))
        
        # Goal check
        if current.i == len(doc1) and current.j == len(doc2):
            return current.path, current.cost
        
        # Transitions
        # 1. Align current sentences
        if current.i < len(doc1) and current.j < len(doc2):
            cost = levenshtein(doc1[current.i], doc2[current.j])
            new_state = State(current.i+1, current.j+1, current.cost+cost,
                              current.path + [(doc1[current.i], doc2[current.j], cost)])
            heapq.heappush(heap, (new_state.cost + heuristic(doc1, doc2, new_state.i, new_state.j), new_state))
        
        # 2. Skip sentence in doc1
        if current.i < len(doc1):
            new_state = State(current.i+1, current.j, current.cost + 1,
                              current.path + [(doc1[current.i], None, 1)])
            heapq.heappush(heap, (new_state.cost + heuristic(doc1, doc2, new_state.i, new_state.j), new_state))
        
        # 3. Skip sentence in doc2
        if current.j < len(doc2):
            new_state = State(current.i, current.j+1, current.cost + 1,
                              current.path + [(None, doc2[current.j], 1)])
            heapq.heappush(heap, (new_state.cost + heuristic(doc1, doc2, new_state.i, new_state.j), new_state))
    
    return [], float('inf')

# --------------------------
# Plagiarism Detection
# --------------------------
def detect_plagiarism(doc1, doc2):
    doc1_sentences = preprocess(doc1)
    doc2_sentences = preprocess(doc2)
    
    alignment, total_cost = a_star(doc1_sentences, doc2_sentences)
    
    print(f"\nTotal Alignment Cost: {total_cost}\n")
    print("Aligned Sentences (Potential Plagiarism if cost is low):")
    for s1, s2, cost in alignment:
        if s1 and s2:
            print(f"Doc1: {s1}")
            print(f"Doc2: {s2}")
            print(f"Edit Distance: {cost}\n")

# --------------------------
# Test Cases
# --------------------------
doc_a = "This is a test document. It contains several sentences. We are testing plagiarism detection."
doc_b = "This is a test document. It has several sentences. We are testing plagiarism detection."

doc_c = "Completely different content. No overlap here."

print("Test Case 1: Identical Documents")
detect_plagiarism(doc_a, doc_a)

print("\nTest Case 2: Slightly Modified Document")
detect_plagiarism(doc_a, doc_b)

print("\nTest Case 3: Completely Different Documents")
detect_plagiarism(doc_a, doc_c)

print("\nTest Case 4: Partial Overlap")
doc_d = "This is a test document. Additional content here. We are testing plagiarism detection."
detect_plagiarism(doc_a, doc_d)
