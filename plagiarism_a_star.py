"""
Plagiarism detection via sentence-level alignment using A* search.

File: plagiarism_a_star.py

Features:
- Simple sentence tokenizer (regex-based)
- Normalization (lowercasing, punctuation removal option)
- Levenshtein edit distance (character-level and token-level options)
- A* search on alignment state space (states are (i,j) indices into sentences lists)
- Backtracking to produce optimal alignment
- Simple heuristic (admissible: 0). You can replace with a better heuristic.
- Plagiarism detection: returns aligned sentence pairs with normalized edit distances below a threshold
- Test cases included as functions at the bottom (matching lab Test Case 1-4)

Notes:
- This implementation doesn't use external libraries so it runs in plain Python 3.x.
- For large documents this will be expensive. Consider chunking, using paragraph-level prefiltering,
  or an approximate / beam-search approach for scaling.

Usage example (from command line or import):
>>> from plagiarism_a_star import detect_plagiarism_from_texts
>>> doc1 = "This is sentence one. This is sentence two."
>>> doc2 = "This is sentence one. This differs slightly."
>>> matches = detect_plagiarism_from_texts(doc1, doc2, threshold=0.3)
>>> for a,b,score in matches:
...     print(a, "<=>", b, score)

"""

from typing import List, Tuple, Dict, Optional
import heapq
import re
import math

# ------------------------- Text preprocessing -------------------------

def sentence_tokenize(text: str) -> List[str]:
    """A simple sentence tokenizer based on punctuation.
    Not perfect but sufficient for lab purposes.
    Splits on . ? ! and keeps abbreviations naive handling.
    """
    # Normalize newlines
    text = text.replace('\n', ' ')
    # ensure space after punctuation to split reliably
    # keep the punctuation to allow minimal heuristics if needed
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty parts
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def normalize_sentence(s: str, remove_punct: bool = True) -> str:
    s = s.lower().strip()
    if remove_punct:
        # remove most punctuation
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s)
    return s

# ------------------------- Edit distance -------------------------

def levenshtein_chars(a: str, b: str) -> int:
    """Character-level Levenshtein distance (classic DP). Returns integer distance."""
    # quick returns
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # use two-row DP for memory efficiency
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,         # deletion
                         cur[j - 1] + 1,      # insertion
                         prev[j - 1] + cost)  # substitution
        prev, cur = cur, prev
    return prev[lb]


def levenshtein_tokens(a: str, b: str) -> int:
    """Token-level Levenshtein distance (split on whitespace)."""
    ta = a.split()
    tb = b.split()
    la, lb = len(ta), len(tb)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        for j in range(1, lb + 1):
            cost = 0 if ta[i - 1] == tb[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[lb]


def normalized_distance(a: str, b: str, mode: str = 'char') -> float:
    """Return normalized edit distance between 0 and 1.
    Uses either character-level or token-level distance.
    Normalization: distance / max(len(a), len(b)). If both empty -> 0.
    """
    if mode == 'token':
        d = levenshtein_tokens(a, b)
        denom = max(len(a.split()), len(b.split()), 1)
    else:
        d = levenshtein_chars(a, b)
        denom = max(len(a), len(b), 1)
    return d / denom

# ------------------------- A* search on alignments -------------------------

class AlignmentState:
    """Represents a state in the alignment: indices i (doc1) and j (doc2)
    We will not store sentences inside each state to keep keys small.
    """
    __slots__ = ('i', 'j')
    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j
    def key(self):
        return (self.i, self.j)
    def __repr__(self):
        return f"State(i={self.i}, j={self.j})"


def a_star_align(sent1: List[str], sent2: List[str],
                 cost_fn, heuristic_fn=None,
                 skip_cost_fn=None) -> Tuple[float, List[Tuple[Optional[int], Optional[int]]]]:
    """
    A* search to find minimum-cost alignment between two sequences of sentences.

    Returns (cost, path) where path is a list of pairs (i_index_or_None, j_index_or_None)
    None indicates a gap (skipped sentence).

    - cost_fn(i,j) gives cost to align sent1[i] with sent2[j]
    - skip_cost_fn(i, j_side) gives cost to skip sentence at index i from side j_side
      where j_side is 'left' or 'right' (left => skip in sent1, right => skip in sent2)
    - heuristic_fn(i,j) returns estimated remaining cost from (i,j) to goal.
      Must be admissible. If None, heuristic=0.
    """
    n1 = len(sent1)
    n2 = len(sent2)
    start = (0, 0)
    goal = (n1, n2)

    if heuristic_fn is None:
        heuristic_fn = lambda i, j: 0.0

    # Priority queue: (f, g, (i,j), parent)
    open_heap = []
    # store best g found so far for (i,j)
    g_score: Dict[Tuple[int,int], float] = {}
    parent: Dict[Tuple[int,int], Tuple[Tuple[int,int], Tuple[Optional[int], Optional[int]]]] = {}
    # parent[state] = (prev_state, action) where action is (i_idx_or_None, j_idx_or_None)

    start_h = heuristic_fn(0,0)
    heapq.heappush(open_heap, (start_h, 0.0, start))
    g_score[start] = 0.0
    parent[start] = (None, (None, None))

    while open_heap:
        f, g, (i, j) = heapq.heappop(open_heap)
        # if this node has an outdated g in heap, skip
        if g_score.get((i,j), math.inf) != g:
            continue
        # goal check
        if (i, j) == goal:
            # reconstruct path
            path = []
            cur = (i,j)
            while cur is not None:
                prev, action = parent[cur]
                if action != (None, None):
                    path.append(action)
                cur = prev
            path.reverse()
            return g, path

        # generate successors
        # 1) align i with j (advance both)
        if i < n1 and j < n2:
            cost = cost_fn(i, j)
            neigh = (i+1, j+1)
            tentative_g = g + cost
            if tentative_g < g_score.get(neigh, math.inf):
                g_score[neigh] = tentative_g
                parent[neigh] = ((i,j), (i, j))
                h = heuristic_fn(i+1, j+1)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, neigh))
        # 2) skip sentence in sent1 (gap in sent2)
        if i < n1:
            if skip_cost_fn:
                cost = skip_cost_fn(i, 'left')
            else:
                # default skip cost: token count of the skipped sentence (admissible?)
                cost = max(len(sent1[i].split()), 1)
            neigh = (i+1, j)
            tentative_g = g + cost
            if tentative_g < g_score.get(neigh, math.inf):
                g_score[neigh] = tentative_g
                parent[neigh] = ((i,j), (i, None))
                h = heuristic_fn(i+1, j)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, neigh))
        # 3) skip sentence in sent2 (gap in sent1)
        if j < n2:
            if skip_cost_fn:
                cost = skip_cost_fn(j, 'right')
            else:
                cost = max(len(sent2[j].split()), 1)
            neigh = (i, j+1)
            tentative_g = g + cost
            if tentative_g < g_score.get(neigh, math.inf):
                g_score[neigh] = tentative_g
                parent[neigh] = ((i,j), (None, j))
                h = heuristic_fn(i, j+1)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, neigh))

    # if open set exhausted without reaching goal
    return math.inf, []

# ------------------------- High-level pipeline -------------------------

def build_cost_functions(sent1_norm: List[str], sent2_norm: List[str],
                         mode: str = 'char'):
    """Return cost_fn(i,j) and skip_cost_fn(i, side) closures.
    cost_fn returns raw edit distance (not normalized) to keep numeric scale consistent for g.
    skip_cost_fn returns cost to skip a sentence (we use sentence length as proxy).
    """
    def cost_fn(i: int, j: int) -> float:
        a = sent1_norm[i]
        b = sent2_norm[j]
        # use absolute (unnormalized) distances so costs are additive
        if mode == 'token':
            return levenshtein_tokens(a, b)
        else:
            return levenshtein_chars(a, b)

    def skip_cost_fn(idx: int, side: str) -> float:
        # side 'left' means skip in sent1 (i consumed), 'right' skip in sent2
        if side == 'left':
            a = sent1_norm[idx]
            # cost is token count (prefer token cost since skipping a sentence is like deleting tokens)
            return max(len(a.split()), 1)
        else:
            b = sent2_norm[idx]
            return max(len(b.split()), 1)

    return cost_fn, skip_cost_fn


def detect_plagiarism_from_texts(text1: str, text2: str,
                                 norm_remove_punct: bool = True,
                                 mode: str = 'char',
                                 threshold: float = 0.25) -> List[Tuple[str, str, float]]:
    """
    Tokenize and normalize two texts, run A* alignment, and return list of aligned sentence pairs
    whose normalized edit distance <= threshold (i.e., likely plagiarism).

    Returns: list of (orig_sent1, orig_sent2, normalized_distance)
    """
    sents1 = sentence_tokenize(text1)
    sents2 = sentence_tokenize(text2)
    if not sents1 or not sents2:
        return []
    sents1_norm = [normalize_sentence(s, remove_punct=norm_remove_punct) for s in sents1]
    sents2_norm = [normalize_sentence(s, remove_punct=norm_remove_punct) for s in sents2]

    cost_fn, skip_cost_fn = build_cost_functions(sents1_norm, sents2_norm, mode=mode)

    # heuristic: zero (admissible). For modest documents this is fine.
    heuristic = lambda i, j: 0.0

    total_cost, path = a_star_align(sents1_norm, sents2_norm, cost_fn, heuristic, skip_cost_fn)

    # reconstruct alignments with original sentences
    align_pairs: List[Tuple[str, str, float]] = []
    # path is list of actions: (i_idx_or_None, j_idx_or_None) in forward order
    for action in path:
        i_idx, j_idx = action
        if i_idx is not None and j_idx is not None:
            # aligned pair is original sentence indices i_idx and j_idx
            a_orig = sents1[i_idx]
            b_orig = sents2[j_idx]
            a_norm = sents1_norm[i_idx]
            b_norm = sents2_norm[j_idx]
            score = normalized_distance(a_norm, b_norm, mode=('token' if mode == 'token' else 'char'))
            if score <= threshold:
                align_pairs.append((a_orig, b_orig, score))
    return align_pairs

# ------------------------- Utilities and small experiments -------------------------

def pretty_print_matches(matches: List[Tuple[str, str, float]]):
    if not matches:
        print("No suspicious matching sentence pairs found.")
        return
    print("Suspicious matches (sentence1 <=> sentence2) [normalized edit distance]:")
    for a, b, score in matches:
        print(f"- [{score:.3f}] \n  1) {a}\n  2) {b}\n")

# ------------------------- Test cases (from lab) -------------------------

def test_case_identical():
    doc = """
    This is the first sentence. Here comes the second sentence. Finally the third sentence.
    """
    matches = detect_plagiarism_from_texts(doc, doc, threshold=0.0)
    print("Test Case 1: Identical Documents")
    pretty_print_matches(matches)


def test_case_slight_modifications():
    doc1 = """
    The quick brown fox jumps over the lazy dog. Machine learning is fun. Plagiarism is bad.
    """
    doc2 = """
    The quick brown fox jumped over the lazy dog. Machine learning can be fun. Plagiarism is unethical.
    """
    matches = detect_plagiarism_from_texts(doc1, doc2, threshold=0.4)
    print("Test Case 2: Slightly Modified Document")
    pretty_print_matches(matches)


def test_case_completely_different():
    doc1 = """
    Quantum mechanics studies tiny particles. The Moon orbits the Earth.
    """
    doc2 = """
    Cooking recipes are about heat and taste. Football is popular in many countries.
    """
    matches = detect_plagiarism_from_texts(doc1, doc2, threshold=0.2)
    print("Test Case 3: Completely Different Documents")
    pretty_print_matches(matches)


def test_case_partial_overlap():
    doc1 = """
    Introduction to AI. Search algorithms are fundamental in AI. The A* search is optimal.
    Some unrelated paragraph about gardening.
    """
    doc2 = """
    Search algorithms are fundamental in artificial intelligence. The A* search algorithm is optimal.
    An unrelated note on cooking.
    """
    matches = detect_plagiarism_from_texts(doc1, doc2, threshold=0.35)
    print("Test Case 4: Partial Overlap")
    pretty_print_matches(matches)


if __name__ == '__main__':
    print("Running built-in tests...\n")
    test_case_identical()
    print("---\n")
    test_case_slight_modifications()
    print("---\n")
    test_case_completely_different()
    print("---\n")
    test_case_partial_overlap()
    print("Done.")
