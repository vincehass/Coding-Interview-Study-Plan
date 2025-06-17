"""
=============================================================================
                        DAY 15: GRAPH ALGORITHMS
                           Meta Interview Preparation
                              Week 3 - Day 15
=============================================================================

FOCUS: Cycle detection, topological sort
TIME ALLOCATION: 4 hours
- Theory (1 hour): Graph algorithms, cycle detection, topological sorting
- Problems (3 hours): Advanced graph algorithm problems

TOPICS COVERED:
- Cycle detection in directed/undirected graphs
- Topological sorting
- Graph coloring and bipartite graphs
- Dependency resolution

=============================================================================
"""

from typing import List, Dict, Set
from collections import defaultdict, deque


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
CYCLE DETECTION:

1. UNDIRECTED GRAPHS:
   - Use DFS with parent tracking
   - If we visit a node that's already visited and it's not the parent, cycle exists

2. DIRECTED GRAPHS:
   - Use DFS with three states: WHITE (0), GRAY (1), BLACK (2)
   - If we encounter a GRAY node during DFS, cycle exists

TOPOLOGICAL SORTING:
- Only possible for Directed Acyclic Graphs (DAGs)
- Kahn's Algorithm: Use in-degree counting
- DFS-based: Use finish times

BIPARTITE GRAPHS:
- Can be colored with 2 colors
- Use BFS/DFS with alternating colors
- If we can't color without conflicts, not bipartite
"""


# =============================================================================
# PROBLEM 1: COURSE SCHEDULE (MEDIUM) - 45 MIN
# =============================================================================

def can_finish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    PROBLEM: Course Schedule
    
    There are a total of numCourses courses you have to take, labeled from 0 
    to numCourses - 1. You are given an array prerequisites where 
    prerequisites[i] = [ai, bi] indicates that you must take course bi first 
    if you want to take course ai.
    
    Return true if you can finish all courses. Otherwise, return false.
    
    Example:
    Input: numCourses = 2, prerequisites = [[1,0]]
    Output: true
    
    TIME: O(V + E), SPACE: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0 = WHITE (unvisited), 1 = GRAY (visiting), 2 = BLACK (visited)
    states = [0] * numCourses
    
    def has_cycle(course):
        if states[course] == 1:  # Currently visiting - cycle detected
            return True
        if states[course] == 2:  # Already processed
            return False
        
        states[course] = 1  # Mark as visiting
        
        for next_course in graph[course]:
            if has_cycle(next_course):
                return True
        
        states[course] = 2  # Mark as visited
        return False
    
    # Check each course for cycles
    for course in range(numCourses):
        if states[course] == 0:
            if has_cycle(course):
                return False
    
    return True


# =============================================================================
# PROBLEM 2: COURSE SCHEDULE II (MEDIUM) - 45 MIN
# =============================================================================

def find_order(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    PROBLEM: Course Schedule II
    
    Return the ordering of courses you should take to finish all courses. 
    If there are many valid answers, return any of them. If it is impossible 
    to finish all courses, return an empty array.
    
    Example:
    Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
    Output: [0,1,2,3] or [0,2,1,3]
    
    TIME: O(V + E), SPACE: O(V + E)
    """
    # Build graph and calculate in-degrees
    graph = defaultdict(list)
    indegree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1
    
    # Start with courses that have no prerequisites
    queue = deque([i for i in range(numCourses) if indegree[i] == 0])
    result = []
    
    while queue:
        course = queue.popleft()
        result.append(course)
        
        # Remove this course and update in-degrees
        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)
    
    return result if len(result) == numCourses else []


# =============================================================================
# PROBLEM 3: DETECT CYCLE IN DIRECTED GRAPH (MEDIUM) - 45 MIN
# =============================================================================

def has_cycle_directed(graph: Dict[int, List[int]]) -> bool:
    """
    PROBLEM: Detect Cycle in Directed Graph
    
    Given a directed graph, detect if there's a cycle in it.
    
    TIME: O(V + E), SPACE: O(V)
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    
    def dfs(node):
        if color[node] == GRAY:
            return True  # Back edge found - cycle detected
        if color[node] == BLACK:
            return False  # Already processed
        
        color[node] = GRAY
        
        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True
        
        color[node] = BLACK
        return False
    
    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    
    return False


# =============================================================================
# PROBLEM 4: IS GRAPH BIPARTITE (MEDIUM) - 45 MIN
# =============================================================================

def is_bipartite(graph: List[List[int]]) -> bool:
    """
    PROBLEM: Is Graph Bipartite?
    
    There is an undirected graph with n nodes, where each node is numbered 
    between 0 and n - 1. Return true if and only if it is bipartite.
    
    Example:
    Input: graph = [[1,2,3],[0,2],[1,3],[0,2]]
    Output: false
    
    TIME: O(V + E), SPACE: O(V)
    """
    n = len(graph)
    color = [-1] * n  # -1 means uncolored
    
    def dfs(node, c):
        color[node] = c
        
        for neighbor in graph[node]:
            if color[neighbor] == c:  # Same color as current node
                return False
            if color[neighbor] == -1 and not dfs(neighbor, 1 - c):
                return False
        
        return True
    
    # Check each connected component
    for i in range(n):
        if color[i] == -1:
            if not dfs(i, 0):
                return False
    
    return True


# =============================================================================
# PROBLEM 5: ALIEN DICTIONARY (HARD) - 60 MIN
# =============================================================================

def alien_order(words: List[str]) -> str:
    """
    PROBLEM: Alien Dictionary
    
    There is a new alien language that uses the English alphabet. However, 
    the order among the letters is unknown to you.
    
    You are given a list of strings words from the alien language's dictionary, 
    where the strings in words are sorted lexicographically by the rules of 
    this new language.
    
    Return a string of the unique letters in the new alien language sorted in 
    lexicographically increasing order by the new language's rules.
    
    Example:
    Input: words = ["wrt","wrf","er","ett","rftt"]
    Output: "wertf"
    
    TIME: O(C) where C is total characters, SPACE: O(1) since at most 26 chars
    """
    # Build graph and in-degree count
    graph = defaultdict(set)
    indegree = defaultdict(int)
    
    # Initialize all characters
    for word in words:
        for char in word:
            indegree[char] = 0
    
    # Build edges based on adjacent words
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Check for invalid case: word1 is longer but prefix matches
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""
        
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    indegree[word2[j]] += 1
                break
    
    # Topological sort using Kahn's algorithm
    queue = deque([char for char in indegree if indegree[char] == 0])
    result = []
    
    while queue:
        char = queue.popleft()
        result.append(char)
        
        for neighbor in graph[char]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return "".join(result) if len(result) == len(indegree) else ""


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day15_problems():
    """Test all Day 15 problems"""
    
    print("=" * 60)
    print("         DAY 15: GRAPH ALGORITHMS")
    print("=" * 60)
    
    # Test Course Schedule
    print("\n🧪 Testing Course Schedule")
    can_finish_1 = can_finish(2, [[1,0]])
    print(f"Course Schedule (2, [[1,0]]): {can_finish_1} (Expected: True)")
    
    can_finish_2 = can_finish(2, [[1,0],[0,1]])
    print(f"Course Schedule (2, [[1,0],[0,1]]): {can_finish_2} (Expected: False)")
    
    # Test Course Schedule II
    print("\n🧪 Testing Course Schedule II")
    order_1 = find_order(4, [[1,0],[2,0],[3,1],[3,2]])
    print(f"Course Order: {order_1} (Should be valid topological order)")
    
    order_2 = find_order(2, [[1,0],[0,1]])
    print(f"Course Order (cycle): {order_2} (Expected: [])")
    
    # Test Cycle Detection
    print("\n🧪 Testing Cycle Detection")
    graph_with_cycle = {0: [1], 1: [2], 2: [0]}
    cycle_result_1 = has_cycle_directed(graph_with_cycle)
    print(f"Has Cycle (0->1->2->0): {cycle_result_1} (Expected: True)")
    
    graph_no_cycle = {0: [1], 1: [2], 2: []}
    cycle_result_2 = has_cycle_directed(graph_no_cycle)
    print(f"Has Cycle (0->1->2): {cycle_result_2} (Expected: False)")
    
    # Test Bipartite
    print("\n🧪 Testing Is Bipartite")
    bipartite_1 = is_bipartite([[1,3],[0,2],[1,3],[0,2]])
    print(f"Is Bipartite (square): {bipartite_1} (Expected: True)")
    
    bipartite_2 = is_bipartite([[1,2,3],[0,2],[0,1,3],[0,2]])
    print(f"Is Bipartite (triangle): {bipartite_2} (Expected: False)")
    
    # Test Alien Dictionary
    print("\n🧪 Testing Alien Dictionary")
    alien_1 = alien_order(["wrt","wrf","er","ett","rftt"])
    print(f"Alien Order: '{alien_1}' (Expected: 'wertf')")
    
    alien_2 = alien_order(["z","x"])
    print(f"Alien Order: '{alien_2}' (Expected: 'zx')")
    
    print("\n" + "=" * 60)
    print("           DAY 15 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 15 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review and optimize
    """
    
    print("🚀 Starting Day 15: Graph Algorithms")
    print("\n📚 Theory Topics:")
    print("- Cycle detection (directed vs undirected)")
    print("- Topological sorting (Kahn's algorithm)")
    print("- Graph coloring and bipartite graphs")
    print("- Dependency resolution problems")
    
    print("\n💻 Practice Problems:")
    print("1. Course Schedule (Medium) - 45 min")
    print("2. Course Schedule II (Medium) - 45 min")
    print("3. Detect Cycle in Directed Graph (Medium) - 45 min")
    print("4. Is Graph Bipartite? (Medium) - 45 min")
    print("5. Alien Dictionary (Hard) - 60 min")
    
    print("\n🧪 Running Tests...")
    test_day15_problems()
    
    print("\n✅ Day 15 Complete!")
    print("📈 Next: Day 16 - Union-Find & Advanced Graph")


if __name__ == "__main__":
    main() 