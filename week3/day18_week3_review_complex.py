"""
=============================================================================
                        DAY 18: WEEK 3 REVIEW & COMPLEX PROBLEMS
                           Meta Interview Preparation
                              Week 3 - Day 18
=============================================================================

FOCUS: Integration of graph concepts, complex problems
TIME ALLOCATION: 4 hours
- Review (1 hour): Week 3 concepts recap
- Complex Problems (3 hours): Multi-concept integration problems

TOPICS COVERED:
- Graph traversal mastery
- Advanced algorithm combinations
- System design considerations
- Interview simulation

=============================================================================
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque
import heapq


# =============================================================================
# WEEK 3 REVIEW SECTION (1 HOUR)
# =============================================================================

"""
WEEK 3 CONCEPTS MASTERY CHECKLIST:

✓ GRAPH REPRESENTATIONS:
  - Adjacency list vs matrix tradeoffs
  - Space complexity considerations

✓ BFS APPLICATIONS:
  - Level-order traversal
  - Shortest path in unweighted graphs
  - Multi-source BFS

✓ DFS & BACKTRACKING:
  - Path exploration
  - State space search
  - Pruning strategies

✓ GRAPH ALGORITHMS:
  - Cycle detection (directed/undirected)
  - Topological sorting
  - Bipartite checking

✓ UNION-FIND:
  - Path compression
  - Union by rank
  - Dynamic connectivity

✓ TRIES:
  - Prefix operations
  - String matching optimization
  - Space-time tradeoffs
"""


# =============================================================================
# PROBLEM 1: CRITICAL CONNECTIONS IN A NETWORK (HARD) - 75 MIN
# =============================================================================

def critical_connections(n: int, connections: List[List[int]]) -> List[List[int]]:
    """
    PROBLEM: Critical Connections in a Network (Tarjan's Bridge Algorithm)
    
    There are n servers numbered from 0 to n-1 connected by undirected 
    server-to-server connections forming a network where connections[i] = [a, b] 
    represents a connection between servers a and b.
    
    A critical connection is a connection that, if removed, will make some 
    servers unable to reach some other server.
    
    Return all critical connections in the network in any order.
    
    Example:
    Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
    Output: [[1,3]]
    
    TIME: O(V + E), SPACE: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = [False] * n
    disc = [0] * n  # Discovery times
    low = [0] * n   # Low-link values
    parent = [-1] * n
    bridges = []
    time = [0]  # Use list to make it mutable in nested function
    
    def bridge_dfs(u):
        visited[u] = True
        disc[u] = low[u] = time[0]
        time[0] += 1
        
        for v in graph[u]:
            if not visited[v]:
                parent[v] = u
                bridge_dfs(v)
                
                # Check if subtree rooted at v has back edge to ancestors of u
                low[u] = min(low[u], low[v])
                
                # If low[v] > disc[u], then u-v is a bridge
                if low[v] > disc[u]:
                    bridges.append([u, v])
            
            elif v != parent[u]:  # Back edge
                low[u] = min(low[u], disc[v])
    
    for i in range(n):
        if not visited[i]:
            bridge_dfs(i)
    
    return bridges


# =============================================================================
# PROBLEM 2: MINIMUM HEIGHT TREES (MEDIUM) - 60 MIN
# =============================================================================

def find_min_height_trees(n: int, edges: List[List[int]]) -> List[int]:
    """
    PROBLEM: Minimum Height Trees
    
    A tree is an undirected graph in which any two vertices are connected by 
    exactly one path. Given such a tree of n nodes labelled from 0 to n - 1, 
    and an array of n - 1 edges where edges[i] = [ai, bi] indicates that there 
    is an undirected edge between the two nodes ai and bi in the tree.
    
    You can choose any node of the tree as the root. When you pick a node x as 
    the root, the resulting tree has height h. Among all possible rooted trees, 
    those with minimum height h are called minimum height trees (MHTs).
    
    Return a list of all MHTs' root labels.
    
    Example:
    Input: n = 4, edges = [[1,0],[1,2],[1,3]]
    Output: [1]
    
    TIME: O(V), SPACE: O(V)
    """
    if n == 1:
        return [0]
    
    # Build adjacency list
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    # Start with leaves (nodes with degree 1)
    leaves = deque([i for i in range(n) if len(graph[i]) == 1])
    remaining = n
    
    # Remove leaves layer by layer until 1 or 2 nodes remain
    while remaining > 2:
        leaf_count = len(leaves)
        remaining -= leaf_count
        
        for _ in range(leaf_count):
            leaf = leaves.popleft()
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            
            if len(graph[neighbor]) == 1:
                leaves.append(neighbor)
    
    return list(leaves)


# =============================================================================
# PROBLEM 3: WORD LADDER II (HARD) - 90 MIN
# =============================================================================

def find_ladders(beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    """
    PROBLEM: Word Ladder II
    
    A transformation sequence from word beginWord to word endWord using a 
    dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk
    such that every adjacent pair of words differs by exactly one letter.
    
    Given two words, beginWord and endWord, and a dictionary wordList, return 
    all the shortest transformation sequences from beginWord to endWord.
    
    Example:
    Input: beginWord = "hit", endWord = "cog", 
           wordList = ["hot","dot","dog","lot","log","cog"]
    Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
    
    TIME: O(M * N * 26 + paths), SPACE: O(M * N)
    """
    if endWord not in wordList:
        return []
    
    wordSet = set(wordList)
    if beginWord in wordSet:
        wordSet.remove(beginWord)
    
    # BFS to find shortest path length and build parent map
    queue = deque([beginWord])
    parents = defaultdict(list)
    level = {beginWord}
    found = False
    
    while queue and not found:
        # Remove words from current level to avoid cycles
        wordSet -= level
        next_level = set()
        
        for _ in range(len(queue)):
            word = queue.popleft()
            
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    
                    next_word = word[:i] + c + word[i+1:]
                    
                    if next_word in wordSet:
                        if next_word == endWord:
                            found = True
                        
                        if next_word not in next_level:
                            next_level.add(next_word)
                            queue.append(next_word)
                        
                        parents[next_word].append(word)
        
        level = next_level
    
    # DFS to build all paths
    def build_paths(word):
        if word == beginWord:
            return [[beginWord]]
        
        paths = []
        for parent in parents[word]:
            for path in build_paths(parent):
                paths.append(path + [word])
        
        return paths
    
    return build_paths(endWord) if found else []


# =============================================================================
# PROBLEM 4: ALIEN DICTIONARY ADVANCED (HARD) - 75 MIN
# =============================================================================

def alien_order_advanced(words: List[str]) -> str:
    """
    PROBLEM: Alien Dictionary (Advanced with error handling)
    
    Enhanced version with comprehensive error checking and cycle detection.
    
    TIME: O(C), SPACE: O(1) where C is total characters
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
            return ""  # Invalid input
        
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
    
    # Check for cycles
    if len(result) != len(indegree):
        return ""  # Cycle detected
    
    return "".join(result)


# =============================================================================
# PROBLEM 5: GRAPH VALID TREE (MEDIUM) - 45 MIN
# =============================================================================

def valid_tree(n: int, edges: List[List[int]]) -> bool:
    """
    PROBLEM: Graph Valid Tree
    
    You have a graph of n nodes labeled from 0 to n - 1. You are given an 
    integer n and a list of edges where edges[i] = [ai, bi] indicates that 
    there is an undirected edge between nodes ai and bi in the graph.
    
    Return true if the edges of the given graph make up a valid tree, and false otherwise.
    
    A valid tree must be:
    1. Connected (all nodes reachable)
    2. Acyclic (no cycles)
    3. Has exactly n-1 edges
    
    TIME: O(V + E), SPACE: O(V + E)
    """
    # A tree with n nodes must have exactly n-1 edges
    if len(edges) != n - 1:
        return False
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            if neighbor in visited:
                return False  # Cycle detected
            if not dfs(neighbor, node):
                return False
        
        return True
    
    # Check if graph is connected and acyclic
    return dfs(0, -1) and len(visited) == n


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day18_problems():
    """Test all Day 18 complex problems"""
    
    print("=" * 60)
    print("         DAY 18: WEEK 3 REVIEW & COMPLEX PROBLEMS")
    print("=" * 60)
    
    # Test Critical Connections
    print("\n🧪 Testing Critical Connections")
    bridges = critical_connections(4, [[0,1],[1,2],[2,0],[1,3]])
    print(f"Critical Connections: {bridges} (Expected: [[1,3]])")
    
    # Test Minimum Height Trees
    print("\n🧪 Testing Minimum Height Trees")
    mht = find_min_height_trees(4, [[1,0],[1,2],[1,3]])
    print(f"Min Height Trees: {mht} (Expected: [1])")
    
    mht2 = find_min_height_trees(6, [[3,0],[3,1],[3,2],[3,4],[5,4]])
    print(f"Min Height Trees: {mht2} (Expected: [3,4])")
    
    # Test Word Ladder II
    print("\n🧪 Testing Word Ladder II")
    ladders = find_ladders("hit", "cog", ["hot","dot","dog","lot","log","cog"])
    print(f"Word Ladders: Found {len(ladders)} paths")
    if ladders:
        print(f"First path: {ladders[0]}")
    
    # Test Alien Dictionary Advanced
    print("\n🧪 Testing Alien Dictionary Advanced")
    alien_advanced = alien_order_advanced(["wrt","wrf","er","ett","rftt"])
    print(f"Alien Order Advanced: '{alien_advanced}' (Expected: 'wertf')")
    
    # Test invalid case
    alien_invalid = alien_order_advanced(["abc","ab"])
    print(f"Alien Order Invalid: '{alien_invalid}' (Expected: '')")
    
    # Test Graph Valid Tree
    print("\n🧪 Testing Graph Valid Tree")
    tree1 = valid_tree(5, [[0,1],[0,2],[0,3],[1,4]])
    print(f"Valid Tree 1: {tree1} (Expected: True)")
    
    tree2 = valid_tree(5, [[0,1],[1,2],[2,3],[1,3],[1,4]])
    print(f"Valid Tree 2: {tree2} (Expected: False)")
    
    print("\n" + "=" * 60)
    print("           DAY 18 TESTING COMPLETED")
    print("           WEEK 3 REVIEW FINISHED")
    print("=" * 60)


# =============================================================================
# WEEK 3 SUMMARY & NEXT STEPS
# =============================================================================

def week3_summary():
    """Summary of Week 3 achievements and Week 4 preview"""
    
    print("\n" + "=" * 70)
    print("                    WEEK 3 COMPLETION SUMMARY")
    print("=" * 70)
    
    print("\n🎯 WEEK 3 ACHIEVEMENTS:")
    print("✅ Graph Representation & BFS (Day 13)")
    print("✅ Graph DFS & Backtracking (Day 14)")
    print("✅ Graph Algorithms & Topological Sort (Day 15)")
    print("✅ Union-Find & Advanced Graph (Day 16)")
    print("✅ Tries & String Algorithms (Day 17)")
    print("✅ Complex Problems Integration (Day 18)")
    
    print("\n📊 SKILLS MASTERED:")
    print("• Graph traversal algorithms (BFS/DFS)")
    print("• Backtracking and state space search")
    print("• Cycle detection and topological sorting")
    print("• Union-Find with optimizations")
    print("• Trie operations and string matching")
    print("• Complex algorithm integration")
    
    print("\n🚀 WEEK 4 PREVIEW - DYNAMIC PROGRAMMING:")
    print("• Day 19: 1D Dynamic Programming")
    print("• Day 20: 2D Dynamic Programming")
    print("• Day 21: Advanced DP Patterns")
    print("• Day 22: DP Optimization Techniques")
    print("• Day 23: DP on Trees and Graphs")
    print("• Day 24: Final Review & Mock Interviews")
    
    print("\n💡 INTERVIEW READINESS:")
    print("• Graph problems: STRONG")
    print("• Algorithm design: STRONG")
    print("• Complex problem solving: DEVELOPING")
    print("• Time/space optimization: GOOD")
    
    print("\n" + "=" * 70)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 18 Practice Routine:
    1. Review Week 3 concepts (1 hour)
    2. Solve complex integration problems (3 hours)
    3. Test solutions and analyze
    4. Prepare for Week 4
    """
    
    print("🚀 Starting Day 18: Week 3 Review & Complex Problems")
    print("\n📚 Review Topics:")
    print("- Graph traversal mastery")
    print("- Advanced algorithm combinations")
    print("- System design considerations")
    print("- Interview simulation techniques")
    
    print("\n💻 Complex Problems:")
    print("1. Critical Connections (Hard) - 75 min")
    print("2. Minimum Height Trees (Medium) - 60 min")
    print("3. Word Ladder II (Hard) - 90 min")
    print("4. Alien Dictionary Advanced (Hard) - 75 min")
    print("5. Graph Valid Tree (Medium) - 45 min")
    
    print("\n🧪 Running Tests...")
    test_day18_problems()
    
    print("\n✅ Day 18 Complete!")
    week3_summary()
    print("📈 Next: Week 4 - Dynamic Programming Mastery")


if __name__ == "__main__":
    main() 