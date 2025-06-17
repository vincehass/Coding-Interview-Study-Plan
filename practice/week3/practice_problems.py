"""
=============================================================================
                        WEEK 3 PRACTICE PROBLEMS
                      GRAPHS & ADVANCED STRUCTURES
                           Meta Interview Preparation
=============================================================================

This file contains practice problems for Week 3. Work through these problems
independently to reinforce your learning from the main study materials.

INSTRUCTIONS:
1. Read each problem statement and constraints carefully
2. Understand the examples and expected outputs
3. Write your solution in the designated space
4. Test your solution with the provided test cases
5. Compare with the reference implementation when stuck

=============================================================================
"""

from collections import defaultdict, deque
from typing import List, Optional, Dict, Set
import heapq


# GRAPH TRAVERSAL PRACTICE

def num_islands_practice(grid: List[List[str]]) -> int:
    """
    PROBLEM: Number of Islands
    
    DESCRIPTION:
    Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), 
    return the number of islands.
    
    CONSTRAINTS:
    - m == grid.length
    - n == grid[i].length
    - 1 <= m, n <= 300
    - grid[i][j] is '0' or '1'.
    
    EXAMPLES:
    Example 1:
        Input: grid = [["1","1","0"],["1","0","0"],["0","0","1"]]
        Output: 2
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(m * n)
    
    YOUR SOLUTION:
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        
        grid[r][c] = '0'  # Mark as visited
        
        # Explore 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)
    
    return islands


def clone_graph_practice(node):
    """
    PROBLEM: Clone Graph
    
    DESCRIPTION:
    Given a reference of a node in a connected undirected graph.
    Return a deep copy (clone) of the graph.
    
    CONSTRAINTS:
    - The number of nodes in the graph is in the range [0, 100].
    - 1 <= Node.val <= 100
    - Node.val is unique for each node.
    - There are no repeated edges and no self-loops in the graph.
    
    EXPECTED TIME COMPLEXITY: O(N + M) where N is nodes, M is edges
    EXPECTED SPACE COMPLEXITY: O(N)
    
    YOUR SOLUTION:
    """
    if not node:
        return None
    
    visited = {}
    
    def dfs(node):
        if node in visited:
            return visited[node]
        
        # Create a clone of the current node
        clone = Node(node.val, [])
        visited[node] = clone
        
        # Clone all neighbors
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)


def course_schedule_practice(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    PROBLEM: Course Schedule
    
    DESCRIPTION:
    There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
    You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you 
    must take course bi first if you want to take course ai.
    
    Return true if you can finish all courses. Otherwise, return false.
    
    CONSTRAINTS:
    - 1 <= numCourses <= 10^5
    - 0 <= prerequisites.length <= 5000
    - prerequisites[i].length == 2
    - 0 <= ai, bi < numCourses
    - All the pairs prerequisites[i] are unique.
    
    EXAMPLES:
    Example 1:
        Input: numCourses = 2, prerequisites = [[1,0]]
        Output: true
        Explanation: There are a total of 2 courses to take. 
        To take course 1 you should have finished course 0. So it is possible.
    
    Example 2:
        Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
        Output: false
        Explanation: There are a total of 2 courses to take. 
        To take course 1 you should have finished course 0, and to take course 0 you should 
        also have finished course 1. So it is impossible.
    
    EXPECTED TIME COMPLEXITY: O(N + P)
    EXPECTED SPACE COMPLEXITY: O(N + P)
    
    YOUR SOLUTION:
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0 = unvisited, 1 = visiting, 2 = visited
    states = [0] * numCourses
    
    def has_cycle(course):
        if states[course] == 1:  # Currently visiting - cycle detected
            return True
        if states[course] == 2:  # Already visited
            return False
        
        states[course] = 1  # Mark as visiting
        
        for next_course in graph[course]:
            if has_cycle(next_course):
                return True
        
        states[course] = 2  # Mark as visited
        return False
    
    # Check for cycles in each component
    for course in range(numCourses):
        if states[course] == 0:
            if has_cycle(course):
                return False
    
    return True


def pacific_atlantic_practice(heights: List[List[int]]) -> List[List[int]]:
    """
    PROBLEM: Pacific Atlantic Water Flow
    
    DESCRIPTION:
    There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean.
    The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches 
    the island's right and bottom edges.
    
    The island is partitioned into a grid of square cells. You are given an m x n integer 
    matrix heights where heights[r][c] represents the height above sea level of the cell at 
    coordinate (r, c).
    
    The island receives a lot of rain, and the rain water can flow to neighboring cells directly 
    north, south, east, and west if the neighboring cell's height is less than or equal to the 
    current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.
    
    Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain 
    water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.
    
    CONSTRAINTS:
    - m == heights.length
    - n == heights[r].length
    - 1 <= m, n <= 200
    - 0 <= heights[r][c] <= 10^5
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(m * n)
    
    YOUR SOLUTION:
    """
    if not heights or not heights[0]:
        return []
    
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    
    def dfs(r, c, visited, prev_height):
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if heights[r][c] < prev_height:
            return
        
        visited.add((r, c))
        
        # Explore 4 directions
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            dfs(r + dr, c + dc, visited, heights[r][c])
    
    # Start DFS from Pacific borders (top and left)
    for r in range(rows):
        dfs(r, 0, pacific, heights[r][0])  # Left border
        dfs(r, cols - 1, atlantic, heights[r][cols - 1])  # Right border
    
    for c in range(cols):
        dfs(0, c, pacific, heights[0][c])  # Top border
        dfs(rows - 1, c, atlantic, heights[rows - 1][c])  # Bottom border
    
    # Find cells that can reach both oceans
    result = []
    for r, c in pacific:
        if (r, c) in atlantic:
            result.append([r, c])
    
    return result


# SHORTEST PATH ALGORITHMS

def shortest_path_binary_matrix_practice(grid: List[List[int]]) -> int:
    """
    PROBLEM: Shortest Path in Binary Matrix
    
    DESCRIPTION:
    Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. 
    If there is no clear path, return -1.
    
    A clear path in a binary matrix is a path from the top-left cell (0, 0) to the bottom-right 
    cell (n - 1, n - 1) such that:
    - All the visited cells of the path are 0.
    - All the adjacent cells of the path are 8-directionally connected.
    
    The length of a clear path is the number of visited cells of this path.
    
    CONSTRAINTS:
    - n == grid.length
    - n == grid[i].length
    - 1 <= n <= 100
    - grid[i][j] is 0 or 1
    
    EXPECTED TIME COMPLEXITY: O(n²)
    EXPECTED SPACE COMPLEXITY: O(n²)
    
    YOUR SOLUTION:
    """
    n = len(grid)
    
    if grid[0][0] != 0 or grid[n-1][n-1] != 0:
        return -1
    
    if n == 1:
        return 1
    
    queue = deque([(0, 0, 1)])  # (row, col, path_length)
    visited = set([(0, 0)])
    
    # 8 directions
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        r, c, length = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (nr == n - 1 and nc == n - 1):
                return length + 1
            
            if (0 <= nr < n and 0 <= nc < n and 
                (nr, nc) not in visited and grid[nr][nc] == 0):
                visited.add((nr, nc))
                queue.append((nr, nc, length + 1))
    
    return -1


def network_delay_time_practice(times: List[List[int]], n: int, k: int) -> int:
    """
    PROBLEM: Network Delay Time
    
    DESCRIPTION:
    You are given a network of n nodes, labeled from 1 to n. You are also given times, 
    a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the 
    source node, vi is the target node, and wi is the time it takes for a signal to 
    travel from source to target.
    
    We will send a signal from a given node k. Return the time it takes for all n nodes 
    to receive the signal. If it is impossible for all n nodes to receive the signal, return -1.
    
    CONSTRAINTS:
    - 1 <= k <= n <= 100
    - 1 <= times.length <= 6000
    - times[i].length == 3
    - 1 <= ui, vi <= n
    - ui != vi
    - 0 <= wi <= 100
    - All the pairs (ui, vi) are unique.
    
    EXPECTED TIME COMPLEXITY: O(E log V) using Dijkstra
    EXPECTED SPACE COMPLEXITY: O(V + E)
    
    YOUR SOLUTION:
    """
    # Build graph
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    # Dijkstra's algorithm
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0
    
    heap = [(0, k)]  # (distance, node)
    
    while heap:
        curr_dist, node = heapq.heappop(heap)
        
        if curr_dist > distances[node]:
            continue
        
        for neighbor, weight in graph[node]:
            new_dist = curr_dist + weight
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    max_time = max(distances.values())
    return max_time if max_time != float('inf') else -1


# ADVANCED DATA STRUCTURES

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie_Practice:
    """
    PROBLEM: Implement Trie (Prefix Tree)
    
    DESCRIPTION:
    A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently 
    store and retrieve keys in a dataset of strings. There are various applications of this 
    data structure, such as autocomplete and spellchecker.
    
    Implement the Trie class:
    - Trie() Initializes the trie object.
    - void insert(String word) Inserts the string word into the trie.
    - boolean search(String word) Returns true if the string word is in the trie.
    - boolean startsWith(String prefix) Returns true if there is a previously inserted string 
      word that has the prefix prefix.
    
    CONSTRAINTS:
    - 1 <= word.length, prefix.length <= 2000
    - word and prefix consist only of lowercase English letters.
    - At most 3 * 10^4 calls in total will be made to insert, search, and startsWith.
    
    EXPECTED TIME COMPLEXITY: O(m) for all operations where m is key length
    EXPECTED SPACE COMPLEXITY: O(ALPHABET_SIZE * N * M) where N is number of keys
    
    YOUR SOLUTION:
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


class UnionFind_Practice:
    """
    PROBLEM: Union Find / Disjoint Set Union
    
    DESCRIPTION:
    Implement a Union-Find data structure with path compression and union by rank.
    
    Operations:
    - find(x): Find the root of the set containing x
    - union(x, y): Union the sets containing x and y
    - connected(x, y): Check if x and y are in the same set
    
    EXPECTED TIME COMPLEXITY: O(α(n)) amortized for all operations
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False  # Already connected
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


def number_of_connected_components_practice(n: int, edges: List[List[int]]) -> int:
    """
    PROBLEM: Number of Connected Components in an Undirected Graph
    
    DESCRIPTION:
    You have a graph of n nodes labeled from 0 to n - 1. You are given an integer n and 
    a list of edges where edges[i] = [ai, bi] indicates that there is an undirected edge 
    between nodes ai and bi in the graph.
    
    Return the number of connected components in the graph.
    
    CONSTRAINTS:
    - 1 <= n <= 2000
    - 1 <= edges.length <= 5000
    - edges[i].length == 2
    - 0 <= ai <= bi < n
    - ai != bi
    - There are no repeated edges.
    
    EXPECTED TIME COMPLEXITY: O(E * α(V)) using Union-Find
    EXPECTED SPACE COMPLEXITY: O(V)
    
    YOUR SOLUTION:
    """
    uf = UnionFind_Practice(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.components


# Helper classes for graph problems
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


# TEST FUNCTIONS

def test_week3_practice():
    """Test your solutions with comprehensive test cases"""
    
    print("=== TESTING WEEK 3 PRACTICE SOLUTIONS ===\n")
    
    # Test Number of Islands
    print("1. Number of Islands:")
    test_cases = [
        ([["1","1","0"],["1","0","0"],["0","0","1"]], 2),
        ([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]], 1),
        ([["0","0","0"],["0","0","0"]], 0)
    ]
    for grid, expected in test_cases:
        grid_copy = [row[:] for row in grid]  # Make copy since function modifies grid
        result = num_islands_practice(grid_copy)
        print(f"   Grid: {grid}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Course Schedule
    print("2. Course Schedule:")
    test_cases = [
        (2, [[1,0]], True),
        (2, [[1,0],[0,1]], False),
        (3, [[1,0],[2,1]], True)
    ]
    for numCourses, prerequisites, expected in test_cases:
        result = course_schedule_practice(numCourses, prerequisites)
        print(f"   Courses: {numCourses}, Prerequisites: {prerequisites}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Trie
    print("3. Trie Implementation:")
    trie = Trie_Practice()
    trie.insert("apple")
    print(f"   Insert 'apple'")
    print(f"   Search 'apple': {trie.search('apple')} (Expected: True)")
    print(f"   Search 'app': {trie.search('app')} (Expected: False)")
    print(f"   StartsWith 'app': {trie.startsWith('app')} (Expected: True)")
    trie.insert("app")
    print(f"   Insert 'app'")
    print(f"   Search 'app': {trie.search('app')} (Expected: True)")
    print()
    
    # Test Union Find
    print("4. Union Find:")
    uf = UnionFind_Practice(5)
    print(f"   Initial components: {uf.components} (Expected: 5)")
    uf.union(0, 1)
    print(f"   After union(0,1): {uf.components} (Expected: 4)")
    uf.union(1, 2)
    print(f"   After union(1,2): {uf.components} (Expected: 3)")
    print(f"   Connected(0,2): {uf.connected(0, 2)} (Expected: True)")
    print(f"   Connected(0,3): {uf.connected(0, 3)} (Expected: False)")
    print()
    
    print("Continue implementing and testing other problems...")


if __name__ == "__main__":
    print("Week 3 Practice Problems")
    print("========================")
    print("Topics: Graphs, Advanced Data Structures")
    print("- Graph Traversal (DFS/BFS)")
    print("- Shortest Path Algorithms")
    print("- Topological Sort")
    print("- Union Find")
    print("- Trie")
    print()
    
    # Uncomment to run tests
    # test_week3_practice() 