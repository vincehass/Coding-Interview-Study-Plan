"""
=============================================================================
                        DAY 13: GRAPH REPRESENTATION & BFS
                           Meta Interview Preparation
                              Week 3 - Day 13
=============================================================================

FOCUS: Graph basics, breadth-first search
TIME ALLOCATION: 4 hours
- Theory (1 hour): Graph representations, BFS algorithm
- Problems (3 hours): Graph traversal problems

TOPICS COVERED:
- Graph representations: adjacency list, matrix
- BFS algorithm and applications
- Level-by-level processing
- Connected components

=============================================================================
"""

from collections import deque, defaultdict
from typing import List, Dict, Set


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
GRAPH REPRESENTATIONS:

1. ADJACENCY LIST:
   - Space: O(V + E)
   - Good for sparse graphs
   - Easy to iterate over neighbors
   
   Example: {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}

2. ADJACENCY MATRIX:
   - Space: O(V²)
   - Good for dense graphs
   - O(1) edge lookup
   
   Example: [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 0],
             [0, 1, 0, 0]]

BFS ALGORITHM:
- Time: O(V + E)
- Space: O(V)
- Uses queue (FIFO)
- Explores level by level
- Finds shortest path in unweighted graphs
"""


# =============================================================================
# PROBLEM 1: NUMBER OF ISLANDS (MEDIUM) - 45 MIN
# =============================================================================

def num_islands(grid: List[List[str]]) -> int:
    """
    PROBLEM: Number of Islands
    
    Given an m x n 2D binary grid which represents a map of '1's (land) 
    and '0's (water), return the number of islands.
    
    An island is surrounded by water and is formed by connecting adjacent 
    lands horizontally or vertically.
    
    Example:
    Input: grid = [
      ["1","1","1","1","0"],
      ["1","1","0","1","0"],
      ["1","1","0","0","0"],
      ["0","0","0","0","0"]
    ]
    Output: 1
    
    TIME: O(m * n), SPACE: O(min(m, n))
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        grid[start_r][start_c] = '0'  # Mark as visited
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == '1'):
                    grid[nr][nc] = '0'
                    queue.append((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                bfs(r, c)
    
    return islands


# =============================================================================
# PROBLEM 2: BINARY TREE LEVEL ORDER TRAVERSAL (MEDIUM) - 30 MIN
# =============================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root: TreeNode) -> List[List[int]]:
    """
    PROBLEM: Binary Tree Level Order Traversal
    
    Given the root of a binary tree, return the level order traversal 
    of its nodes' values (i.e., from left to right, level by level).
    
    Example:
    Input: root = [3,9,20,null,null,15,7]
    Output: [[3],[9,20],[15,7]]
    
    TIME: O(n), SPACE: O(w) where w is max width
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result


# =============================================================================
# PROBLEM 3: WORD LADDER (HARD) - 75 MIN
# =============================================================================

def ladder_length(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    PROBLEM: Word Ladder
    
    A transformation sequence from word beginWord to word endWord using a 
    dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk
    such that:
    - Every adjacent pair of words differs by a single letter.
    - Every si for 1 <= i <= k is in wordList. Note that beginWord does not 
      need to be in wordList.
    - sk == endWord
    
    Return the length of the shortest transformation sequence from beginWord 
    to endWord, or 0 if no such sequence exists.
    
    Example:
    Input: beginWord = "hit", endWord = "cog", 
           wordList = ["hot","dot","dog","lot","log","cog"]
    Output: 5
    Explanation: "hit" -> "hot" -> "dot" -> "dog" -> "cog"
    
    TIME: O(M * N) where M is length of words, N is total words
    SPACE: O(M * N)
    """
    if endWord not in wordList:
        return 0
    
    wordSet = set(wordList)
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        word, length = queue.popleft()
        
        if word == endWord:
            return length
        
        # Try all possible single character changes
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                
                if new_word in wordSet and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))
    
    return 0


# =============================================================================
# PROBLEM 4: ROTTING ORANGES (MEDIUM) - 45 MIN
# =============================================================================

def oranges_rotting(grid: List[List[int]]) -> int:
    """
    PROBLEM: Rotting Oranges
    
    You are given an m x n grid where each cell can have one of three values:
    - 0 representing an empty cell,
    - 1 representing a fresh orange, or
    - 2 representing a rotten orange.
    
    Every minute, any fresh orange that is 4-directionally adjacent to a 
    rotten orange becomes rotten.
    
    Return the minimum number of minutes that must elapse until no cell 
    has a fresh orange. If this is impossible, return -1.
    
    Example:
    Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
    Output: 4
    
    TIME: O(m * n), SPACE: O(m * n)
    """
    if not grid or not grid[0]:
        return -1
    
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    # Find all initially rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, time)
            elif grid[r][c] == 1:
                fresh_count += 1
    
    if fresh_count == 0:
        return 0
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    max_time = 0
    
    while queue:
        r, c, time = queue.popleft()
        max_time = max(max_time, time)
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                grid[nr][nc] == 1):
                grid[nr][nc] = 2
                fresh_count -= 1
                queue.append((nr, nc, time + 1))
    
    return max_time if fresh_count == 0 else -1


# =============================================================================
# PROBLEM 5: CLONE GRAPH (MEDIUM) - 45 MIN
# =============================================================================

class GraphNode:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: GraphNode) -> GraphNode:
    """
    PROBLEM: Clone Graph
    
    Given a reference of a node in a connected undirected graph.
    Return a deep copy (clone) of the graph.
    
    Each node in the graph contains a value (int) and a list (List[Node]) 
    of its neighbors.
    
    TIME: O(N + M), SPACE: O(N)
    """
    if not node:
        return None
    
    visited = {node: GraphNode(node.val, [])}
    queue = deque([node])
    
    while queue:
        curr = queue.popleft()
        
        for neighbor in curr.neighbors:
            if neighbor not in visited:
                visited[neighbor] = GraphNode(neighbor.val, [])
                queue.append(neighbor)
            
            visited[curr].neighbors.append(visited[neighbor])
    
    return visited[node]


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day13_problems():
    """Test all Day 13 problems"""
    
    print("=" * 60)
    print("         DAY 13: GRAPH REPRESENTATION & BFS")
    print("=" * 60)
    
    # Test Number of Islands
    print("\n🧪 Testing Number of Islands")
    test_grid1 = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
    ]
    result1 = num_islands([row[:] for row in test_grid1])  # Deep copy
    print(f"Grid 1 Islands: {result1} (Expected: 1)")
    
    test_grid2 = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    result2 = num_islands([row[:] for row in test_grid2])
    print(f"Grid 2 Islands: {result2} (Expected: 3)")
    
    # Test Level Order Traversal
    print("\n🧪 Testing Level Order Traversal")
    # Create tree: [3,9,20,null,null,15,7]
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)
    
    level_result = level_order(root)
    print(f"Level Order: {level_result} (Expected: [[3],[9,20],[15,7]])")
    
    # Test Word Ladder
    print("\n🧪 Testing Word Ladder")
    ladder_result = ladder_length("hit", "cog", ["hot","dot","dog","lot","log","cog"])
    print(f"Word Ladder Length: {ladder_result} (Expected: 5)")
    
    # Test Rotting Oranges
    print("\n🧪 Testing Rotting Oranges")
    orange_grid = [[2,1,1],[1,1,0],[0,1,1]]
    orange_result = oranges_rotting([row[:] for row in orange_grid])
    print(f"Rotting Time: {orange_result} (Expected: 4)")
    
    print("\n" + "=" * 60)
    print("           DAY 13 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 13 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review and optimize
    """
    
    print("🚀 Starting Day 13: Graph Representation & BFS")
    print("\n📚 Theory Topics:")
    print("- Graph representations (adjacency list vs matrix)")
    print("- BFS algorithm and implementation")
    print("- Level-by-level processing")
    print("- Connected components")
    
    print("\n💻 Practice Problems:")
    print("1. Number of Islands (Medium) - 45 min")
    print("2. Binary Tree Level Order Traversal (Medium) - 30 min") 
    print("3. Word Ladder (Hard) - 75 min")
    print("4. Rotting Oranges (Medium) - 45 min")
    print("5. Clone Graph (Medium) - 45 min")
    
    print("\n🧪 Running Tests...")
    test_day13_problems()
    
    print("\n✅ Day 13 Complete!")
    print("📈 Next: Day 14 - Graph DFS & Backtracking")


if __name__ == "__main__":
    main() 