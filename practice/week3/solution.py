"""
=============================================================================
                        WEEK 3 SOLUTION FILE
                     COMPLETE SOLUTIONS & VARIANTS
                           Meta Interview Preparation
=============================================================================

This file contains complete solutions for all Week 3 practice problems with
multiple approaches, variants, and comprehensive test cases.

TOPICS COVERED:
- Graph Traversal (DFS/BFS)
- Shortest Path Algorithms
- Topological Sort
- Union Find
- Trie
- Advanced Graph Problems

=============================================================================
"""

from collections import defaultdict, deque, Counter
from typing import List, Optional, Dict, Set, Tuple
import heapq


# =============================================================================
# GRAPH TRAVERSAL SOLUTIONS
# =============================================================================

def num_islands(grid: List[List[str]]) -> int:
    """
    PROBLEM: Number of Islands (DFS)
    TIME: O(m * n), SPACE: O(m * n)
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

def num_islands_bfs(grid: List[List[str]]) -> int:
    """
    VARIANT: Number of Islands (BFS)
    TIME: O(m * n), SPACE: O(m * n)
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        grid[start_r][start_c] = '0'
        
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

def max_area_of_island(grid: List[List[int]]) -> int:
    """
    VARIANT: Max Area of Island
    TIME: O(m * n), SPACE: O(m * n)
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    max_area = 0
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
            return 0
        
        grid[r][c] = 0  # Mark as visited
        
        area = 1
        area += dfs(r + 1, c)
        area += dfs(r - 1, c)
        area += dfs(r, c + 1)
        area += dfs(r, c - 1)
        
        return area
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(max_area, dfs(r, c))
    
    return max_area

def surrounded_regions(board: List[List[str]]) -> None:
    """
    VARIANT: Surrounded Regions
    TIME: O(m * n), SPACE: O(m * n)
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != 'O':
            return
        
        board[r][c] = 'E'  # Mark as escaped
        
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    # Mark all 'O's connected to borders as escaped
    for r in range(rows):
        if board[r][0] == 'O':
            dfs(r, 0)
        if board[r][cols - 1] == 'O':
            dfs(r, cols - 1)
    
    for c in range(cols):
        if board[0][c] == 'O':
            dfs(0, c)
        if board[rows - 1][c] == 'O':
            dfs(rows - 1, c)
    
    # Convert remaining 'O's to 'X' and 'E's back to 'O'
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'E':
                board[r][c] = 'O'

# Node class for graph problems
class GraphNode:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: GraphNode) -> GraphNode:
    """
    PROBLEM: Clone Graph
    TIME: O(N + M), SPACE: O(N)
    """
    if not node:
        return None
    
    visited = {}
    
    def dfs(node):
        if node in visited:
            return visited[node]
        
        clone = GraphNode(node.val, [])
        visited[node] = clone
        
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)

def clone_graph_bfs(node: GraphNode) -> GraphNode:
    """
    VARIANT: Clone Graph (BFS)
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

def course_schedule(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    PROBLEM: Course Schedule (Cycle Detection)
    TIME: O(N + P), SPACE: O(N + P)
    """
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
    
    for course in range(numCourses):
        if states[course] == 0:
            if has_cycle(course):
                return False
    
    return True

def course_schedule_ii(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    VARIANT: Course Schedule II (Topological Sort)
    TIME: O(N + P), SPACE: O(N + P)
    """
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
        
        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)
    
    return result if len(result) == numCourses else []

def pacific_atlantic(heights: List[List[int]]) -> List[List[int]]:
    """
    PROBLEM: Pacific Atlantic Water Flow
    TIME: O(m * n), SPACE: O(m * n)
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
        
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            dfs(r + dr, c + dc, visited, heights[r][c])
    
    # Start DFS from borders
    for r in range(rows):
        dfs(r, 0, pacific, heights[r][0])  # Left border (Pacific)
        dfs(r, cols - 1, atlantic, heights[r][cols - 1])  # Right border (Atlantic)
    
    for c in range(cols):
        dfs(0, c, pacific, heights[0][c])  # Top border (Pacific)
        dfs(rows - 1, c, atlantic, heights[rows - 1][c])  # Bottom border (Atlantic)
    
    # Find cells that can reach both oceans
    result = []
    for r, c in pacific:
        if (r, c) in atlantic:
            result.append([r, c])
    
    return result

def word_ladder(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    VARIANT: Word Ladder (BFS)
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
        
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                
                if new_word in wordSet and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))
    
    return 0

# =============================================================================
# SHORTEST PATH ALGORITHMS
# =============================================================================

def shortest_path_binary_matrix(grid: List[List[int]]) -> int:
    """
    PROBLEM: Shortest Path in Binary Matrix
    TIME: O(n²), SPACE: O(n²)
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
            
            if nr == n - 1 and nc == n - 1:
                return length + 1
            
            if (0 <= nr < n and 0 <= nc < n and 
                (nr, nc) not in visited and grid[nr][nc] == 0):
                visited.add((nr, nc))
                queue.append((nr, nc, length + 1))
    
    return -1

def walls_and_gates(rooms: List[List[int]]) -> None:
    """
    VARIANT: Walls and Gates
    TIME: O(m * n), SPACE: O(m * n)
    """
    if not rooms or not rooms[0]:
        return
    
    rows, cols = len(rooms), len(rooms[0])
    queue = deque()
    
    # Find all gates
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c))
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while queue:
        r, c = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                rooms[nr][nc] > rooms[r][c] + 1):
                rooms[nr][nc] = rooms[r][c] + 1
                queue.append((nr, nc))

def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """
    PROBLEM: Network Delay Time (Dijkstra's Algorithm)
    TIME: O(E log V), SPACE: O(V + E)
    """
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
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

def cheapest_flights_k_stops(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """
    VARIANT: Cheapest Flights Within K Stops (Bellman-Ford)
    TIME: O(k * E), SPACE: O(V)
    """
    distances = [float('inf')] * n
    distances[src] = 0
    
    for _ in range(k + 1):
        temp_distances = distances.copy()
        
        for u, v, price in flights:
            if distances[u] != float('inf'):
                temp_distances[v] = min(temp_distances[v], distances[u] + price)
        
        distances = temp_distances
    
    return distances[dst] if distances[dst] != float('inf') else -1

def find_ladder_length(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    VARIANT: Word Ladder II (Bidirectional BFS)
    TIME: O(M * N), SPACE: O(M * N)
    """
    if endWord not in wordList:
        return 0
    
    wordSet = set(wordList)
    
    # Bidirectional BFS
    begin_set = {beginWord}
    end_set = {endWord}
    visited = set()
    length = 1
    
    while begin_set and end_set:
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set
        
        temp_set = set()
        
        for word in begin_set:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]
                    
                    if new_word in end_set:
                        return length + 1
                    
                    if new_word in wordSet and new_word not in visited:
                        visited.add(new_word)
                        temp_set.add(new_word)
        
        begin_set = temp_set
        length += 1
    
    return 0

# =============================================================================
# UNION FIND SOLUTIONS
# =============================================================================

class UnionFind:
    """
    PROBLEM: Union Find / Disjoint Set Union
    TIME: O(α(n)) amortized for all operations
    SPACE: O(n)
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

def number_of_connected_components(n: int, edges: List[List[int]]) -> int:
    """
    PROBLEM: Number of Connected Components using Union Find
    TIME: O(E * α(V)), SPACE: O(V)
    """
    uf = UnionFind(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.components

def accounts_merge(accounts: List[List[str]]) -> List[List[str]]:
    """
    VARIANT: Accounts Merge
    TIME: O(N * K * α(N)), SPACE: O(N * K)
    """
    email_to_name = {}
    email_to_id = {}
    
    # Map emails to account IDs
    for i, account in enumerate(accounts):
        name = account[0]
        for email in account[1:]:
            email_to_name[email] = name
            email_to_id[email] = i
    
    uf = UnionFind(len(accounts))
    
    # Union accounts with common emails
    for account in accounts:
        first_email = account[1] if len(account) > 1 else None
        if first_email:
            for email in account[2:]:
                uf.union(email_to_id[first_email], email_to_id[email])
    
    # Group emails by connected components
    groups = defaultdict(set)
    for email, account_id in email_to_id.items():
        root = uf.find(account_id)
        groups[root].add(email)
    
    result = []
    for emails in groups.values():
        if emails:
            name = email_to_name[next(iter(emails))]
            result.append([name] + sorted(emails))
    
    return result

def redundant_connection(edges: List[List[int]]) -> List[int]:
    """
    VARIANT: Redundant Connection
    TIME: O(N * α(N)), SPACE: O(N)
    """
    uf = UnionFind(len(edges) + 1)
    
    for u, v in edges:
        if uf.connected(u, v):
            return [u, v]
        uf.union(u, v)
    
    return []

# =============================================================================
# TRIE SOLUTIONS
# =============================================================================

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None  # For word search problems

class Trie:
    """
    PROBLEM: Implement Trie (Prefix Tree)
    TIME: O(m) for all operations where m is key length
    SPACE: O(ALPHABET_SIZE * N * M)
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
        node.word = word

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

def word_search_ii(board: List[List[str]], words: List[str]) -> List[str]:
    """
    VARIANT: Word Search II
    TIME: O(M * N * 4^L) where L is max word length
    SPACE: O(W * L) where W is number of words
    """
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    rows, cols = len(board), len(board[0])
    result = set()
    
    def dfs(r, c, node, visited):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if (r, c) in visited:
            return
        
        char = board[r][c]
        if char not in node.children:
            return
        
        node = node.children[char]
        if node.is_end and node.word:
            result.add(node.word)
        
        visited.add((r, c))
        
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            dfs(r + dr, c + dc, node, visited)
        
        visited.remove((r, c))
    
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root, set())
    
    return list(result)

def replace_words(dictionary: List[str], sentence: str) -> str:
    """
    VARIANT: Replace Words
    TIME: O(D + S) where D is dictionary size, S is sentence size
    SPACE: O(D)
    """
    trie = Trie()
    for root in dictionary:
        trie.insert(root)
    
    def find_root(word):
        node = trie.root
        prefix = ""
        
        for char in word:
            if char not in node.children:
                return word
            node = node.children[char]
            prefix += char
            if node.is_end:
                return prefix
        
        return word
    
    words = sentence.split()
    return " ".join(find_root(word) for word in words)

def word_search_i(board: List[List[str]], word: str) -> bool:
    """
    VARIANT: Word Search I
    TIME: O(M * N * 4^L), SPACE: O(L)
    """
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, index, visited):
        if index == len(word):
            return True
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            (r, c) in visited or board[r][c] != word[index]):
            return False
        
        visited.add((r, c))
        
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if dfs(r + dr, c + dc, index + 1, visited):
                return True
        
        visited.remove((r, c))
        return False
    
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0, set()):
                return True
    
    return False

# =============================================================================
# ADVANCED GRAPH PROBLEMS
# =============================================================================

def alien_dictionary(words: List[str]) -> str:
    """
    VARIANT: Alien Dictionary (Topological Sort)
    TIME: O(C) where C is total characters, SPACE: O(1) since at most 26 characters
    """
    # Build graph
    graph = defaultdict(set)
    indegree = defaultdict(int)
    
    # Initialize all characters
    for word in words:
        for char in word:
            indegree[char] = 0
    
    # Build edges based on word pairs
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Check if word1 is longer but prefix matches (invalid case)
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""
        
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    indegree[word2[j]] += 1
                break
    
    # Topological sort
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

def critical_connections(n: int, connections: List[List[int]]) -> List[List[int]]:
    """
    VARIANT: Critical Connections in a Network (Tarjan's Algorithm)
    TIME: O(V + E), SPACE: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = [False] * n
    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    bridges = []
    time = [0]
    
    def bridge_dfs(u):
        visited[u] = True
        disc[u] = low[u] = time[0]
        time[0] += 1
        
        for v in graph[u]:
            if not visited[v]:
                parent[v] = u
                bridge_dfs(v)
                
                low[u] = min(low[u], low[v])
                
                # If low[v] > disc[u], then u-v is a bridge
                if low[v] > disc[u]:
                    bridges.append([u, v])
            
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    
    for i in range(n):
        if not visited[i]:
            bridge_dfs(i)
    
    return bridges

def minimum_spanning_tree_kruskal(n: int, edges: List[List[int]]) -> int:
    """
    VARIANT: Minimum Spanning Tree (Kruskal's Algorithm)
    TIME: O(E log E), SPACE: O(V)
    """
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(n)
    total_weight = 0
    edges_used = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):
            total_weight += weight
            edges_used += 1
            if edges_used == n - 1:
                break
    
    return total_weight

def strongly_connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    VARIANT: Strongly Connected Components (Kosaraju's Algorithm)
    TIME: O(V + E), SPACE: O(V + E)
    """
    visited = set()
    stack = []
    
    def dfs1(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph.get(node, []):
            dfs1(neighbor)
        stack.append(node)
    
    # First DFS to fill stack
    for node in graph:
        dfs1(node)
    
    # Create transpose graph
    transpose = defaultdict(list)
    for node in graph:
        for neighbor in graph[node]:
            transpose[neighbor].append(node)
    
    visited = set()
    components = []
    
    def dfs2(node, component):
        if node in visited:
            return
        visited.add(node)
        component.append(node)
        for neighbor in transpose.get(node, []):
            dfs2(neighbor, component)
    
    # Second DFS on transpose graph
    while stack:
        node = stack.pop()
        if node not in visited:
            component = []
            dfs2(node, component)
            components.append(component)
    
    return components

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_graph_from_edges(n: int, edges: List[List[int]], directed: bool = False) -> Dict[int, List[int]]:
    """Create adjacency list from edge list"""
    graph = defaultdict(list)
    
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    
    return graph

def print_graph(graph: Dict[int, List[int]]) -> None:
    """Print graph in readable format"""
    for node in sorted(graph.keys()):
        print(f"{node}: {graph[node]}")

def create_grid_from_string(grid_str: str) -> List[List[str]]:
    """Create grid from string representation"""
    lines = grid_str.strip().split('\n')
    return [list(line) for line in lines]

# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def test_all_week3_solutions():
    """Comprehensive test suite for all Week 3 solutions"""
    
    print("=" * 80)
    print("                    WEEK 3 COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Test Graph Traversal
    print("\n🧪 TESTING GRAPH TRAVERSAL")
    print("-" * 50)
    
    # Test Number of Islands
    test_cases_islands = [
        ([["1","1","0"],["1","0","0"],["0","0","1"]], 2),
        ([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]], 1),
        ([["0","0","0"],["0","0","0"]], 0),
        ([["1"]], 1)
    ]
    
    for i, (grid, expected) in enumerate(test_cases_islands, 1):
        grid_copy_dfs = [row[:] for row in grid]
        grid_copy_bfs = [row[:] for row in grid]
        
        result_dfs = num_islands(grid_copy_dfs)
        result_bfs = num_islands_bfs(grid_copy_bfs)
        
        print(f"Islands Test {i}: {len(grid)}x{len(grid[0])} grid")
        print(f"  DFS - Expected: {expected}, Got: {result_dfs}")
        print(f"  BFS - Expected: {expected}, Got: {result_bfs}")
        print(f"  ✅ PASS" if result_dfs == expected and result_bfs == expected else f"  ❌ FAIL")
    
    # Test Max Area of Island
    island_area_grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
    max_area_result = max_area_of_island([row[:] for row in island_area_grid])
    print(f"Max Area of Island: {max_area_result} (Expected: 4)")
    
    # Test Clone Graph
    print("\nTesting Clone Graph:")
    node1 = GraphNode(1)
    node2 = GraphNode(2)
    node3 = GraphNode(3)
    node4 = GraphNode(4)
    
    node1.neighbors = [node2, node4]
    node2.neighbors = [node1, node3]
    node3.neighbors = [node2, node4]
    node4.neighbors = [node1, node3]
    
    cloned_dfs = clone_graph(node1)
    cloned_bfs = clone_graph_bfs(node1)
    print(f"Clone Graph DFS: Success (node value: {cloned_dfs.val})")
    print(f"Clone Graph BFS: Success (node value: {cloned_bfs.val})")
    
    # Test Course Schedule
    print("\n🧪 TESTING COURSE SCHEDULE")
    print("-" * 50)
    
    test_cases_courses = [
        (2, [[1,0]], True),
        (2, [[1,0],[0,1]], False),
        (3, [[1,0],[2,1]], True),
        (4, [[1,0],[2,0],[3,1],[3,2]], True)
    ]
    
    for i, (numCourses, prerequisites, expected) in enumerate(test_cases_courses, 1):
        result = course_schedule(numCourses, prerequisites)
        order = course_schedule_ii(numCourses, prerequisites)
        
        print(f"Course Schedule Test {i}: {numCourses} courses, {len(prerequisites)} prereqs")
        print(f"  Can finish: Expected {expected}, Got {result}")
        print(f"  Order: {order if order else 'No valid order'}")
        print(f"  ✅ PASS" if result == expected else f"  ❌ FAIL")
    
    # Test Pacific Atlantic
    print("\n🧪 TESTING PACIFIC ATLANTIC")
    print("-" * 50)
    
    heights_test = [
        [1,2,2,3,5],
        [3,2,3,4,4],
        [2,4,5,3,1],
        [6,7,1,4,5],
        [5,1,1,2,4]
    ]
    
    pacific_atlantic_result = pacific_atlantic(heights_test)
    print(f"Pacific Atlantic cells: {len(pacific_atlantic_result)} cells can reach both oceans")
    print(f"Sample cells: {pacific_atlantic_result[:3] if pacific_atlantic_result else 'None'}")
    
    # Test Shortest Path Algorithms
    print("\n🧪 TESTING SHORTEST PATH ALGORITHMS")
    print("-" * 50)
    
    # Test Shortest Path in Binary Matrix
    binary_matrix_test = [[0,0,0],[1,1,0],[1,1,0]]
    shortest_path_result = shortest_path_binary_matrix(binary_matrix_test)
    print(f"Shortest Path in Binary Matrix: {shortest_path_result} (Expected: 4)")
    
    # Test Network Delay Time
    network_result = network_delay_time([[2,1,1],[2,3,1],[3,4,1]], 4, 2)
    print(f"Network Delay Time: {network_result} (Expected: 2)")
    
    # Test Union Find
    print("\n🧪 TESTING UNION FIND")
    print("-" * 50)
    
    uf = UnionFind(5)
    print(f"Initial components: {uf.components} (Expected: 5)")
    
    uf.union(0, 1)
    print(f"After union(0,1): {uf.components} (Expected: 4)")
    
    uf.union(1, 2)
    print(f"After union(1,2): {uf.components} (Expected: 3)")
    
    print(f"Connected(0,2): {uf.connected(0, 2)} (Expected: True)")
    print(f"Connected(0,3): {uf.connected(0, 3)} (Expected: False)")
    
    # Test Number of Connected Components
    components_result = number_of_connected_components(5, [[0,1],[1,2],[3,4]])
    print(f"Connected Components: {components_result} (Expected: 2)")
    
    # Test Trie
    print("\n🧪 TESTING TRIE")
    print("-" * 50)
    
    trie = Trie()
    trie.insert("apple")
    print(f"Insert 'apple'")
    print(f"Search 'apple': {trie.search('apple')} (Expected: True)")
    print(f"Search 'app': {trie.search('app')} (Expected: False)")
    print(f"Starts with 'app': {trie.starts_with('app')} (Expected: True)")
    
    trie.insert("app")
    print(f"Insert 'app'")
    print(f"Search 'app': {trie.search('app')} (Expected: True)")
    
    # Test Word Search II
    board_test = [
        ["o","a","a","n"],
        ["e","t","a","e"],
        ["i","h","k","r"],
        ["i","f","l","v"]
    ]
    words_test = ["oath","pea","eat","rain"]
    word_search_result = word_search_ii(board_test, words_test)
    print(f"Word Search II: {word_search_result} (Expected: ['eat', 'oath'])")
    
    # Test Advanced Graph Problems
    print("\n🧪 TESTING ADVANCED GRAPH PROBLEMS")
    print("-" * 50)
    
    # Test Word Ladder
    ladder_result = word_ladder("hit", "cog", ["hot","dot","dog","lot","log","cog"])
    print(f"Word Ladder: {ladder_result} (Expected: 5)")
    
    # Test Alien Dictionary
    alien_result = alien_dictionary(["wrt","wrf","er","ett","rftt"])
    print(f"Alien Dictionary: '{alien_result}' (Expected: 'wertf')")
    
    # Test Critical Connections
    critical_result = critical_connections(4, [[0,1],[1,2],[2,0],[1,3]])
    print(f"Critical Connections: {critical_result} (Expected: [[1,3]])")
    
    # Test Complex Scenarios
    print("\n🧪 TESTING COMPLEX SCENARIOS")
    print("-" * 50)
    
    # Test Replace Words
    replace_result = replace_words(["cat","bat","rat"], "the cattle was rattled by the battery")
    print(f"Replace Words: '{replace_result}'")
    print(f"Expected: 'the cat was rat by the bat'")
    
    # Test Accounts Merge
    accounts_test = [
        ["John","johnsmith@mail.com","john_newyork@mail.com"],
        ["John","johnsmith@mail.com","john00@mail.com"],
        ["Mary","mary@mail.com"],
        ["John","johnnybravo@mail.com"]
    ]
    merge_result = accounts_merge(accounts_test)
    print(f"Accounts Merge: {len(merge_result)} accounts after merging")
    
    # Test Cheapest Flights
    flights_result = cheapest_flights_k_stops(3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 1)
    print(f"Cheapest Flights: {flights_result} (Expected: 200)")
    
    print("\n" + "=" * 80)
    print("                    TESTING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_all_week3_solutions() 