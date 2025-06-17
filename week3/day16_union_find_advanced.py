"""
=============================================================================
                        DAY 16: UNION-FIND & ADVANCED GRAPH
                           Meta Interview Preparation
                              Week 3 - Day 16
=============================================================================

FOCUS: Disjoint sets, connectivity problems
TIME ALLOCATION: 4 hours
- Theory (1 hour): Union-Find data structure, path compression
- Problems (3 hours): Connectivity and disjoint set problems

TOPICS COVERED:
- Union-Find data structure
- Path compression and union by rank
- Applications in connectivity problems
- Dynamic connectivity

=============================================================================
"""

from typing import List, Dict, Set
from collections import defaultdict


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
UNION-FIND (DISJOINT SET UNION):

Operations:
1. FIND(x): Find the root/representative of set containing x
2. UNION(x, y): Merge sets containing x and y

Optimizations:
1. PATH COMPRESSION: Make nodes point directly to root during find
2. UNION BY RANK: Attach smaller tree under root of larger tree

Time Complexity:
- Without optimizations: O(n) per operation
- With optimizations: O(α(n)) amortized per operation
  where α is inverse Ackermann function (practically constant)

Applications:
- Connected components
- Cycle detection in undirected graphs
- Minimum spanning tree (Kruskal's algorithm)
- Dynamic connectivity
"""


# =============================================================================
# UNION-FIND IMPLEMENTATION
# =============================================================================

class UnionFind:
    """
    Union-Find data structure with path compression and union by rank
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x: int) -> int:
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank"""
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
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
        """Check if two elements are in the same set"""
        return self.find(x) == self.find(y)


# =============================================================================
# PROBLEM 1: NUMBER OF CONNECTED COMPONENTS (MEDIUM) - 45 MIN
# =============================================================================

def count_components(n: int, edges: List[List[int]]) -> int:
    """
    PROBLEM: Number of Connected Components in Undirected Graph
    
    You have a graph of n nodes labeled from 0 to n - 1. You are given an 
    integer n and a list of edges where edges[i] = [ai, bi] indicates that 
    there is an undirected edge between nodes ai and bi in the graph.
    
    Return the number of connected components in the graph.
    
    Example:
    Input: n = 5, edges = [[0,1],[1,2],[3,4]]
    Output: 2
    
    TIME: O(E * α(V)), SPACE: O(V)
    """
    uf = UnionFind(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.components


# =============================================================================
# PROBLEM 2: ACCOUNTS MERGE (MEDIUM) - 60 MIN
# =============================================================================

def accounts_merge(accounts: List[List[str]]) -> List[List[str]]:
    """
    PROBLEM: Accounts Merge
    
    Given a list of accounts where each element accounts[i] is a list of 
    strings, where the first element accounts[i][0] is a name, and the rest 
    of the elements are emails representing emails of the account.
    
    Example:
    Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],
                       ["John","johnsmith@mail.com","john00@mail.com"],
                       ["Mary","mary@mail.com"],
                       ["John","johnnybravo@mail.com"]]
    Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],
             ["Mary","mary@mail.com"],
             ["John","johnnybravo@mail.com"]]
    
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
        if len(account) > 1:
            first_email = account[1]
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


# =============================================================================
# PROBLEM 3: REDUNDANT CONNECTION (MEDIUM) - 45 MIN
# =============================================================================

def find_redundant_connection(edges: List[List[int]]) -> List[int]:
    """
    PROBLEM: Redundant Connection
    
    In this problem, a tree is an undirected graph that is connected and has 
    no cycles. You are given a graph that started as a tree with n nodes 
    labeled from 1 to n, with one additional edge added.
    
    Return an edge that can be removed so that the resulting graph is a tree.
    
    Example:
    Input: edges = [[1,2],[1,3],[2,3]]
    Output: [2,3]
    
    TIME: O(N * α(N)), SPACE: O(N)
    """
    uf = UnionFind(len(edges) + 1)
    
    for u, v in edges:
        if uf.connected(u, v):
            return [u, v]
        uf.union(u, v)
    
    return []


# =============================================================================
# PROBLEM 4: MOST STONES REMOVED (MEDIUM) - 60 MIN
# =============================================================================

def remove_stones(stones: List[List[int]]) -> int:
    """
    PROBLEM: Most Stones Removed with Same Row or Column
    
    On a 2D plane, we place n stones at some integer coordinate points. 
    Each coordinate point may have at most one stone.
    
    A stone can be removed if it shares either the same row or the same 
    column as another stone that has not been removed.
    
    Given an array stones of length n where stones[i] = [xi, yi] represents 
    the location of the ith stone, return the largest possible number of 
    stones that can be removed.
    
    Example:
    Input: stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
    Output: 5
    
    TIME: O(N * α(N)), SPACE: O(N)
    """
    n = len(stones)
    uf = UnionFind(n)
    
    # Group stones by row and column
    row_map = defaultdict(list)
    col_map = defaultdict(list)
    
    for i, (x, y) in enumerate(stones):
        row_map[x].append(i)
        col_map[y].append(i)
    
    # Union stones in same row
    for stone_indices in row_map.values():
        for i in range(1, len(stone_indices)):
            uf.union(stone_indices[0], stone_indices[i])
    
    # Union stones in same column
    for stone_indices in col_map.values():
        for i in range(1, len(stone_indices)):
            uf.union(stone_indices[0], stone_indices[i])
    
    # Maximum stones we can remove = total stones - number of components
    return n - uf.components


# =============================================================================
# PROBLEM 5: SATISFIABILITY OF EQUALITY EQUATIONS (MEDIUM) - 45 MIN
# =============================================================================

def equations_possible(equations: List[str]) -> bool:
    """
    PROBLEM: Satisfiability of Equality Equations
    
    You are given an array of strings equations that represent relationships 
    between variables where each string equations[i] is of length 4 and takes 
    one of two different forms: "xi==yi" or "xi!=yi".
    
    Return true if it is possible to assign integers to variable names so as 
    to satisfy all the given equations, or false otherwise.
    
    Example:
    Input: equations = ["a==b","b!=a"]
    Output: false
    
    TIME: O(N), SPACE: O(1)
    """
    uf = UnionFind(26)  # 26 letters
    
    # Process equality equations first
    for eq in equations:
        if eq[1] == '=':  # "a==b"
            x, y = ord(eq[0]) - ord('a'), ord(eq[3]) - ord('a')
            uf.union(x, y)
    
    # Check inequality equations
    for eq in equations:
        if eq[1] == '!':  # "a!=b"
            x, y = ord(eq[0]) - ord('a'), ord(eq[3]) - ord('a')
            if uf.connected(x, y):
                return False
    
    return True


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day16_problems():
    """Test all Day 16 problems"""
    
    print("=" * 60)
    print("         DAY 16: UNION-FIND & ADVANCED GRAPH")
    print("=" * 60)
    
    # Test Union-Find basic operations
    print("\n🧪 Testing Union-Find Operations")
    uf = UnionFind(5)
    print(f"Initial components: {uf.components} (Expected: 5)")
    
    uf.union(0, 1)
    print(f"After union(0,1): {uf.components} (Expected: 4)")
    
    uf.union(1, 2)
    print(f"After union(1,2): {uf.components} (Expected: 3)")
    
    print(f"Connected(0,2): {uf.connected(0, 2)} (Expected: True)")
    print(f"Connected(0,3): {uf.connected(0, 3)} (Expected: False)")
    
    # Test Connected Components
    print("\n🧪 Testing Connected Components")
    components = count_components(5, [[0,1],[1,2],[3,4]])
    print(f"Connected Components: {components} (Expected: 2)")
    
    # Test Accounts Merge
    print("\n🧪 Testing Accounts Merge")
    accounts = [
        ["John","johnsmith@mail.com","john_newyork@mail.com"],
        ["John","johnsmith@mail.com","john00@mail.com"],
        ["Mary","mary@mail.com"],
        ["John","johnnybravo@mail.com"]
    ]
    merged = accounts_merge(accounts)
    print(f"Accounts Merge: {len(merged)} accounts after merging")
    for account in merged:
        print(f"  {account[0]}: {len(account)-1} emails")
    
    # Test Redundant Connection
    print("\n🧪 Testing Redundant Connection")
    redundant = find_redundant_connection([[1,2],[1,3],[2,3]])
    print(f"Redundant Connection: {redundant} (Expected: [2,3])")
    
    # Test Most Stones Removed
    print("\n🧪 Testing Most Stones Removed")
    stones_removed = remove_stones([[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]])
    print(f"Most Stones Removed: {stones_removed} (Expected: 5)")
    
    # Test Equality Equations
    print("\n🧪 Testing Equality Equations")
    eq_possible_1 = equations_possible(["a==b","b!=a"])
    print(f"Equations ['a==b','b!=a']: {eq_possible_1} (Expected: False)")
    
    eq_possible_2 = equations_possible(["b==a","a==b"])
    print(f"Equations ['b==a','a==b']: {eq_possible_2} (Expected: True)")
    
    print("\n" + "=" * 60)
    print("           DAY 16 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 16 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review and optimize
    """
    
    print("🚀 Starting Day 16: Union-Find & Advanced Graph")
    print("\n📚 Theory Topics:")
    print("- Union-Find data structure")
    print("- Path compression optimization")
    print("- Union by rank optimization")
    print("- Applications in connectivity problems")
    
    print("\n💻 Practice Problems:")
    print("1. Number of Connected Components (Medium) - 45 min")
    print("2. Accounts Merge (Medium) - 60 min")
    print("3. Redundant Connection (Medium) - 45 min")
    print("4. Most Stones Removed (Medium) - 60 min")
    print("5. Satisfiability of Equality Equations (Medium) - 45 min")
    
    print("\n🧪 Running Tests...")
    test_day16_problems()
    
    print("\n✅ Day 16 Complete!")
    print("📈 Next: Day 17 - Tries & String Algorithms")


if __name__ == "__main__":
    main() 