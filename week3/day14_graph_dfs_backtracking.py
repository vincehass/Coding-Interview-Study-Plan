"""
=============================================================================
                        DAY 14: GRAPH DFS & BACKTRACKING
                           Meta Interview Preparation
                              Week 3 - Day 14
=============================================================================

FOCUS: Depth-first search, path exploration
TIME ALLOCATION: 4 hours
- Theory (1 hour): DFS algorithm, backtracking patterns
- Problems (3 hours): DFS and backtracking problems

TOPICS COVERED:
- DFS algorithm and recursion
- Backtracking patterns
- Path tracking and state management
- Tree and graph traversal with DFS

=============================================================================
"""

from typing import List, Dict, Set
from collections import defaultdict


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
DFS ALGORITHM:
- Time: O(V + E)
- Space: O(V) for recursion stack
- Uses stack (LIFO) or recursion
- Explores as far as possible before backtracking
- Good for finding paths, cycles, connected components

BACKTRACKING PATTERN:
1. Choose: Make a choice and move forward
2. Explore: Recursively explore the consequences
3. Unchoose: Undo the choice (backtrack)

Common Applications:
- Finding all solutions
- Path problems
- Permutations and combinations
- Constraint satisfaction problems
"""


# =============================================================================
# PROBLEM 1: PATH SUM II (MEDIUM) - 45 MIN
# =============================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def path_sum(root: TreeNode, targetSum: int) -> List[List[int]]:
    """
    PROBLEM: Path Sum II
    
    Given the root of a binary tree and an integer targetSum, return all 
    root-to-leaf paths where the sum of the node values equals targetSum.
    
    Example:
    Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
    Output: [[5,4,11,2],[5,8,4,5]]
    
    TIME: O(N²), SPACE: O(N²)
    """
    result = []
    
    def dfs(node, remaining, path):
        if not node:
            return
        
        path.append(node.val)
        
        # Check if it's a leaf and sum equals target
        if not node.left and not node.right and remaining == node.val:
            result.append(path.copy())
        else:
            dfs(node.left, remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)
        
        path.pop()  # Backtrack
    
    dfs(root, targetSum, [])
    return result


# =============================================================================
# PROBLEM 2: GENERATE PARENTHESES (MEDIUM) - 45 MIN
# =============================================================================

def generate_parentheses(n: int) -> List[str]:
    """
    PROBLEM: Generate Parentheses
    
    Given n pairs of parentheses, write a function to generate all 
    combinations of well-formed parentheses.
    
    Example:
    Input: n = 3
    Output: ["((()))","(()())","(())()","()(())","()()()"]
    
    TIME: O(4^n / √n), SPACE: O(4^n / √n)
    """
    result = []
    
    def backtrack(current, open_count, close_count):
        # Base case: we've used all n pairs
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Add opening parenthesis if we haven't used all n
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        
        # Add closing parenthesis if it would still be valid
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)
    
    backtrack("", 0, 0)
    return result


# =============================================================================
# PROBLEM 3: LETTER COMBINATIONS OF PHONE NUMBER (MEDIUM) - 45 MIN
# =============================================================================

def letter_combinations(digits: str) -> List[str]:
    """
    PROBLEM: Letter Combinations of a Phone Number
    
    Given a string containing digits from 2-9 inclusive, return all possible 
    letter combinations that the number could represent.
    
    Example:
    Input: digits = "23"
    Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
    
    TIME: O(3^m * 4^n), SPACE: O(3^m * 4^n)
    where m is digits with 3 letters, n is digits with 4 letters
    """
    if not digits:
        return []
    
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, current):
        if index == len(digits):
            result.append(current)
            return
        
        for letter in phone_map[digits[index]]:
            backtrack(index + 1, current + letter)
    
    backtrack(0, "")
    return result


# =============================================================================
# PROBLEM 4: WORD SEARCH (MEDIUM) - 60 MIN
# =============================================================================

def exist(board: List[List[str]], word: str) -> bool:
    """
    PROBLEM: Word Search
    
    Given an m x n grid of characters board and a string word, return true 
    if word exists in the grid.
    
    The word can be constructed from letters of sequentially adjacent cells, 
    where adjacent cells are horizontally or vertically neighboring.
    
    Example:
    Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], 
           word = "ABCCED"
    Output: true
    
    TIME: O(M * N * 4^L), SPACE: O(L)
    """
    if not board or not board[0]:
        return False
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, index, visited):
        if index == len(word):
            return True
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            (r, c) in visited or board[r][c] != word[index]):
            return False
        
        visited.add((r, c))
        
        # Explore all 4 directions
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dr, dc in directions:
            if dfs(r + dr, c + dc, index + 1, visited):
                return True
        
        visited.remove((r, c))  # Backtrack
        return False
    
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0, set()):
                return True
    
    return False


# =============================================================================
# PROBLEM 5: N-QUEENS (HARD) - 75 MIN
# =============================================================================

def solve_n_queens(n: int) -> List[List[str]]:
    """
    PROBLEM: N-Queens
    
    The n-queens puzzle is the problem of placing n queens on an n x n 
    chessboard such that no two queens attack each other.
    
    Given an integer n, return all distinct solutions to the n-queens puzzle.
    
    Example:
    Input: n = 4
    Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
    
    TIME: O(N!), SPACE: O(N²)
    """
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonal (top-left to bottom-right)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j - 1
        
        # Check diagonal (top-right to bottom-left)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j + 1
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'  # Backtrack
    
    backtrack(0)
    return result


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day14_problems():
    """Test all Day 14 problems"""
    
    print("=" * 60)
    print("         DAY 14: GRAPH DFS & BACKTRACKING")
    print("=" * 60)
    
    # Test Path Sum II
    print("\n🧪 Testing Path Sum II")
    # Create tree: [5,4,8,11,null,13,4,7,2,null,null,5,1]
    root = TreeNode(5)
    root.left = TreeNode(4)
    root.right = TreeNode(8)
    root.left.left = TreeNode(11)
    root.left.left.left = TreeNode(7)
    root.left.left.right = TreeNode(2)
    root.right.left = TreeNode(13)
    root.right.right = TreeNode(4)
    root.right.right.left = TreeNode(5)
    root.right.right.right = TreeNode(1)
    
    paths = path_sum(root, 22)
    print(f"Path Sum II (target=22): {paths}")
    print(f"Expected: [[5,4,11,2],[5,8,4,5]]")
    
    # Test Generate Parentheses
    print("\n🧪 Testing Generate Parentheses")
    parens = generate_parentheses(3)
    print(f"Generate Parentheses (n=3): {parens}")
    print(f"Expected 5 combinations")
    
    # Test Letter Combinations
    print("\n🧪 Testing Letter Combinations")
    combinations = letter_combinations("23")
    print(f"Letter Combinations ('23'): {combinations}")
    print(f"Expected: ['ad','ae','af','bd','be','bf','cd','ce','cf']")
    
    # Test Word Search
    print("\n🧪 Testing Word Search")
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    word_found = exist([row[:] for row in board], "ABCCED")
    print(f"Word Search 'ABCCED': {word_found} (Expected: True)")
    
    word_not_found = exist([row[:] for row in board], "ABCB")
    print(f"Word Search 'ABCB': {word_not_found} (Expected: False)")
    
    # Test N-Queens
    print("\n🧪 Testing N-Queens")
    queens_4 = solve_n_queens(4)
    print(f"N-Queens (n=4): Found {len(queens_4)} solutions (Expected: 2)")
    if queens_4:
        print(f"First solution: {queens_4[0]}")
    
    print("\n" + "=" * 60)
    print("           DAY 14 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 14 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review and optimize
    """
    
    print("🚀 Starting Day 14: Graph DFS & Backtracking")
    print("\n📚 Theory Topics:")
    print("- DFS algorithm and recursion")
    print("- Backtracking patterns")
    print("- Path tracking and state management")
    print("- Choose-Explore-Unchoose paradigm")
    
    print("\n💻 Practice Problems:")
    print("1. Path Sum II (Medium) - 45 min")
    print("2. Generate Parentheses (Medium) - 45 min")
    print("3. Letter Combinations of Phone Number (Medium) - 45 min")
    print("4. Word Search (Medium) - 60 min")
    print("5. N-Queens (Hard) - 75 min")
    
    print("\n🧪 Running Tests...")
    test_day14_problems()
    
    print("\n✅ Day 14 Complete!")
    print("📈 Next: Day 15 - Graph Algorithms")


if __name__ == "__main__":
    main() 