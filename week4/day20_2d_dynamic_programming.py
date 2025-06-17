"""
=============================================================================
                        DAY 20: 2D DYNAMIC PROGRAMMING
                           Meta Interview Preparation
                              Week 4 - Day 20
=============================================================================

FOCUS: Grid-based DP, string matching
TIME ALLOCATION: 4 hours
- Theory (1 hour): 2D DP patterns, state transitions
- Problems (3 hours): Classic 2D DP problems

TOPICS COVERED:
- Grid path problems
- String matching algorithms
- 2D state representation
- Space optimization in 2D DP

=============================================================================
"""

from typing import List, Dict
from functools import lru_cache


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
2D DYNAMIC PROGRAMMING FUNDAMENTALS:

Key Patterns:
1. GRID PATHS: dp[i][j] represents optimal solution at position (i,j)
2. STRING MATCHING: dp[i][j] represents solution using first i chars of string1, j chars of string2
3. RANGE DP: dp[i][j] represents solution for subarray/substring from i to j

Common Recurrence Relations:
- Grid: dp[i][j] = f(dp[i-1][j], dp[i][j-1])
- String: dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

Space Optimization:
- Often reduce from O(m*n) to O(min(m,n))
- Use rolling arrays when only previous row/column needed
"""


# =============================================================================
# PROBLEM 1: UNIQUE PATHS (MEDIUM) - 30 MIN
# =============================================================================

def unique_paths(m: int, n: int) -> int:
    """
    PROBLEM: Unique Paths
    
    There is a robot on an m x n grid. The robot is initially located at the 
    top-left corner (i.e., grid[0][0]). The robot tries to move to the 
    bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move 
    either down or right at any point in time.
    
    Given the two integers m and n, return the number of possible unique paths 
    that the robot can take to reach the bottom-right corner.
    
    CONSTRAINTS:
    - 1 <= m, n <= 100
    
    EXAMPLES:
    Example 1:
        Input: m = 3, n = 7
        Output: 28
        Explanation: There are 28 unique paths from top-left to bottom-right
    
    Example 2:
        Input: m = 3, n = 2
        Output: 3
        Explanation: From top-left corner, there are 3 ways to reach bottom-right:
        1. Right -> Down -> Down
        2. Down -> Down -> Right
        3. Down -> Right -> Down
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(n) after optimization
    
    GOAL: Master basic 2D DP grid traversal pattern
    """
    # Space optimized version - only need previous row
    prev = [1] * n
    
    for i in range(1, m):
        curr = [1] * n
        for j in range(1, n):
            curr[j] = curr[j-1] + prev[j]
        prev = curr
    
    return prev[n-1]


def unique_paths_with_obstacles(obstacleGrid: List[List[int]]) -> int:
    """
    VARIANT: Unique Paths II - with obstacles
    
    PROBLEM: A robot is located at the top-left corner of a m x n grid. 
    The robot can only move either down or right. An obstacle and space 
    is marked as 1 and 0 respectively in the grid.
    
    GOAL: Handle obstacles in grid DP problems
    
    TIME: O(m * n), SPACE: O(n)
    """
    if not obstacleGrid or obstacleGrid[0][0] == 1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [0] * n
    dp[0] = 1
    
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    
    return dp[n-1]


# =============================================================================
# PROBLEM 2: MINIMUM PATH SUM (MEDIUM) - 45 MIN
# =============================================================================

def min_path_sum(grid: List[List[int]]) -> int:
    """
    PROBLEM: Minimum Path Sum
    
    Given a m x n grid filled with non-negative numbers, find a path from 
    top left to bottom right, which minimizes the sum of all numbers along 
    its path. You can only move either down or right at any point in time.
    
    CONSTRAINTS:
    - m == grid.length
    - n == grid[i].length
    - 1 <= m, n <= 200
    - 0 <= grid[i][j] <= 100
    
    EXAMPLES:
    Example 1:
        Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
        Output: 7
        Explanation: Path 1→3→1→1→1 minimizes the sum
    
    Example 2:
        Input: grid = [[1,2,3],[4,5,6]]
        Output: 12
        Explanation: Path 1→2→3→6 minimizes the sum
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(1) - modify input or O(n) - separate array
    
    GOAL: Apply optimization objectives in grid DP
    """
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    
    # Initialize first row and column
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    
    for j in range(1, n):
        grid[0][j] += grid[0][j-1]
    
    # Fill the rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    
    return grid[m-1][n-1]


# =============================================================================
# PROBLEM 3: LONGEST COMMON SUBSEQUENCE (MEDIUM) - 60 MIN
# =============================================================================

def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    PROBLEM: Longest Common Subsequence
    
    Given two strings text1 and text2, return the length of their longest 
    common subsequence. If there is no common subsequence, return 0.
    
    A subsequence of a string is a new string generated from the original 
    string with some characters (can be none) deleted without changing the 
    relative order of the remaining characters.
    
    CONSTRAINTS:
    - 1 <= text1.length, text2.length <= 1000
    - text1 and text2 consist of only lowercase English characters
    
    EXAMPLES:
    Example 1:
        Input: text1 = "abcde", text2 = "ace"
        Output: 3
        Explanation: The longest common subsequence is "ace" with length 3
    
    Example 2:
        Input: text1 = "abc", text2 = "abc"
        Output: 3
        Explanation: The longest common subsequence is "abc" with length 3
    
    Example 3:
        Input: text1 = "abc", text2 = "def"
        Output: 0
        Explanation: There is no common subsequence
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(min(m, n))
    
    GOAL: Master string matching DP pattern
    """
    m, n = len(text1), len(text2)
    
    # Space optimized: only need previous row
    prev = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    
    return prev[n]


def lcs_with_path(text1: str, text2: str) -> str:
    """
    VARIANT: Return the actual LCS string, not just length
    
    GOAL: Reconstruct solution path in DP problems
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))


# =============================================================================
# PROBLEM 4: EDIT DISTANCE (HARD) - 75 MIN
# =============================================================================

def min_distance(word1: str, word2: str) -> int:
    """
    PROBLEM: Edit Distance (Levenshtein Distance)
    
    Given two strings word1 and word2, return the minimum number of operations 
    required to convert word1 to word2.
    
    You have the following three operations permitted on a word:
    - Insert a character
    - Delete a character
    - Replace a character
    
    CONSTRAINTS:
    - 0 <= word1.length, word2.length <= 500
    - word1 and word2 consist of lowercase English letters
    
    EXAMPLES:
    Example 1:
        Input: word1 = "horse", word2 = "ros"
        Output: 3
        Explanation: 
        horse -> rorse (replace 'h' with 'r')
        rorse -> rose (remove 'r')
        rose -> ros (remove 'e')
    
    Example 2:
        Input: word1 = "intention", word2 = "execution"
        Output: 5
        Explanation:
        intention -> inention (remove 't')
        inention -> enention (replace 'i' with 'e')
        enention -> exention (replace 'n' with 'x')
        exention -> exection (replace 'n' with 'c')
        exection -> execution (insert 'u')
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(min(m, n))
    
    GOAL: Master complex string transformation DP
    """
    m, n = len(word1), len(word2)
    
    # Space optimized version
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(
                    prev[j],      # Delete
                    curr[j-1],    # Insert
                    prev[j-1]     # Replace
                )
        prev = curr
    
    return prev[n]


# =============================================================================
# PROBLEM 5: MAXIMAL SQUARE (MEDIUM) - 60 MIN
# =============================================================================

def maximal_square(matrix: List[List[str]]) -> int:
    """
    PROBLEM: Maximal Square
    
    Given an m x n binary matrix filled with 0's and 1's, find the largest 
    square containing only 1's and return its area.
    
    CONSTRAINTS:
    - m == matrix.length
    - n == matrix[i].length
    - 1 <= m, n <= 300
    - matrix[i][j] is '0' or '1'
    
    EXAMPLES:
    Example 1:
        Input: matrix = [["1","0","1","0","0"],
                        ["1","0","1","1","1"],
                        ["1","1","1","1","1"],
                        ["1","0","0","1","0"]]
        Output: 4
        Explanation: The largest square has side length 2, so area = 4
    
    Example 2:
        Input: matrix = [["0","1"],["1","0"]]
        Output: 1
        Explanation: The largest square has side length 1, so area = 1
    
    Example 3:
        Input: matrix = [["0"]]
        Output: 0
        Explanation: No square of 1's exists
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    GOAL: Apply DP to geometric optimization problems
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    prev = [0] * (n + 1)
    max_side = 0
    
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if matrix[i-1][j-1] == '1':
                curr[j] = min(prev[j], curr[j-1], prev[j-1]) + 1
                max_side = max(max_side, curr[j])
        prev = curr
    
    return max_side * max_side


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day20_problems():
    """Test all Day 20 problems"""
    
    print("=" * 60)
    print("         DAY 20: 2D DYNAMIC PROGRAMMING")
    print("=" * 60)
    
    # Test Unique Paths
    print("\n🧪 Testing Unique Paths")
    paths1 = unique_paths(3, 7)
    print(f"Unique Paths (3x7): {paths1} (Expected: 28)")
    
    paths2 = unique_paths(3, 2)
    print(f"Unique Paths (3x2): {paths2} (Expected: 3)")
    
    # Test Unique Paths with Obstacles
    print("\n🧪 Testing Unique Paths with Obstacles")
    obstacle_grid = [[0,0,0],[0,1,0],[0,0,0]]
    paths_obstacles = unique_paths_with_obstacles(obstacle_grid)
    print(f"Paths with Obstacles: {paths_obstacles} (Expected: 2)")
    
    # Test Minimum Path Sum
    print("\n🧪 Testing Minimum Path Sum")
    grid1 = [[1,3,1],[1,5,1],[4,2,1]]
    min_sum = min_path_sum([row[:] for row in grid1])  # Deep copy
    print(f"Min Path Sum: {min_sum} (Expected: 7)")
    
    # Test Longest Common Subsequence
    print("\n🧪 Testing Longest Common Subsequence")
    lcs_len = longest_common_subsequence("abcde", "ace")
    print(f"LCS Length ('abcde', 'ace'): {lcs_len} (Expected: 3)")
    
    lcs_str = lcs_with_path("abcde", "ace")
    print(f"LCS String ('abcde', 'ace'): '{lcs_str}' (Expected: 'ace')")
    
    # Test Edit Distance
    print("\n🧪 Testing Edit Distance")
    edit_dist1 = min_distance("horse", "ros")
    print(f"Edit Distance ('horse', 'ros'): {edit_dist1} (Expected: 3)")
    
    edit_dist2 = min_distance("intention", "execution")
    print(f"Edit Distance ('intention', 'execution'): {edit_dist2} (Expected: 5)")
    
    # Test Maximal Square
    print("\n🧪 Testing Maximal Square")
    matrix = [["1","0","1","0","0"],
              ["1","0","1","1","1"],
              ["1","1","1","1","1"],
              ["1","0","0","1","0"]]
    max_square = maximal_square(matrix)
    print(f"Maximal Square Area: {max_square} (Expected: 4)")
    
    print("\n" + "=" * 60)
    print("           DAY 20 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 20 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review and optimize
    """
    
    print("🚀 Starting Day 20: 2D Dynamic Programming")
    print("\n📚 Theory Topics:")
    print("- Grid path problems and state transitions")
    print("- String matching DP patterns")
    print("- 2D state representation techniques")
    print("- Space optimization in 2D DP")
    
    print("\n💻 Practice Problems:")
    print("1. Unique Paths (Medium) - 30 min")
    print("2. Minimum Path Sum (Medium) - 45 min")
    print("3. Longest Common Subsequence (Medium) - 60 min")
    print("4. Edit Distance (Hard) - 75 min")
    print("5. Maximal Square (Medium) - 60 min")
    
    print("\n🧪 Running Tests...")
    test_day20_problems()
    
    print("\n✅ Day 20 Complete!")
    print("📈 Next: Day 21 - Advanced DP Patterns")


if __name__ == "__main__":
    main() 