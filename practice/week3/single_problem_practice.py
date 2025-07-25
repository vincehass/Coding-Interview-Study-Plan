"""
=============================================================================
                        WEEK 3 SINGLE PROBLEM PRACTICE
                            NUMBER OF ISLANDS
                           Meta Interview Preparation
=============================================================================

Focus on mastering one core graph problem with comprehensive testing.
This represents the most fundamental graph traversal pattern in interviews.

=============================================================================
"""

from typing import List
from collections import deque


def num_islands(grid: List[List[str]]) -> int:
    """
    PROBLEM: Number of Islands
    
    DESCRIPTION:
    Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), 
    return the number of islands.
    
    An island is surrounded by water and is formed by connecting adjacent lands 
    horizontally or vertically. You may assume all four edges of the grid are 
    all surrounded by water.
    
    CONSTRAINTS:
    - m == grid.length
    - n == grid[i].length
    - 1 <= m, n <= 300
    - grid[i][j] is '0' or '1'.
    
    EXAMPLES:
    Example 1:
        Input: grid = [
          ["1","1","1","1","0"],
          ["1","1","0","1","0"],
          ["1","1","0","0","0"],
          ["0","0","0","0","0"]
        ]
        Output: 1
    
    Example 2:
        Input: grid = [
          ["1","1","0","0","0"],
          ["1","1","0","0","0"],
          ["0","0","1","0","0"],
          ["0","0","0","1","1"]
        ]
        Output: 3
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(m * n) worst case for recursion stack
    
    Args:
        grid (List[List[str]]): 2D grid of '0's and '1's
        
    Returns:
        int: Number of islands in the grid
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        # Base case: out of bounds or water/visited
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        
        # Mark current cell as visited by changing it to '0'
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r + 1, c)  # down
        dfs(r - 1, c)  # up
        dfs(r, c + 1)  # right
        dfs(r, c - 1)  # left
    
    # Traverse the entire grid
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':  # Found unvisited land
                islands += 1
                dfs(r, c)  # Mark entire island as visited
    
    return islands


def print_grid(grid: List[List[str]], title: str = "Grid"):
    """Helper function to visualize the grid"""
    print(f"{title}:")
    for row in grid:
        print("  " + " ".join(row))
    print()


def main():
    """Test the num_islands function with various test cases"""
    
    print("=" * 60)
    print("           WEEK 3 SINGLE PROBLEM PRACTICE")
    print("                NUMBER OF ISLANDS")
    print("=" * 60)
    
    # Test Case 1: Example 1 from problem description
    print("\n🧪 Test Case 1: Single Large Island")
    grid1 = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
    ]
    expected1 = 1
    # Make a copy since our solution modifies the grid
    grid1_copy = [row[:] for row in grid1]
    result1 = num_islands(grid1_copy)
    
    print_grid(grid1, "Input Grid")
    print(f"Expected: {expected1}")
    print(f"Got: {result1}")
    print(f"✅ PASS" if result1 == expected1 else f"❌ FAIL")
    
    # Test Case 2: Example 2 from problem description  
    print("\n🧪 Test Case 2: Multiple Islands")
    grid2 = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    expected2 = 3
    grid2_copy = [row[:] for row in grid2]
    result2 = num_islands(grid2_copy)
    
    print_grid(grid2, "Input Grid")
    print(f"Expected: {expected2}")
    print(f"Got: {result2}")
    print(f"✅ PASS" if result2 == expected2 else f"❌ FAIL")
    
    # Test Case 3: No islands (all water)
    print("\n🧪 Test Case 3: No Islands (All Water)")
    grid3 = [
        ["0","0","0"],
        ["0","0","0"],
        ["0","0","0"]
    ]
    expected3 = 0
    grid3_copy = [row[:] for row in grid3]
    result3 = num_islands(grid3_copy)
    
    print_grid(grid3, "Input Grid")
    print(f"Expected: {expected3}")
    print(f"Got: {result3}")
    print(f"✅ PASS" if result3 == expected3 else f"❌ FAIL")
    
    # Test Case 4: All land (single island)
    print("\n🧪 Test Case 4: All Land (Single Island)")
    grid4 = [
        ["1","1","1"],
        ["1","1","1"],
        ["1","1","1"]
    ]
    expected4 = 1
    grid4_copy = [row[:] for row in grid4]
    result4 = num_islands(grid4_copy)
    
    print_grid(grid4, "Input Grid")
    print(f"Expected: {expected4}")
    print(f"Got: {result4}")
    print(f"✅ PASS" if result4 == expected4 else f"❌ FAIL")
    
    # Test Case 5: Single cell island
    print("\n🧪 Test Case 5: Single Cell Island")
    grid5 = [["1"]]
    expected5 = 1
    grid5_copy = [row[:] for row in grid5]
    result5 = num_islands(grid5_copy)
    
    print_grid(grid5, "Input Grid")
    print(f"Expected: {expected5}")
    print(f"Got: {result5}")
    print(f"✅ PASS" if result5 == expected5 else f"❌ FAIL")
    
    # Test Case 6: Single cell water
    print("\n🧪 Test Case 6: Single Cell Water")
    grid6 = [["0"]]
    expected6 = 0
    grid6_copy = [row[:] for row in grid6]
    result6 = num_islands(grid6_copy)
    
    print_grid(grid6, "Input Grid")
    print(f"Expected: {expected6}")
    print(f"Got: {result6}")
    print(f"✅ PASS" if result6 == expected6 else f"❌ FAIL")
    
    # Test Case 7: Diagonal islands (should be separate)
    print("\n🧪 Test Case 7: Diagonal Islands (Not Connected)")
    grid7 = [
        ["1","0","1"],
        ["0","1","0"],
        ["1","0","1"]
    ]
    expected7 = 5  # Each '1' is a separate island (no diagonal connections)
    grid7_copy = [row[:] for row in grid7]
    result7 = num_islands(grid7_copy)
    
    print_grid(grid7, "Input Grid")
    print(f"Expected: {expected7} (diagonal doesn't count as connected)")
    print(f"Got: {result7}")
    print(f"✅ PASS" if result7 == expected7 else f"❌ FAIL")
    
    # Test Case 8: L-shaped island
    print("\n🧪 Test Case 8: L-Shaped Island")
    grid8 = [
        ["1","1","0","0"],
        ["1","0","0","0"],
        ["1","0","1","1"],
        ["1","1","1","0"]
    ]
    expected8 = 2  # One L-shaped island on left, one small island on right
    grid8_copy = [row[:] for row in grid8]
    result8 = num_islands(grid8_copy)
    
    print_grid(grid8, "Input Grid")
    print(f"Expected: {expected8}")
    print(f"Got: {result8}")
    print(f"✅ PASS" if result8 == expected8 else f"❌ FAIL")
    
    # Test Case 9: Many Small Islands
    print("\n🧪 Test Case 9: Many Small Islands")
    grid9 = [
        ["1","0","1","0","1"],
        ["0","1","0","1","0"],
        ["1","0","1","0","1"],
        ["0","1","0","1","0"]
    ]
    expected9 = 10  # Each '1' is separate
    grid9_copy = [row[:] for row in grid9]
    result9 = num_islands(grid9_copy)
    
    print_grid(grid9, "Input Grid")
    print(f"Expected: {expected9}")
    print(f"Got: {result9}")
    print(f"✅ PASS" if result9 == expected9 else f"❌ FAIL")
    
    print("\n" + "=" * 60)
    print("                  TEST SUMMARY")
    print("=" * 60)
    
    # Count passes
    test_results = [
        result1 == expected1,
        result2 == expected2,
        result3 == expected3,
        result4 == expected4,
        result5 == expected5,
        result6 == expected6,
        result7 == expected7,
        result8 == expected8,
        result9 == expected9
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Great job!")
    else:
        print("❌ Some tests failed. Review your solution.")
    
    print("\n💡 SOLUTION APPROACHES:")
    print("1. DFS (Depth-First Search): Use recursion to explore connected land")
    print("2. BFS (Breadth-First Search): Use queue to explore level by level")
    print("3. Union-Find: Use disjoint set data structure")
    
    print("\n📚 LEARNING OBJECTIVES:")
    print("- Master graph traversal algorithms (DFS/BFS)")
    print("- Understand connected components in graphs")
    print("- Practice 2D grid traversal patterns")
    print("- Learn to handle boundary conditions and edge cases")


# Reference solutions (uncomment to check your work)
def num_islands_dfs(grid: List[List[str]]) -> int:
    """
    Reference DFS solution
    Time: O(m * n), Space: O(m * n) for recursion stack
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        # Base case: out of bounds or water
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        
        # Mark as visited
        grid[r][c] = '0'
        
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
    Reference BFS solution
    Time: O(m * n), Space: O(m * n) for queue
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
                    grid[nr][nc] = '0'  # Mark as visited
                    queue.append((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                bfs(r, c)
    
    return islands


if __name__ == "__main__":
    main() 