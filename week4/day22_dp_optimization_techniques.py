"""
=============================================================================
                        DAY 22: DP OPTIMIZATION TECHNIQUES
                           Meta Interview Preparation
                              Week 4 - Day 22
=============================================================================

FOCUS: Space/time optimization, advanced techniques
TIME ALLOCATION: 4 hours
- Theory (1 hour): Optimization techniques and trade-offs
- Problems (3 hours): Problems demonstrating optimization strategies

TOPICS COVERED:
- Space optimization in DP
- Rolling arrays and state compression
- Matrix exponentiation for DP
- Monotonic deque optimization

=============================================================================
"""

from typing import List, Dict, Tuple
from collections import deque
import heapq


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
DP OPTIMIZATION TECHNIQUES:

1. SPACE OPTIMIZATION:
   - Reduce from O(n²) to O(n) or O(1)
   - Use rolling arrays when only previous states needed
   - State compression for boolean states

2. SLIDING WINDOW MAXIMUM:
   - Use monotonic deque for range queries
   - Maintain decreasing order in deque
   - Efficient for DP with range dependencies

3. MATRIX EXPONENTIATION:
   - For linear recurrences with large n
   - Reduce O(n) to O(log n)
   - Useful for Fibonacci-like sequences

4. CONVEX HULL OPTIMIZATION:
   - For DP with quadratic cost functions
   - Maintain convex hull of cost functions
   - Reduce O(n²) to O(n log n) or O(n)
"""


# =============================================================================
# PROBLEM 1: PERFECT SQUARES (MEDIUM) - 45 MIN
# =============================================================================

def num_squares(n: int) -> int:
    """
    PROBLEM: Perfect Squares
    
    Given an integer n, return the least number of perfect square numbers 
    that sum to n. A perfect square is an integer that is the square of 
    an integer; in other words, it is the product of some integer with itself.
    
    CONSTRAINTS:
    - 1 <= n <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: n = 12
        Output: 3
        Explanation: 12 = 4 + 4 + 4 (three perfect squares)
    
    Example 2:
        Input: n = 13
        Output: 2
        Explanation: 13 = 4 + 9 (two perfect squares)
    
    Example 3:
        Input: n = 1
        Output: 1
        Explanation: 1 = 1 (one perfect square)
    
    EXPECTED TIME COMPLEXITY: O(n * √n)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    GOAL: Apply coin change pattern to mathematical problems
    """
    # Generate perfect squares up to n
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1
    
    # DP: minimum squares needed for each number
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        for square in squares:
            if square > i:
                break
            dp[i] = min(dp[i], dp[i - square] + 1)
    
    return dp[n]


def num_squares_optimized(n: int) -> int:
    """
    OPTIMIZED: Using mathematical properties (Lagrange's four-square theorem)
    
    GOAL: Understand when mathematical insights can optimize DP
    """
    import math
    
    # Check if n is a perfect square
    sqrt_n = int(math.sqrt(n))
    if sqrt_n * sqrt_n == n:
        return 1
    
    # Check if n can be represented as sum of two squares
    for i in range(1, int(math.sqrt(n)) + 1):
        remainder = n - i * i
        sqrt_remainder = int(math.sqrt(remainder))
        if sqrt_remainder * sqrt_remainder == remainder:
            return 2
    
    # Check if n is of the form 4^k * (8m + 7)
    while n % 4 == 0:
        n //= 4
    if n % 8 == 7:
        return 4
    
    return 3


# =============================================================================
# PROBLEM 2: MAXIMAL RECTANGLE (HARD) - 75 MIN
# =============================================================================

def maximal_rectangle(matrix: List[List[str]]) -> int:
    """
    PROBLEM: Maximal Rectangle
    
    Given a rows x cols binary matrix filled with 0's and 1's, find the 
    largest rectangle containing only 1's and return its area.
    
    CONSTRAINTS:
    - rows == matrix.length
    - cols == matrix[i].length
    - 1 <= rows, cols <= 200
    - matrix[i][j] is '0' or '1'
    
    EXAMPLES:
    Example 1:
        Input: matrix = [["1","0","1","0","0"],
                        ["1","0","1","1","1"],
                        ["1","1","1","1","1"],
                        ["1","0","0","1","0"]]
        Output: 6
        Explanation: The maximal rectangle has area 6
    
    Example 2:
        Input: matrix = [["0"]]
        Output: 0
    
    Example 3:
        Input: matrix = [["1"]]
        Output: 1
    
    EXPECTED TIME COMPLEXITY: O(rows * cols)
    EXPECTED SPACE COMPLEXITY: O(cols)
    
    GOAL: Combine histogram optimization with DP
    """
    if not matrix or not matrix[0]:
        return 0
    
    cols = len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    def largest_rectangle_in_histogram(heights):
        """Helper: Largest rectangle in histogram using stack"""
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)
        
        while stack:
            h = heights[stack.pop()]
            w = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        return max_area
    
    for row in matrix:
        # Update heights for current row
        for j in range(cols):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        # Find max rectangle in current histogram
        max_area = max(max_area, largest_rectangle_in_histogram(heights))
    
    return max_area


# =============================================================================
# PROBLEM 3: SLIDING WINDOW MAXIMUM (HARD) - 60 MIN
# =============================================================================

def max_sliding_window_dp(nums: List[int], k: int) -> List[int]:
    """
    PROBLEM: Sliding Window Maximum (DP Perspective)
    
    You are given an array of integers nums, there is a sliding window of 
    size k which is moving from the very left of the array to the very right. 
    You can only see the k numbers in the window. Each time the sliding window 
    moves right by one position. Return the max sliding window.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    - 1 <= k <= nums.length
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
        Output: [3,3,5,5,6,7]
        Explanation: 
        Window [1,3,-1] -> max is 3
        Window [3,-1,-3] -> max is 3
        Window [-1,-3,5] -> max is 5
        Window [-3,5,3] -> max is 5
        Window [5,3,6] -> max is 6
        Window [3,6,7] -> max is 7
    
    Example 2:
        Input: nums = [1], k = 1
        Output: [1]
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(k)
    
    GOAL: Master monotonic deque optimization technique
    """
    if not nums or k == 0:
        return []
    
    # Monotonic deque to maintain maximum elements
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values (they'll never be maximum)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum of current window to result
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


# =============================================================================
# PROBLEM 4: FIBONACCI WITH MATRIX EXPONENTIATION (MEDIUM) - 60 MIN
# =============================================================================

def fibonacci_matrix(n: int) -> int:
    """
    PROBLEM: Fibonacci Number (Optimized)
    
    The Fibonacci numbers form a sequence where each number is the sum of 
    the two preceding ones, starting from 0 and 1. Given n, calculate F(n).
    
    CONSTRAINTS:
    - 0 <= n <= 30
    
    EXAMPLES:
    Example 1:
        Input: n = 2
        Output: 1
        Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1
    
    Example 2:
        Input: n = 3
        Output: 2
        Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2
    
    Example 3:
        Input: n = 4
        Output: 3
        Explanation: F(4) = F(3) + F(2) = 2 + 1 = 3
    
    EXPECTED TIME COMPLEXITY: O(log n)
    EXPECTED SPACE COMPLEXITY: O(log n)
    
    GOAL: Learn matrix exponentiation for linear recurrences
    """
    if n <= 1:
        return n
    
    def matrix_multiply(A, B):
        """Multiply two 2x2 matrices"""
        return [
            [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
        ]
    
    def matrix_power(matrix, power):
        """Calculate matrix^power using fast exponentiation"""
        if power == 1:
            return matrix
        
        if power % 2 == 0:
            half = matrix_power(matrix, power // 2)
            return matrix_multiply(half, half)
        else:
            return matrix_multiply(matrix, matrix_power(matrix, power - 1))
    
    # Fibonacci transformation matrix
    # [F(n+1)]   [1 1] [F(n)  ]
    # [F(n)  ] = [1 0] [F(n-1)]
    base_matrix = [[1, 1], [1, 0]]
    result_matrix = matrix_power(base_matrix, n)
    
    return result_matrix[0][1]  # F(n)


# =============================================================================
# PROBLEM 5: LARGEST DIVISIBLE SUBSET (MEDIUM) - 75 MIN
# =============================================================================

def largest_divisible_subset(nums: List[int]) -> List[int]:
    """
    PROBLEM: Largest Divisible Subset
    
    Given a set of distinct positive integers nums, return the largest subset 
    such that every pair (nums[i], nums[j]) of elements in this subset satisfies:
    - nums[i] % nums[j] == 0, or
    - nums[j] % nums[i] == 0
    
    If there are multiple solutions, return any of them.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 1000
    - 1 <= nums[i] <= 2 * 10^9
    - All integers in nums are unique
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,2,3]
        Output: [1,2]
        Explanation: [1,3] is also a valid result
    
    Example 2:
        Input: nums = [1,2,4,8]
        Output: [1,2,4,8]
        Explanation: All pairs satisfy the divisibility condition
    
    Example 3:
        Input: nums = [1,4,8,2]
        Output: [1,2,4,8]
        Explanation: Sorted subset maintains divisibility
    
    EXPECTED TIME COMPLEXITY: O(n²)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    GOAL: Apply LIS pattern with custom constraints
    """
    if not nums:
        return []
    
    nums.sort()
    n = len(nums)
    
    # dp[i] = length of largest divisible subset ending at index i
    dp = [1] * n
    parent = [-1] * n
    
    max_length = 1
    max_index = 0
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
        
        if dp[i] > max_length:
            max_length = dp[i]
            max_index = i
    
    # Reconstruct the subset
    result = []
    current = max_index
    while current != -1:
        result.append(nums[current])
        current = parent[current]
    
    return result[::-1]


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day22_problems():
    """Test all Day 22 problems"""
    
    print("=" * 60)
    print("         DAY 22: DP OPTIMIZATION TECHNIQUES")
    print("=" * 60)
    
    # Test Perfect Squares
    print("\n🧪 Testing Perfect Squares")
    squares1 = num_squares(12)
    print(f"Perfect Squares (12): {squares1} (Expected: 3)")
    
    squares2 = num_squares(13)
    print(f"Perfect Squares (13): {squares2} (Expected: 2)")
    
    squares_opt = num_squares_optimized(12)
    print(f"Perfect Squares Optimized (12): {squares_opt} (Expected: 3)")
    
    # Test Maximal Rectangle
    print("\n🧪 Testing Maximal Rectangle")
    matrix = [["1","0","1","0","0"],
              ["1","0","1","1","1"],
              ["1","1","1","1","1"],
              ["1","0","0","1","0"]]
    max_rect = maximal_rectangle(matrix)
    print(f"Maximal Rectangle: {max_rect} (Expected: 6)")
    
    # Test Sliding Window Maximum
    print("\n🧪 Testing Sliding Window Maximum")
    window_max = max_sliding_window_dp([1,3,-1,-3,5,3,6,7], 3)
    print(f"Sliding Window Max: {window_max} (Expected: [3,3,5,5,6,7])")
    
    # Test Fibonacci Matrix
    print("\n🧪 Testing Fibonacci Matrix Exponentiation")
    fib_4 = fibonacci_matrix(4)
    print(f"Fibonacci(4): {fib_4} (Expected: 3)")
    
    fib_10 = fibonacci_matrix(10)
    print(f"Fibonacci(10): {fib_10} (Expected: 55)")
    
    # Test Largest Divisible Subset
    print("\n🧪 Testing Largest Divisible Subset")
    subset1 = largest_divisible_subset([1,2,4,8])
    print(f"Largest Divisible Subset [1,2,4,8]: {subset1} (Expected: [1,2,4,8])")
    
    subset2 = largest_divisible_subset([1,2,3])
    print(f"Largest Divisible Subset [1,2,3]: {subset2} (Expected: [1,2] or [1,3])")
    
    print("\n" + "=" * 60)
    print("           DAY 22 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# OPTIMIZATION TECHNIQUES SUMMARY
# =============================================================================

def optimization_summary():
    """Summary of key optimization techniques"""
    
    print("\n" + "=" * 70)
    print("                 DP OPTIMIZATION TECHNIQUES SUMMARY")
    print("=" * 70)
    
    print("\n🚀 SPACE OPTIMIZATION:")
    print("• Rolling Arrays: Reduce O(n²) to O(n)")
    print("• State Compression: Use bits for boolean states")
    print("• In-place Updates: Modify input when possible")
    
    print("\n⚡ TIME OPTIMIZATION:")
    print("• Monotonic Deque: O(n) for sliding window problems")
    print("• Matrix Exponentiation: O(log n) for linear recurrences")
    print("• Mathematical Insights: Reduce complexity with theory")
    
    print("\n🎯 PROBLEM PATTERNS:")
    print("• Coin Change → Perfect Squares")
    print("• LIS → Largest Divisible Subset")
    print("• Histogram → Maximal Rectangle")
    print("• Fibonacci → Matrix Exponentiation")
    
    print("\n💡 OPTIMIZATION STRATEGIES:")
    print("1. Identify dependencies between states")
    print("2. Eliminate unnecessary dimensions")
    print("3. Use data structures for efficient queries")
    print("4. Apply mathematical properties when available")
    
    print("=" * 70)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 22 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review optimization techniques
    """
    
    print("🚀 Starting Day 22: DP Optimization Techniques")
    print("\n📚 Theory Topics:")
    print("- Space optimization strategies")
    print("- Rolling arrays and state compression")
    print("- Matrix exponentiation for DP")
    print("- Monotonic deque optimization")
    
    print("\n💻 Practice Problems:")
    print("1. Perfect Squares (Medium) - 45 min")
    print("2. Maximal Rectangle (Hard) - 75 min")
    print("3. Sliding Window Maximum (Hard) - 60 min")
    print("4. Fibonacci Matrix Exponentiation (Medium) - 60 min")
    print("5. Largest Divisible Subset (Medium) - 75 min")
    
    print("\n🧪 Running Tests...")
    test_day22_problems()
    
    print("\n📊 Optimization Summary...")
    optimization_summary()
    
    print("\n✅ Day 22 Complete!")
    print("📈 Next: Day 23 - DP on Trees and Graphs")


if __name__ == "__main__":
    main() 