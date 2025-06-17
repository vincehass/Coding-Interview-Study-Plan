"""
=============================================================================
                        DAY 19: 1D DYNAMIC PROGRAMMING
                           Meta Interview Preparation
                              Week 4 - Day 19
=============================================================================

FOCUS: Linear DP patterns, optimization
TIME ALLOCATION: 4 hours
- Theory (1 hour): DP fundamentals, memoization vs tabulation
- Problems (3 hours): Classic 1D DP problems

TOPICS COVERED:
- DP problem identification
- Recurrence relation formulation
- Memoization and tabulation
- Space optimization techniques

=============================================================================
"""

from typing import List, Dict
from functools import lru_cache


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
DYNAMIC PROGRAMMING FUNDAMENTALS:

Key Characteristics:
1. OPTIMAL SUBSTRUCTURE: Solution can be constructed from optimal solutions of subproblems
2. OVERLAPPING SUBPROBLEMS: Same subproblems solved multiple times

Approaches:
1. MEMOIZATION (Top-down): Recursion + caching
2. TABULATION (Bottom-up): Iterative approach

Steps to Solve DP Problems:
1. Identify if it's a DP problem
2. Define state and variables
3. Formulate recurrence relation
4. Determine base cases
5. Implement and optimize

Space Optimization:
- Often can reduce from O(n) to O(1) space
- Keep only necessary previous states
"""


# =============================================================================
# PROBLEM 1: CLIMBING STAIRS (EASY) - 30 MIN
# =============================================================================

def climb_stairs(n: int) -> int:
    """
    PROBLEM: Climbing Stairs
    
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. In how many distinct 
    ways can you climb to the top?
    
    Example:
    Input: n = 3
    Output: 3 (1+1+1, 1+2, 2+1)
    
    TIME: O(n), SPACE: O(1)
    """
    if n <= 2:
        return n
    
    # dp[i] = number of ways to reach step i
    # dp[i] = dp[i-1] + dp[i-2]
    prev2, prev1 = 1, 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1


def climb_stairs_memoization(n: int) -> int:
    """
    VARIANT: Using memoization approach
    TIME: O(n), SPACE: O(n)
    """
    @lru_cache(maxsize=None)
    def dp(i):
        if i <= 2:
            return i
        return dp(i - 1) + dp(i - 2)
    
    return dp(n)


# =============================================================================
# PROBLEM 2: HOUSE ROBBER (MEDIUM) - 45 MIN
# =============================================================================

def rob(nums: List[int]) -> int:
    """
    PROBLEM: House Robber
    
    You are a professional robber planning to rob houses along a street. 
    Each house has a certain amount of money stashed, the only constraint 
    stopping you from robbing each of them is that adjacent houses have 
    security systems connected and it will automatically contact the police 
    if two adjacent houses were broken into on the same night.
    
    Example:
    Input: nums = [2,7,9,3,1]
    Output: 12 (rob house 0, 2, 4)
    
    TIME: O(n), SPACE: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    # dp[i] = max money robbed up to house i
    # dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, current
    
    return prev1


def rob_circular(nums: List[int]) -> int:
    """
    VARIANT: House Robber II (houses arranged in circle)
    
    Since houses are arranged in circle, first and last house are adjacent.
    
    TIME: O(n), SPACE: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    
    def rob_linear(houses):
        prev2, prev1 = 0, 0
        for money in houses:
            current = max(prev1, prev2 + money)
            prev2, prev1 = prev1, current
        return prev1
    
    # Case 1: Rob first house (can't rob last)
    # Case 2: Don't rob first house (can rob last)
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))


# =============================================================================
# PROBLEM 3: MAXIMUM SUBARRAY (MEDIUM) - 45 MIN
# =============================================================================

def max_subarray(nums: List[int]) -> int:
    """
    PROBLEM: Maximum Subarray (Kadane's Algorithm)
    
    Given an integer array nums, find the contiguous subarray (containing 
    at least one number) which has the largest sum and return its sum.
    
    Example:
    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6 ([4,-1,2,1])
    
    TIME: O(n), SPACE: O(1)
    """
    max_ending_here = max_so_far = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend existing subarray or start new one
        max_ending_here = max(nums[i], max_ending_here + nums[i])
        max_so_far = max(max_so_far, max_ending_here)
    
    return max_so_far


def max_subarray_with_indices(nums: List[int]) -> tuple:
    """
    VARIANT: Return max sum and the subarray indices
    """
    max_sum = current_sum = nums[0]
    start = end = temp_start = 0
    
    for i in range(1, len(nums)):
        if current_sum < 0:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end


# =============================================================================
# PROBLEM 4: COIN CHANGE (MEDIUM) - 60 MIN
# =============================================================================

def coin_change(coins: List[int], amount: int) -> int:
    """
    PROBLEM: Coin Change
    
    You are given an integer array coins representing coins of different 
    denominations and an integer amount representing a total amount of money.
    
    Return the fewest number of coins that you need to make up that amount.
    If that amount of money cannot be made up by any combination of the coins, 
    return -1.
    
    Example:
    Input: coins = [1,3,4], amount = 6
    Output: 2 ([3,3])
    
    TIME: O(amount * len(coins)), SPACE: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_combinations(coins: List[int], amount: int) -> int:
    """
    VARIANT: Coin Change II - Number of ways to make amount
    
    TIME: O(amount * len(coins)), SPACE: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]


# =============================================================================
# PROBLEM 5: LONGEST INCREASING SUBSEQUENCE (MEDIUM) - 60 MIN
# =============================================================================

def length_of_lis(nums: List[int]) -> int:
    """
    PROBLEM: Longest Increasing Subsequence
    
    Given an integer array nums, return the length of the longest strictly 
    increasing subsequence.
    
    Example:
    Input: nums = [10,9,2,5,3,7,101,18]
    Output: 4 ([2,3,7,18])
    
    TIME: O(n²), SPACE: O(n)
    """
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def length_of_lis_optimized(nums: List[int]) -> int:
    """
    OPTIMIZED: Using binary search
    TIME: O(n log n), SPACE: O(n)
    """
    import bisect
    
    tails = []
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day19_problems():
    """Test all Day 19 problems"""
    
    print("=" * 60)
    print("         DAY 19: 1D DYNAMIC PROGRAMMING")
    print("=" * 60)
    
    # Test Climbing Stairs
    print("\n🧪 Testing Climbing Stairs")
    stairs1 = climb_stairs(5)
    stairs2 = climb_stairs_memoization(5)
    print(f"Climb Stairs (n=5): {stairs1} (Expected: 8)")
    print(f"Climb Stairs Memo (n=5): {stairs2} (Expected: 8)")
    
    # Test House Robber
    print("\n🧪 Testing House Robber")
    rob1 = rob([2,7,9,3,1])
    print(f"House Robber [2,7,9,3,1]: {rob1} (Expected: 12)")
    
    rob2 = rob_circular([2,3,2])
    print(f"House Robber Circular [2,3,2]: {rob2} (Expected: 3)")
    
    # Test Maximum Subarray
    print("\n🧪 Testing Maximum Subarray")
    max_sub = max_subarray([-2,1,-3,4,-1,2,1,-5,4])
    print(f"Max Subarray: {max_sub} (Expected: 6)")
    
    max_sub_idx = max_subarray_with_indices([-2,1,-3,4,-1,2,1,-5,4])
    print(f"Max Subarray with indices: {max_sub_idx} (Expected: (6, 3, 6))")
    
    # Test Coin Change
    print("\n🧪 Testing Coin Change")
    coins1 = coin_change([1,3,4], 6)
    print(f"Coin Change [1,3,4], amount=6: {coins1} (Expected: 2)")
    
    coins2 = coin_change_combinations([1,2,5], 5)
    print(f"Coin Change Ways [1,2,5], amount=5: {coins2} (Expected: 4)")
    
    # Test Longest Increasing Subsequence
    print("\n🧪 Testing Longest Increasing Subsequence")
    lis1 = length_of_lis([10,9,2,5,3,7,101,18])
    print(f"LIS [10,9,2,5,3,7,101,18]: {lis1} (Expected: 4)")
    
    lis2 = length_of_lis_optimized([10,9,2,5,3,7,101,18])
    print(f"LIS Optimized: {lis2} (Expected: 4)")
    
    print("\n" + "=" * 60)
    print("           DAY 19 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 19 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review and optimize
    """
    
    print("🚀 Starting Day 19: 1D Dynamic Programming")
    print("\n📚 Theory Topics:")
    print("- DP problem identification")
    print("- Recurrence relation formulation")
    print("- Memoization vs tabulation")
    print("- Space optimization techniques")
    
    print("\n💻 Practice Problems:")
    print("1. Climbing Stairs (Easy) - 30 min")
    print("2. House Robber (Medium) - 45 min")
    print("3. Maximum Subarray (Medium) - 45 min")
    print("4. Coin Change (Medium) - 60 min")
    print("5. Longest Increasing Subsequence (Medium) - 60 min")
    
    print("\n🧪 Running Tests...")
    test_day19_problems()
    
    print("\n✅ Day 19 Complete!")
    print("📈 Next: Day 20 - 2D Dynamic Programming")


if __name__ == "__main__":
    main() 