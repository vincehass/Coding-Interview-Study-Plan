"""
=============================================================================
                        WEEK 4 SOLUTION FILE
                     COMPLETE SOLUTIONS & VARIANTS
                           Meta Interview Preparation
=============================================================================

TOPICS COVERED:
- 1D Dynamic Programming
- 2D Dynamic Programming
- Advanced DP Patterns
- Optimization Techniques

=============================================================================
"""

from typing import List, Dict, Set, Tuple
from functools import lru_cache


# =============================================================================
# 1D DYNAMIC PROGRAMMING SOLUTIONS
# =============================================================================

def climbing_stairs(n: int) -> int:
    """
    PROBLEM: Climbing Stairs
    TIME: O(n), SPACE: O(1)
    """
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

def climbing_stairs_k_steps(n: int, k: int) -> int:
    """
    VARIANT: Climbing Stairs with K Steps
    TIME: O(n * k), SPACE: O(n)
    """
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i] += dp[i - j]
    
    return dp[n]

def house_robber(nums: List[int]) -> int:
    """
    PROBLEM: House Robber
    TIME: O(n), SPACE: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, current
    
    return prev1

def house_robber_ii(nums: List[int]) -> int:
    """
    VARIANT: House Robber II (Circular)
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
        for house in houses:
            current = max(prev1, prev2 + house)
            prev2, prev1 = prev1, current
        return prev1
    
    # Case 1: rob houses[0] to houses[n-2]
    # Case 2: rob houses[1] to houses[n-1]
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))

def decode_ways(s: str) -> int:
    """
    PROBLEM: Decode Ways
    TIME: O(n), SPACE: O(1)
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    prev2, prev1 = 1, 1
    
    for i in range(1, n):
        current = 0
        
        # Single digit
        if s[i] != '0':
            current += prev1
        
        # Two digits
        two_digit = int(s[i-1:i+1])
        if 10 <= two_digit <= 26:
            current += prev2
        
        prev2, prev1 = prev1, current
    
    return prev1

def coin_change(coins: List[int], amount: int) -> int:
    """
    PROBLEM: Coin Change
    TIME: O(amount * coins), SPACE: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def coin_change_ii(amount: int, coins: List[int]) -> int:
    """
    VARIANT: Coin Change II (Number of combinations)
    TIME: O(amount * coins), SPACE: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]

def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    PROBLEM: Longest Increasing Subsequence
    TIME: O(n log n), SPACE: O(n)
    """
    if not nums:
        return 0
    
    tails = []
    
    for num in nums:
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)

def longest_increasing_subsequence_dp(nums: List[int]) -> int:
    """
    VARIANT: LIS with DP approach
    TIME: O(n²), SPACE: O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# =============================================================================
# 2D DYNAMIC PROGRAMMING SOLUTIONS
# =============================================================================

def unique_paths(m: int, n: int) -> int:
    """
    PROBLEM: Unique Paths
    TIME: O(m * n), SPACE: O(n)
    """
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]

def unique_paths_with_obstacles(obstacleGrid: List[List[int]]) -> int:
    """
    VARIANT: Unique Paths II
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
                dp[j] += dp[j - 1]
    
    return dp[n - 1]

def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    PROBLEM: Longest Common Subsequence
    TIME: O(m * n), SPACE: O(min(m, n))
    """
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    
    return prev[n]

def edit_distance(word1: str, word2: str) -> int:
    """
    PROBLEM: Edit Distance
    TIME: O(m * n), SPACE: O(min(m, n))
    """
    m, n = len(word1), len(word2)
    
    if m < n:
        word1, word2 = word2, word1
        m, n = n, m
    
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    
    return prev[n]

def minimum_path_sum(grid: List[List[int]]) -> int:
    """
    VARIANT: Minimum Path Sum
    TIME: O(m * n), SPACE: O(n)
    """
    m, n = len(grid), len(grid[0])
    dp = [float('inf')] * n
    dp[0] = 0
    
    for i in range(m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = min(dp[j], dp[j - 1]) + grid[i][j]
    
    return dp[n - 1]

def maximal_square(matrix: List[List[str]]) -> int:
    """
    VARIANT: Maximal Square
    TIME: O(m * n), SPACE: O(n)
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    dp = [0] * (n + 1)
    max_side = 0
    prev = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = dp[j]
            if matrix[i - 1][j - 1] == '1':
                dp[j] = min(dp[j - 1], dp[j], prev) + 1
                max_side = max(max_side, dp[j])
            else:
                dp[j] = 0
            prev = temp
    
    return max_side * max_side

# =============================================================================
# ADVANCED DP PATTERNS
# =============================================================================

def word_break(s: str, wordDict: List[str]) -> bool:
    """
    PROBLEM: Word Break
    TIME: O(n²), SPACE: O(n)
    """
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[len(s)]

def word_break_ii(s: str, wordDict: List[str]) -> List[str]:
    """
    VARIANT: Word Break II
    TIME: O(2^n), SPACE: O(2^n)
    """
    word_set = set(wordDict)
    
    @lru_cache(maxsize=None)
    def backtrack(start):
        if start == len(s):
            return [""]
        
        result = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for rest in backtrack(end):
                    if rest:
                        result.append(word + " " + rest)
                    else:
                        result.append(word)
        
        return result
    
    return backtrack(0)

def maximum_subarray(nums: List[int]) -> int:
    """
    PROBLEM: Maximum Subarray (Kadane's Algorithm)
    TIME: O(n), SPACE: O(1)
    """
    max_sum = current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def maximum_product_subarray(nums: List[int]) -> int:
    """
    VARIANT: Maximum Product Subarray
    TIME: O(n), SPACE: O(1)
    """
    max_prod = min_prod = result = nums[0]
    
    for i in range(1, len(nums)):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(nums[i], max_prod * nums[i])
        min_prod = min(nums[i], min_prod * nums[i])
        
        result = max(result, max_prod)
    
    return result

def palindromic_substrings(s: str) -> int:
    """
    PROBLEM: Palindromic Substrings
    TIME: O(n²), SPACE: O(1)
    """
    def expand_around_centers(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    total = 0
    for i in range(len(s)):
        # Odd length palindromes
        total += expand_around_centers(i, i)
        # Even length palindromes
        total += expand_around_centers(i, i + 1)
    
    return total

def longest_palindromic_substring(s: str) -> str:
    """
    VARIANT: Longest Palindromic Substring
    TIME: O(n²), SPACE: O(1)
    """
    if not s:
        return ""
    
    start = 0
    max_len = 1
    
    def expand_around_centers(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    for i in range(len(s)):
        # Odd length
        len1 = expand_around_centers(i, i)
        # Even length
        len2 = expand_around_centers(i, i + 1)
        
        curr_max = max(len1, len2)
        if curr_max > max_len:
            max_len = curr_max
            start = i - (curr_max - 1) // 2
    
    return s[start:start + max_len]

def target_sum(nums: List[int], target: int) -> int:
    """
    VARIANT: Target Sum (0/1 Knapsack variant)
    TIME: O(n * sum), SPACE: O(sum)
    """
    total = sum(nums)
    if target > total or target < -total or (target + total) % 2 == 1:
        return 0
    
    # Transform to subset sum problem
    subset_sum = (target + total) // 2
    
    dp = [0] * (subset_sum + 1)
    dp[0] = 1
    
    for num in nums:
        for j in range(subset_sum, num - 1, -1):
            dp[j] += dp[j - num]
    
    return dp[subset_sum]

def partition_equal_subset_sum(nums: List[int]) -> bool:
    """
    VARIANT: Partition Equal Subset Sum
    TIME: O(n * sum), SPACE: O(sum)
    """
    total = sum(nums)
    if total % 2 == 1:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]

# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def test_all_week4_solutions():
    """Comprehensive test suite for all Week 4 solutions"""
    
    print("=" * 80)
    print("                    WEEK 4 COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Test 1D DP
    print("\n🧪 TESTING 1D DYNAMIC PROGRAMMING")
    print("-" * 50)
    
    # Climbing Stairs
    stairs_tests = [(2, 2), (3, 3), (4, 5), (5, 8)]
    for n, expected in stairs_tests:
        result = climbing_stairs(n)
        print(f"Climbing Stairs {n}: {result} (Expected: {expected})")
    
    # House Robber
    robber_tests = [
        ([1,2,3,1], 4),
        ([2,7,9,3,1], 12),
        ([2,1,1,2], 4)
    ]
    for nums, expected in robber_tests:
        result = house_robber(nums)
        circular_result = house_robber_ii(nums)
        print(f"House Robber {nums}: {result} (Expected: {expected})")
        print(f"House Robber II {nums}: {circular_result}")
    
    # Decode Ways
    decode_tests = [("12", 2), ("226", 3), ("06", 0), ("10", 1)]
    for s, expected in decode_tests:
        result = decode_ways(s)
        print(f"Decode Ways '{s}': {result} (Expected: {expected})")
    
    # Coin Change
    coin_tests = [
        ([1,3,4], 6, 2),
        ([2], 3, -1),
        ([1], 0, 0)
    ]
    for coins, amount, expected in coin_tests:
        result = coin_change(coins, amount)
        combinations = coin_change_ii(amount, coins)
        print(f"Coin Change {coins}, {amount}: {result} (Expected: {expected})")
        print(f"Coin Change II combinations: {combinations}")
    
    # Test 2D DP
    print("\n🧪 TESTING 2D DYNAMIC PROGRAMMING")
    print("-" * 50)
    
    # Unique Paths
    paths_tests = [(3, 7, 28), (3, 2, 3), (7, 3, 28)]
    for m, n, expected in paths_tests:
        result = unique_paths(m, n)
        print(f"Unique Paths {m}x{n}: {result} (Expected: {expected})")
    
    # LCS
    lcs_tests = [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0)
    ]
    for text1, text2, expected in lcs_tests:
        result = longest_common_subsequence(text1, text2)
        print(f"LCS '{text1}', '{text2}': {result} (Expected: {expected})")
    
    # Edit Distance
    edit_tests = [
        ("horse", "ros", 3),
        ("intention", "execution", 5)
    ]
    for word1, word2, expected in edit_tests:
        result = edit_distance(word1, word2)
        print(f"Edit Distance '{word1}' -> '{word2}': {result} (Expected: {expected})")
    
    # Advanced DP
    print("\n🧪 TESTING ADVANCED DP")
    print("-" * 50)
    
    # Word Break
    wb_tests = [
        ("leetcode", ["leet","code"], True),
        ("applepenapple", ["apple","pen"], True),
        ("catsandog", ["cats","dog","sand","and","cat"], False)
    ]
    for s, wordDict, expected in wb_tests:
        result = word_break(s, wordDict)
        print(f"Word Break '{s}': {result} (Expected: {expected})")
    
    # Maximum Subarray
    subarray_tests = [
        ([-2,1,-3,4,-1,2,1,-5,4], 6),
        ([1], 1),
        ([5,4,-1,7,8], 23)
    ]
    for nums, expected in subarray_tests:
        result = maximum_subarray(nums)
        product_result = maximum_product_subarray(nums)
        print(f"Max Subarray {nums}: {result} (Expected: {expected})")
        print(f"Max Product Subarray: {product_result}")
    
    # Palindromic Substrings
    palindrome_tests = [("abc", 3), ("aaa", 6), ("", 0)]
    for s, expected in palindrome_tests:
        result = palindromic_substrings(s)
        longest = longest_palindromic_substring(s)
        print(f"Palindromic Substrings '{s}': {result} (Expected: {expected})")
        print(f"Longest Palindromic Substring: '{longest}'")
    
    print("\n" + "=" * 80)
    print("                    TESTING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_all_week4_solutions() 