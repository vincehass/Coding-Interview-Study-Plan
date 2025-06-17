"""
=============================================================================
                        WEEK 4 PRACTICE PROBLEMS
                    DYNAMIC PROGRAMMING & SYSTEM DESIGN
                           Meta Interview Preparation
=============================================================================

This file contains practice problems for Week 4. Work through these problems
independently to reinforce your learning from the main study materials.

INSTRUCTIONS:
1. Read each problem statement and constraints carefully
2. Understand the examples and expected outputs
3. Write your solution in the designated space
4. Test your solution with the provided test cases
5. Compare with the reference implementation when stuck

=============================================================================
"""

from typing import List, Dict, Optional
from collections import defaultdict


# 1D DYNAMIC PROGRAMMING

def house_robber_practice(nums: List[int]) -> int:
    """
    PROBLEM: House Robber
    
    DESCRIPTION:
    You are a professional robber planning to rob houses along a street. Each house has 
    a certain amount of money stashed, the only constraint stopping you from robbing each 
    of them is that adjacent houses have security systems connected and it will automatically 
    contact the police if two adjacent houses were broken into on the same night.
    
    Given an integer array nums representing the amount of money of each house, return the 
    maximum amount of money you can rob tonight without alerting the police.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 100
    - 0 <= nums[i] <= 400
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,2,3,1]
        Output: 4
        Explanation: Rob house 1 (money = 1) then rob house 3 (money = 3).
        Total amount you can rob = 1 + 3 = 4.
    
    Example 2:
        Input: nums = [2,7,9,3,1]
        Output: 12
        Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
        Total amount you can rob = 2 + 9 + 1 = 12.
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    # dp[i] = max money that can be robbed up to house i
    # dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    prev2 = nums[0]  # max money up to house 0
    prev1 = max(nums[0], nums[1])  # max money up to house 1
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current
    
    return prev1


def decode_ways_practice(s: str) -> int:
    """
    PROBLEM: Decode Ways
    
    DESCRIPTION:
    A message containing letters from A-Z can be encoded into numbers using the following mapping:
    'A' -> "1", 'B' -> "2", ..., 'Z' -> "26"
    
    To decode an encoded message, all the digits must be grouped then mapped back into letters 
    using the reverse of the mapping above (there may be multiple ways). For example, "11106" 
    can be mapped into: "AAJF" with the grouping (1 1 10 6) or "KJF" with the grouping (11 10 6).
    
    Given a string s containing only digits, return the number of ways to decode it.
    
    CONSTRAINTS:
    - 1 <= s.length <= 100
    - s contains only digits and may contain leading zeros.
    
    EXAMPLES:
    Example 1:
        Input: s = "12"
        Output: 2
        Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
    
    Example 2:
        Input: s = "226"
        Output: 3
        Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    if n == 1:
        return 1
    
    # dp[i] = number of ways to decode s[:i]
    prev2 = 1  # dp[0] = 1
    prev1 = 1  # dp[1] = 1 if s[0] != '0'
    
    for i in range(2, n + 1):
        current = 0
        
        # Single digit
        if s[i-1] != '0':
            current += prev1
        
        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            current += prev2
        
        prev2 = prev1
        prev1 = current
    
    return prev1


def coin_change_practice(coins: List[int], amount: int) -> int:
    """
    PROBLEM: Coin Change
    
    DESCRIPTION:
    You are given an integer array coins representing coins of different denominations and 
    an integer amount representing a total amount of money.
    
    Return the fewest number of coins that you need to make up that amount. If that amount 
    of money cannot be made up by any combination of the coins, return -1.
    
    You may assume that you have an infinite number of each kind of coin.
    
    CONSTRAINTS:
    - 1 <= coins.length <= 12
    - 1 <= coins[i] <= 2^31 - 1
    - 0 <= amount <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: coins = [1,3,4], amount = 6
        Output: 2
        Explanation: 6 = 3 + 3
    
    Example 2:
        Input: coins = [2], amount = 3
        Output: -1
    
    EXPECTED TIME COMPLEXITY: O(amount * coins)
    EXPECTED SPACE COMPLEXITY: O(amount)
    
    YOUR SOLUTION:
    """
    # dp[i] = minimum coins needed to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def longest_increasing_subsequence_practice(nums: List[int]) -> int:
    """
    PROBLEM: Longest Increasing Subsequence
    
    DESCRIPTION:
    Given an integer array nums, return the length of the longest strictly increasing subsequence.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 2500
    - -10^4 <= nums[i] <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: nums = [10,9,2,5,3,7,101,18]
        Output: 4
        Explanation: The longest increasing subsequence is [2,3,7,18], therefore the length is 4.
    
    Example 2:
        Input: nums = [0,1,0,3,2,3]
        Output: 4
    
    EXPECTED TIME COMPLEXITY: O(n²) basic DP, O(n log n) with binary search
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """
    if not nums:
        return 0
    
    # dp[i] = length of LIS ending at index i
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


# 2D DYNAMIC PROGRAMMING

def unique_paths_practice(m: int, n: int) -> int:
    """
    PROBLEM: Unique Paths
    
    DESCRIPTION:
    There is a robot on an m x n grid. The robot is initially located at the top-left 
    corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner 
    (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.
    
    Given the two integers m and n, return the number of possible unique paths that the 
    robot can take to reach the bottom-right corner.
    
    CONSTRAINTS:
    - 1 <= m, n <= 100
    
    EXAMPLES:
    Example 1:
        Input: m = 3, n = 7
        Output: 28
    
    Example 2:
        Input: m = 3, n = 2
        Output: 3
        Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
        1. Right -> Down -> Down
        2. Down -> Right -> Down
        3. Down -> Down -> Right
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(n) optimized
    
    YOUR SOLUTION:
    """
    # Space-optimized solution using 1D array
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]


def longest_common_subsequence_practice(text1: str, text2: str) -> int:
    """
    PROBLEM: Longest Common Subsequence
    
    DESCRIPTION:
    Given two strings text1 and text2, return the length of their longest common subsequence. 
    If there is no common subsequence, return 0.
    
    A subsequence of a string is a new string generated from the original string with some 
    characters (can be none) deleted without changing the relative order of the remaining characters.
    
    A common subsequence of two strings is a subsequence that is common to both strings.
    
    CONSTRAINTS:
    - 1 <= text1.length, text2.length <= 1000
    - text1 and text2 consist of only lowercase English characters.
    
    EXAMPLES:
    Example 1:
        Input: text1 = "abcde", text2 = "ace" 
        Output: 3  
        Explanation: The longest common subsequence is "ace" and its length is 3.
    
    Example 2:
        Input: text1 = "abc", text2 = "abc"
        Output: 3
        Explanation: The longest common subsequence is "abc" and its length is 3.
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(m * n)
    
    YOUR SOLUTION:
    """
    m, n = len(text1), len(text2)
    
    # dp[i][j] = LCS length of text1[:i] and text2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def edit_distance_practice(word1: str, word2: str) -> int:
    """
    PROBLEM: Edit Distance
    
    DESCRIPTION:
    Given two strings word1 and word2, return the minimum number of operations required 
    to convert word1 to word2.
    
    You have the following three operations permitted on a word:
    - Insert a character
    - Delete a character
    - Replace a character
    
    CONSTRAINTS:
    - 0 <= word1.length, word2.length <= 500
    - word1 and word2 consist of lowercase English letters.
    
    EXAMPLES:
    Example 1:
        Input: word1 = "horse", word2 = "ros"
        Output: 3
        Explanation: 
        horse -> rorse (replace 'h' with 'r')
        rorse -> rose (remove 'r')
        rose -> ros (remove 'e')
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(m * n)
    
    YOUR SOLUTION:
    """
    m, n = len(word1), len(word2)
    
    # dp[i][j] = min operations to convert word1[:i] to word2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
    
    return dp[m][n]


# ADVANCED DYNAMIC PROGRAMMING

def word_break_practice(s: str, wordDict: List[str]) -> bool:
    """
    PROBLEM: Word Break
    
    DESCRIPTION:
    Given a string s and a dictionary of strings wordDict, return true if s can be 
    segmented into a space-separated sequence of one or more dictionary words.
    
    Note that the same word in the dictionary may be reused multiple times in the segmentation.
    
    CONSTRAINTS:
    - 1 <= s.length <= 300
    - 1 <= wordDict.length <= 1000
    - 1 <= wordDict[i].length <= 20
    - s and wordDict[i] consist of only lowercase English letters.
    - All the strings of wordDict are unique.
    
    EXAMPLES:
    Example 1:
        Input: s = "leetcode", wordDict = ["leet","code"]
        Output: true
        Explanation: Return true because "leetcode" can be segmented as "leet code".
    
    Example 2:
        Input: s = "applepenapple", wordDict = ["apple","pen"]
        Output: true
        Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
        Note that you are allowed to reuse a dictionary word.
    
    EXPECTED TIME COMPLEXITY: O(n³)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True  # Empty string can always be segmented
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[len(s)]


def maximum_subarray_practice(nums: List[int]) -> int:
    """
    PROBLEM: Maximum Subarray (Kadane's Algorithm)
    
    DESCRIPTION:
    Given an integer array nums, find the contiguous subarray (containing at least one number) 
    which has the largest sum and return its sum.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
        Output: 6
        Explanation: [4,-1,2,1] has the largest sum = 6.
    
    Example 2:
        Input: nums = [1]
        Output: 1
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
    """
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend the existing subarray or start a new one
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum


def palindromic_substrings_practice(s: str) -> int:
    """
    PROBLEM: Palindromic Substrings
    
    DESCRIPTION:
    Given a string s, return the number of palindromic substrings in it.
    A string is a palindrome when it reads the same backward as forward.
    A substring is a contiguous sequence of characters within the string.
    
    CONSTRAINTS:
    - 1 <= s.length <= 1000
    - s consists of lowercase English letters.
    
    EXAMPLES:
    Example 1:
        Input: s = "abc"
        Output: 3
        Explanation: Three palindromic strings: "a", "b", "c".
    
    Example 2:
        Input: s = "aaa"
        Output: 6
        Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
    
    EXPECTED TIME COMPLEXITY: O(n²)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
    """
    def expand_around_center(left: int, right: int) -> int:
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    total_count = 0
    
    for i in range(len(s)):
        # Odd length palindromes (center at i)
        total_count += expand_around_center(i, i)
        
        # Even length palindromes (center between i and i+1)
        total_count += expand_around_center(i, i + 1)
    
    return total_count


# TEST FUNCTIONS

def test_week4_practice():
    """Test your solutions with comprehensive test cases"""
    
    print("=== TESTING WEEK 4 PRACTICE SOLUTIONS ===\n")
    
    # Test House Robber
    print("1. House Robber:")
    test_cases = [
        ([1,2,3,1], 4),
        ([2,7,9,3,1], 12),
        ([5], 5),
        ([1,2], 2)
    ]
    for nums, expected in test_cases:
        result = house_robber_practice(nums)
        print(f"   Input: {nums}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Coin Change
    print("2. Coin Change:")
    test_cases = [
        ([1,3,4], 6, 2),
        ([2], 3, -1),
        ([1], 0, 0)
    ]
    for coins, amount, expected in test_cases:
        result = coin_change_practice(coins, amount)
        print(f"   Coins: {coins}, Amount: {amount}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Longest Increasing Subsequence
    print("3. Longest Increasing Subsequence:")
    test_cases = [
        ([10,9,2,5,3,7,101,18], 4),
        ([0,1,0,3,2,3], 4),
        ([7,7,7,7,7,7,7], 1)
    ]
    for nums, expected in test_cases:
        result = longest_increasing_subsequence_practice(nums)
        print(f"   Input: {nums}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Unique Paths
    print("4. Unique Paths:")
    test_cases = [
        (3, 7, 28),
        (3, 2, 3),
        (1, 1, 1)
    ]
    for m, n, expected in test_cases:
        result = unique_paths_practice(m, n)
        print(f"   Grid: {m}x{n}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Longest Common Subsequence
    print("5. Longest Common Subsequence:")
    test_cases = [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0)
    ]
    for text1, text2, expected in test_cases:
        result = longest_common_subsequence_practice(text1, text2)
        print(f"   Text1: '{text1}', Text2: '{text2}'")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Maximum Subarray
    print("6. Maximum Subarray:")
    test_cases = [
        ([-2,1,-3,4,-1,2,1,-5,4], 6),
        ([1], 1),
        ([5,4,-1,7,8], 23)
    ]
    for nums, expected in test_cases:
        result = maximum_subarray_practice(nums)
        print(f"   Input: {nums}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    print("Continue implementing and testing other problems...")


if __name__ == "__main__":
    print("Week 4 Practice Problems")
    print("========================")
    print("Topics: Dynamic Programming & System Design")
    print("- 1D Dynamic Programming")
    print("- 2D Dynamic Programming")
    print("- Advanced DP Patterns")
    print("- Optimization Techniques")
    print()
    
    # Uncomment to run tests
    # test_week4_practice() 