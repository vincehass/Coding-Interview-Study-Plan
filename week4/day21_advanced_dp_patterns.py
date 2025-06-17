"""
=============================================================================
                        DAY 21: ADVANCED DP PATTERNS
                           Meta Interview Preparation
                              Week 4 - Day 21
=============================================================================

FOCUS: Complex DP patterns, optimization
TIME ALLOCATION: 4 hours
- Theory (1 hour): Advanced DP patterns and optimization techniques
- Problems (3 hours): Complex DP problems with multiple states

TOPICS COVERED:
- Interval DP and range problems
- State machine DP patterns
- Bitmask DP for subset problems
- Multi-dimensional state spaces

=============================================================================
"""

from typing import List, Dict, Tuple
from functools import lru_cache
import heapq


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
ADVANCED DP PATTERNS:

1. INTERVAL DP:
   - dp[i][j] represents optimal solution for range [i, j]
   - Often involves choosing split points within ranges
   - Examples: Matrix Chain Multiplication, Burst Balloons

2. STATE MACHINE DP:
   - Multiple states at each position
   - Transitions between states based on actions
   - Examples: Stock Trading, House Robber variants

3. BITMASK DP:
   - Use bitmasks to represent subsets
   - dp[mask] represents solution for subset represented by mask
   - Examples: Traveling Salesman, Assignment problems

4. MULTI-DIMENSIONAL STATES:
   - More than 2 dimensions in DP table
   - Each dimension represents different constraints
   - Examples: Knapsack variants, complex optimization
"""


# =============================================================================
# PROBLEM 1: PALINDROMIC SUBSTRINGS (MEDIUM) - 45 MIN
# =============================================================================

def count_substrings(s: str) -> int:
    """
    PROBLEM: Palindromic Substrings
    
    Given a string s, return the number of palindromic substrings in it.
    A string is a palindrome when it reads the same backward as forward.
    A substring is a contiguous sequence of characters within the string.
    
    CONSTRAINTS:
    - 1 <= s.length <= 1000
    - s consists of lowercase English letters
    
    EXAMPLES:
    Example 1:
        Input: s = "abc"
        Output: 3
        Explanation: Three palindromic strings: "a", "b", "c"
    
    Example 2:
        Input: s = "aaa"
        Output: 6
        Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa"
    
    EXPECTED TIME COMPLEXITY: O(n²)
    EXPECTED SPACE COMPLEXITY: O(1) - expand around centers approach
    
    GOAL: Master center expansion technique for palindrome problems
    """
    def expand_around_center(left: int, right: int) -> int:
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    total = 0
    for i in range(len(s)):
        # Odd length palindromes (center at i)
        total += expand_around_center(i, i)
        # Even length palindromes (center between i and i+1)
        total += expand_around_center(i, i + 1)
    
    return total


def longest_palindromic_substring(s: str) -> str:
    """
    VARIANT: Return the longest palindromic substring
    
    GOAL: Apply center expansion to find actual palindrome
    """
    def expand_around_center(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    for i in range(len(s)):
        # Check odd and even length palindromes
        palindrome1 = expand_around_center(i, i)
        palindrome2 = expand_around_center(i, i + 1)
        
        current_longest = palindrome1 if len(palindrome1) > len(palindrome2) else palindrome2
        if len(current_longest) > len(longest):
            longest = current_longest
    
    return longest


# =============================================================================
# PROBLEM 2: DECODE WAYS (MEDIUM) - 60 MIN
# =============================================================================

def num_decodings(s: str) -> int:
    """
    PROBLEM: Decode Ways
    
    A message containing letters from A-Z can be encoded into numbers using 
    the following mapping:
    'A' -> "1", 'B' -> "2", ..., 'Z' -> "26"
    
    To decode an encoded message, all the digits must be grouped then mapped 
    back into letters using the reverse of the mapping above. Given a string s 
    containing only digits, return the number of ways to decode it.
    
    CONSTRAINTS:
    - 1 <= s.length <= 100
    - s contains only digits and may contain leading zero(s)
    
    EXAMPLES:
    Example 1:
        Input: s = "12"
        Output: 2
        Explanation: "12" could be decoded as "AB" (1 2) or "L" (12)
    
    Example 2:
        Input: s = "226"
        Output: 3
        Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6)
    
    Example 3:
        Input: s = "06"
        Output: 0
        Explanation: "06" cannot be mapped to "F" because of the leading zero
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    GOAL: Handle constraint-based DP with validation
    """
    if not s or s[0] == '0':
        return 0
    
    # dp[i] represents number of ways to decode s[:i]
    prev2 = prev1 = 1
    
    for i in range(1, len(s)):
        current = 0
        
        # Single digit decode
        if s[i] != '0':
            current += prev1
        
        # Two digit decode
        two_digit = int(s[i-1:i+1])
        if 10 <= two_digit <= 26:
            current += prev2
        
        prev2, prev1 = prev1, current
    
    return prev1


# =============================================================================
# PROBLEM 3: BEST TIME TO BUY AND SELL STOCK WITH COOLDOWN (MEDIUM) - 75 MIN
# =============================================================================

def max_profit_with_cooldown(prices: List[int]) -> int:
    """
    PROBLEM: Best Time to Buy and Sell Stock with Cooldown
    
    You are given an array prices where prices[i] is the price of a given 
    stock on the ith day. Find the maximum profit you can achieve with the 
    following restrictions:
    - After you sell your stock, you cannot buy stock on the next day (cooldown)
    - You may complete as many transactions as you like
    
    CONSTRAINTS:
    - 1 <= prices.length <= 5000
    - 0 <= prices[i] <= 1000
    
    EXAMPLES:
    Example 1:
        Input: prices = [1,2,3,0,2]
        Output: 3
        Explanation: Buy on day 0 (price = 1), sell on day 1 (price = 2), 
        cooldown on day 2, buy on day 3 (price = 0), sell on day 4 (price = 2)
        Total profit = 2 - 1 + 2 - 0 = 3
    
    Example 2:
        Input: prices = [1]
        Output: 0
        Explanation: Cannot make profit with single price
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    GOAL: Master state machine DP with multiple states
    """
    if len(prices) <= 1:
        return 0
    
    # Three states: hold stock, sold stock (cooldown), no stock (can buy)
    hold = -prices[0]  # Bought stock on day 0
    sold = 0           # No transactions yet
    rest = 0           # No stock, can buy
    
    for i in range(1, len(prices)):
        prev_hold, prev_sold, prev_rest = hold, sold, rest
        
        # Hold: either already holding or buy today
        hold = max(prev_hold, prev_rest - prices[i])
        
        # Sold: sell the stock we were holding
        sold = prev_hold + prices[i]
        
        # Rest: either already resting or finished cooldown
        rest = max(prev_rest, prev_sold)
    
    # At the end, we want to have no stock (either sold or rest)
    return max(sold, rest)


# =============================================================================
# PROBLEM 4: BURST BALLOONS (HARD) - 90 MIN
# =============================================================================

def max_coins(nums: List[int]) -> int:
    """
    PROBLEM: Burst Balloons
    
    You are given n balloons, indexed from 0 to n - 1. Each balloon is painted 
    with a number on it represented by an array nums. You are asked to burst 
    all the balloons.
    
    If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] 
    coins. If i - 1 or i + 1 goes out of bounds of the array, then treat it as 
    if there is a balloon with a 1 painted on it.
    
    Return the maximum coins you can collect by bursting the balloons wisely.
    
    CONSTRAINTS:
    - n == nums.length
    - 1 <= n <= 300
    - 0 <= nums[i] <= 100
    
    EXAMPLES:
    Example 1:
        Input: nums = [3,1,5,8]
        Output: 167
        Explanation: 
        Burst balloon 1: [3,5,8] + 3*1*5 = 15
        Burst balloon 2: [3,8] + 3*5*8 = 120  
        Burst balloon 0: [8] + 1*3*8 = 24
        Burst balloon 3: [] + 1*8*1 = 8
        Total = 15 + 120 + 24 + 8 = 167
    
    Example 2:
        Input: nums = [1,5]
        Output: 10
        Explanation: Burst balloon 0 then 1, or 1 then 0
    
    EXPECTED TIME COMPLEXITY: O(n³)
    EXPECTED SPACE COMPLEXITY: O(n²)
    
    GOAL: Master interval DP with complex state transitions
    """
    # Add boundary balloons with value 1
    balloons = [1] + nums + [1]
    n = len(balloons)
    
    # dp[i][j] = max coins from bursting balloons between i and j (exclusive)
    dp = [[0] * n for _ in range(n)]
    
    # Length of interval (at least 3 to have balloons between boundaries)
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Try bursting each balloon k between i and j as the last one
            for k in range(i + 1, j):
                coins = balloons[i] * balloons[k] * balloons[j]
                total = dp[i][k] + coins + dp[k][j]
                dp[i][j] = max(dp[i][j], total)
    
    return dp[0][n-1]


# =============================================================================
# PROBLEM 5: REGULAR EXPRESSION MATCHING (HARD) - 90 MIN
# =============================================================================

def is_match(s: str, p: str) -> bool:
    """
    PROBLEM: Regular Expression Matching
    
    Given an input string s and a pattern p, implement regular expression 
    matching with support for '.' and '*' where:
    - '.' Matches any single character
    - '*' Matches zero or more of the preceding element
    
    The matching should cover the entire input string (not partial).
    
    CONSTRAINTS:
    - 1 <= s.length <= 20
    - 1 <= p.length <= 30
    - s contains only lowercase English letters
    - p contains only lowercase English letters, '.', and '*'
    - It is guaranteed for each appearance of '*', there will be a previous valid character
    
    EXAMPLES:
    Example 1:
        Input: s = "aa", p = "a"
        Output: false
        Explanation: "a" does not match the entire string "aa"
    
    Example 2:
        Input: s = "aa", p = "a*"
        Output: true
        Explanation: '*' means zero or more of the preceding element, 'a'
    
    Example 3:
        Input: s = "ab", p = ".*"
        Output: true
        Explanation: ".*" means "zero or more (*) of any character (.)"
    
    EXPECTED TIME COMPLEXITY: O(m * n)
    EXPECTED SPACE COMPLEXITY: O(m * n)
    
    GOAL: Handle complex pattern matching with multiple cases
    """
    m, n = len(s), len(p)
    
    # dp[i][j] = True if s[:i] matches p[:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty string matches empty pattern
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, a*b*c* that can match empty string
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # Two cases for '*':
                # 1. Match zero occurrences of preceding character
                dp[i][j] = dp[i][j-2]
                
                # 2. Match one or more occurrences (if current chars match)
                if p[j-2] == s[i-1] or p[j-2] == '.':
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            else:
                # Direct character match or '.' wildcard
                if p[j-1] == s[i-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day21_problems():
    """Test all Day 21 problems"""
    
    print("=" * 60)
    print("         DAY 21: ADVANCED DP PATTERNS")
    print("=" * 60)
    
    # Test Palindromic Substrings
    print("\n🧪 Testing Palindromic Substrings")
    palindromes1 = count_substrings("abc")
    print(f"Palindromic Substrings 'abc': {palindromes1} (Expected: 3)")
    
    palindromes2 = count_substrings("aaa")
    print(f"Palindromic Substrings 'aaa': {palindromes2} (Expected: 6)")
    
    longest_pal = longest_palindromic_substring("babad")
    print(f"Longest Palindrome 'babad': '{longest_pal}' (Expected: 'bab' or 'aba')")
    
    # Test Decode Ways
    print("\n🧪 Testing Decode Ways")
    decode1 = num_decodings("12")
    print(f"Decode Ways '12': {decode1} (Expected: 2)")
    
    decode2 = num_decodings("226")
    print(f"Decode Ways '226': {decode2} (Expected: 3)")
    
    decode3 = num_decodings("06")
    print(f"Decode Ways '06': {decode3} (Expected: 0)")
    
    # Test Stock with Cooldown
    print("\n🧪 Testing Stock with Cooldown")
    profit1 = max_profit_with_cooldown([1,2,3,0,2])
    print(f"Max Profit with Cooldown [1,2,3,0,2]: {profit1} (Expected: 3)")
    
    profit2 = max_profit_with_cooldown([1])
    print(f"Max Profit with Cooldown [1]: {profit2} (Expected: 0)")
    
    # Test Burst Balloons
    print("\n🧪 Testing Burst Balloons")
    coins1 = max_coins([3,1,5,8])
    print(f"Max Coins [3,1,5,8]: {coins1} (Expected: 167)")
    
    coins2 = max_coins([1,5])
    print(f"Max Coins [1,5]: {coins2} (Expected: 10)")
    
    # Test Regular Expression Matching
    print("\n🧪 Testing Regular Expression Matching")
    match1 = is_match("aa", "a")
    print(f"RegEx Match ('aa', 'a'): {match1} (Expected: False)")
    
    match2 = is_match("aa", "a*")
    print(f"RegEx Match ('aa', 'a*'): {match2} (Expected: True)")
    
    match3 = is_match("ab", ".*")
    print(f"RegEx Match ('ab', '.*'): {match3} (Expected: True)")
    
    print("\n" + "=" * 60)
    print("           DAY 21 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 21 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review and optimize
    """
    
    print("🚀 Starting Day 21: Advanced DP Patterns")
    print("\n📚 Theory Topics:")
    print("- Interval DP and range problems")
    print("- State machine DP patterns")
    print("- Complex pattern matching")
    print("- Multi-dimensional state optimization")
    
    print("\n💻 Practice Problems:")
    print("1. Palindromic Substrings (Medium) - 45 min")
    print("2. Decode Ways (Medium) - 60 min")
    print("3. Stock with Cooldown (Medium) - 75 min")
    print("4. Burst Balloons (Hard) - 90 min")
    print("5. Regular Expression Matching (Hard) - 90 min")
    
    print("\n🧪 Running Tests...")
    test_day21_problems()
    
    print("\n✅ Day 21 Complete!")
    print("📈 Next: Day 22 - DP Optimization Techniques")


if __name__ == "__main__":
    main() 