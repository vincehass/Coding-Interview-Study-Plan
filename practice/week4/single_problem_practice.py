"""
=============================================================================
                        WEEK 4 SINGLE PROBLEM PRACTICE
                            CLIMBING STAIRS
                           Meta Interview Preparation
=============================================================================

Focus on mastering one core dynamic programming problem with comprehensive testing.
This represents the most fundamental DP pattern in interviews.

=============================================================================
"""

def climb_stairs(n: int) -> int:
    """
    PROBLEM: Climbing Stairs
    
    DESCRIPTION:
    You are climbing a staircase. It takes n steps to reach the top.
    
    Each time you can either climb 1 or 2 steps. In how many distinct ways 
    can you climb to the top?
    
    CONSTRAINTS:
    - 1 <= n <= 45
    
    EXAMPLES:
    Example 1:
        Input: n = 2
        Output: 2
        Explanation: There are two ways to climb to the top.
        1. 1 step + 1 step
        2. 2 steps
    
    Example 2:
        Input: n = 3
        Output: 3
        Explanation: There are three ways to climb to the top.
        1. 1 step + 1 step + 1 step
        2. 1 step + 2 steps
        3. 2 steps + 1 step
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1) optimized, O(n) with memoization
    
    Args:
        n (int): Number of steps in the staircase
        
    Returns:
        int: Number of distinct ways to climb to the top
    """
    # Base cases
    if n <= 2:
        return n
    
    # Dynamic programming approach with O(1) space
    # dp[i] = dp[i-1] + dp[i-2] (Fibonacci pattern)
    prev2 = 1  # ways to reach step 1
    prev1 = 2  # ways to reach step 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def main():
    """Test the climb_stairs function with various test cases"""
    
    print("=" * 60)
    print("           WEEK 4 SINGLE PROBLEM PRACTICE")
    print("                CLIMBING STAIRS")
    print("=" * 60)
    
    # Test Case 1: Example 1 from problem description
    print("\n🧪 Test Case 1: n = 2")
    n1 = 2
    expected1 = 2
    result1 = climb_stairs(n1)
    
    print(f"Input: n = {n1}")
    print(f"Ways to climb:")
    print(f"  1. 1 step + 1 step")
    print(f"  2. 2 steps")
    print(f"Expected: {expected1}")
    print(f"Got: {result1}")
    print(f"✅ PASS" if result1 == expected1 else f"❌ FAIL")
    
    # Test Case 2: Example 2 from problem description
    print("\n🧪 Test Case 2: n = 3")
    n2 = 3
    expected2 = 3
    result2 = climb_stairs(n2)
    
    print(f"Input: n = {n2}")
    print(f"Ways to climb:")
    print(f"  1. 1 step + 1 step + 1 step")
    print(f"  2. 1 step + 2 steps")
    print(f"  3. 2 steps + 1 step")
    print(f"Expected: {expected2}")
    print(f"Got: {result2}")
    print(f"✅ PASS" if result2 == expected2 else f"❌ FAIL")
    
    # Test Case 3: Base case n = 1
    print("\n🧪 Test Case 3: n = 1 (Base Case)")
    n3 = 1
    expected3 = 1
    result3 = climb_stairs(n3)
    
    print(f"Input: n = {n3}")
    print(f"Ways to climb:")
    print(f"  1. 1 step")
    print(f"Expected: {expected3}")
    print(f"Got: {result3}")
    print(f"✅ PASS" if result3 == expected3 else f"❌ FAIL")
    
    # Test Case 4: n = 4
    print("\n🧪 Test Case 4: n = 4")
    n4 = 4
    expected4 = 5
    result4 = climb_stairs(n4)
    
    print(f"Input: n = {n4}")
    print(f"Ways to climb:")
    print(f"  1. 1+1+1+1")
    print(f"  2. 1+1+2")
    print(f"  3. 1+2+1")
    print(f"  4. 2+1+1")
    print(f"  5. 2+2")
    print(f"Expected: {expected4}")
    print(f"Got: {result4}")
    print(f"✅ PASS" if result4 == expected4 else f"❌ FAIL")
    
    # Test Case 5: n = 5
    print("\n🧪 Test Case 5: n = 5")
    n5 = 5
    expected5 = 8
    result5 = climb_stairs(n5)
    
    print(f"Input: n = {n5}")
    print(f"Expected: {expected5} (follows Fibonacci: 1,1,2,3,5,8...)")
    print(f"Got: {result5}")
    print(f"✅ PASS" if result5 == expected5 else f"❌ FAIL")
    
    # Test Case 6: n = 10
    print("\n🧪 Test Case 6: n = 10")
    n6 = 10
    expected6 = 89
    result6 = climb_stairs(n6)
    
    print(f"Input: n = {n6}")
    print(f"Expected: {expected6}")
    print(f"Got: {result6}")
    print(f"✅ PASS" if result6 == expected6 else f"❌ FAIL")
    
    # Test Case 7: Larger input n = 20
    print("\n🧪 Test Case 7: n = 20 (Performance Test)")
    n7 = 20
    expected7 = 10946
    result7 = climb_stairs(n7)
    
    print(f"Input: n = {n7}")
    print(f"Expected: {expected7}")
    print(f"Got: {result7}")
    print(f"✅ PASS" if result7 == expected7 else f"❌ FAIL")
    
    # Test Case 8: Even larger input n = 30
    print("\n🧪 Test Case 8: n = 30 (Efficiency Test)")
    n8 = 30
    expected8 = 1346269
    result8 = climb_stairs(n8)
    
    print(f"Input: n = {n8}")
    print(f"Expected: {expected8}")
    print(f"Got: {result8}")
    print(f"✅ PASS" if result8 == expected8 else f"❌ FAIL")
    
    # Test Case 9: Pattern verification
    print("\n🧪 Test Case 9: Fibonacci Pattern Verification")
    print("Verifying that climb_stairs follows Fibonacci sequence:")
    fibonacci_expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    
    all_fibonacci_correct = True
    for i in range(1, 11):
        result = climb_stairs(i)
        expected = fibonacci_expected[i-1]
        is_correct = result == expected
        print(f"  n={i}: got {result}, expected {expected} {'✓' if is_correct else '✗'}")
        if not is_correct:
            all_fibonacci_correct = False
    
    print(f"Fibonacci pattern: {'✅ PASS' if all_fibonacci_correct else '❌ FAIL'}")
    
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
        all_fibonacci_correct
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Great job!")
    else:
        print("❌ Some tests failed. Review your solution.")
    
    print("\n💡 SOLUTION APPROACHES:")
    print("1. RECURSIVE: Simple but inefficient - O(2^n) time complexity")
    print("2. MEMOIZATION: Top-down DP with caching - O(n) time, O(n) space")
    print("3. TABULATION: Bottom-up DP with array - O(n) time, O(n) space")
    print("4. OPTIMIZED: Space-optimized DP - O(n) time, O(1) space")
    
    print("\n📚 LEARNING OBJECTIVES:")
    print("- Recognize Fibonacci pattern in DP problems")
    print("- Understand state transitions: dp[i] = dp[i-1] + dp[i-2]")
    print("- Practice space optimization techniques")
    print("- Master the fundamental 1D DP pattern")
    
    print("\n🔍 PROBLEM ANALYSIS:")
    print("- To reach step n, you can come from step (n-1) or step (n-2)")
    print("- Number of ways = ways to reach (n-1) + ways to reach (n-2)")
    print("- This creates the recurrence relation: f(n) = f(n-1) + f(n-2)")
    print("- Base cases: f(1) = 1, f(2) = 2")


# Reference solutions (uncomment to check your work)
def climb_stairs_recursive(n: int) -> int:
    """
    Naive recursive solution - O(2^n) time, O(n) space
    Too slow for large inputs but demonstrates the recurrence relation
    """
    if n <= 2:
        return n
    return climb_stairs_recursive(n - 1) + climb_stairs_recursive(n - 2)


def climb_stairs_memoization(n: int) -> int:
    """
    Top-down DP with memoization - O(n) time, O(n) space
    """
    memo = {}
    
    def helper(n):
        if n in memo:
            return memo[n]
        
        if n <= 2:
            return n
        
        memo[n] = helper(n - 1) + helper(n - 2)
        return memo[n]
    
    return helper(n)


def climb_stairs_dp(n: int) -> int:
    """
    Bottom-up DP with array - O(n) time, O(n) space
    """
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def climb_stairs_optimized(n: int) -> int:
    """
    Space-optimized DP - O(n) time, O(1) space
    Since we only need the previous two values, we can optimize space
    """
    if n <= 2:
        return n
    
    prev2 = 1  # f(1)
    prev1 = 2  # f(2)
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


if __name__ == "__main__":
    main() 