"""
=============================================================================
                        WEEK 1 SINGLE PROBLEM PRACTICE
                              TWO SUM PROBLEM
                           Meta Interview Preparation
=============================================================================

Focus on mastering one core problem with comprehensive testing.
This represents the most fundamental array/hash table pattern in interviews.

=============================================================================
"""

def two_sum(nums, target):
    """
    PROBLEM: Two Sum
    
    DESCRIPTION:
    Given an array of integers `nums` and an integer `target`, return indices of 
    the two numbers such that they add up to `target`.
    
    You may assume that each input would have exactly one solution, and you may 
    not use the same element twice. You can return the answer in any order.
    
    CONSTRAINTS:
    - 2 <= nums.length <= 10^4
    - -10^9 <= nums[i] <= 10^9
    - -10^9 <= target <= 10^9
    - Only one valid answer exists
    
    EXAMPLES:
    Example 1:
        Input: nums = [2,7,11,15], target = 9
        Output: [0,1] (because nums[0] + nums[1] == 9)
    
    Example 2:
        Input: nums = [3,2,4], target = 6
        Output: [1,2]
    
    Example 3:
        Input: nums = [3,3], target = 6
        Output: [0,1]
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    Args:
        nums (List[int]): Array of integers
        target (int): Target sum to find
        
    Returns:
        List[int]: Indices of the two numbers that add up to target
    """
    # Write your solution here
    # Hint: Use a hash map to store numbers and their indices
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i

    return []


def main():
    """Test the two_sum function with various test cases"""
    
    print("=" * 60)
    print("           WEEK 1 SINGLE PROBLEM PRACTICE")
    print("                  TWO SUM TESTING")
    print("=" * 60)
    
    # Test Case 1: Basic example
    print("\n🧪 Test Case 1: Basic Example")
    nums1 = [2, 7, 11, 15]
    target1 = 9
    expected1 = [0, 1]
    result1 = two_sum(nums1, target1)
    print(f"Input: nums = {nums1}, target = {target1}")
    print(f"Expected: {expected1}")
    print(f"Got: {result1}")
    print(f"✅ PASS" if result1 == expected1 else f"❌ FAIL")
    
    # Test Case 2: Different order
    print("\n🧪 Test Case 2: Different Order")
    nums2 = [3, 2, 4]
    target2 = 6
    expected2 = [1, 2]
    result2 = two_sum(nums2, target2)
    print(f"Input: nums = {nums2}, target = {target2}")
    print(f"Expected: {expected2}")
    print(f"Got: {result2}")
    print(f"✅ PASS" if result2 == expected2 else f"❌ FAIL")
    
    # Test Case 3: Duplicate numbers
    print("\n🧪 Test Case 3: Duplicate Numbers")
    nums3 = [3, 3]
    target3 = 6
    expected3 = [0, 1]
    result3 = two_sum(nums3, target3)
    print(f"Input: nums = {nums3}, target = {target3}")
    print(f"Expected: {expected3}")
    print(f"Got: {result3}")
    print(f"✅ PASS" if result3 == expected3 else f"❌ FAIL")
    
    # Test Case 4: Negative numbers
    print("\n🧪 Test Case 4: Negative Numbers")
    nums4 = [-1, -2, -3, -4, -5]
    target4 = -8
    expected4 = [2, 4]  # -3 + -5 = -8
    result4 = two_sum(nums4, target4)
    print(f"Input: nums = {nums4}, target = {target4}")
    print(f"Expected: {expected4}")
    print(f"Got: {result4}")
    print(f"✅ PASS" if result4 == expected4 else f"❌ FAIL")
    
    # Test Case 5: Mixed positive/negative
    print("\n🧪 Test Case 5: Mixed Positive/Negative")
    nums5 = [-3, 4, 3, 90]
    target5 = 0
    expected5 = [0, 2]  # -3 + 3 = 0
    result5 = two_sum(nums5, target5)
    print(f"Input: nums = {nums5}, target = {target5}")
    print(f"Expected: {expected5}")
    print(f"Got: {result5}")
    print(f"✅ PASS" if result5 == expected5 else f"❌ FAIL")
    
    # Test Case 6: Large numbers
    print("\n🧪 Test Case 6: Large Numbers")
    nums6 = [1000000000, 999999999, 1]
    target6 = 1000000001
    expected6 = [0, 2]  # 1000000000 + 1 = 1000000001
    result6 = two_sum(nums6, target6)
    print(f"Input: nums = {nums6}, target = {target6}")
    print(f"Expected: {expected6}")
    print(f"Got: {result6}")
    print(f"✅ PASS" if result6 == expected6 else f"❌ FAIL")
    
    # Test Case 7: Zero target
    print("\n🧪 Test Case 7: Zero Target")
    nums7 = [0, 4, 3, 0]
    target7 = 0
    expected7 = [0, 3]  # 0 + 0 = 0
    result7 = two_sum(nums7, target7)
    print(f"Input: nums = {nums7}, target = {target7}")
    print(f"Expected: {expected7}")
    print(f"Got: {result7}")
    print(f"✅ PASS" if result7 == expected7 else f"❌ FAIL")
    
    # Test Case 8: Minimum array size
    print("\n🧪 Test Case 8: Minimum Array Size")
    nums8 = [1, 2]
    target8 = 3
    expected8 = [0, 1]
    result8 = two_sum(nums8, target8)
    print(f"Input: nums = {nums8}, target = {target8}")
    print(f"Expected: {expected8}")
    print(f"Got: {result8}")
    print(f"✅ PASS" if result8 == expected8 else f"❌ FAIL")
    
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
        result8 == expected8
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Great job!")
    else:
        print("❌ Some tests failed. Review your solution.")
    
    print("\n💡 SOLUTION HINTS:")
    print("1. Use a hash map to store numbers as keys and indices as values")
    print("2. For each number, check if (target - number) exists in the hash map")
    print("3. If found, return the current index and the stored index")
    print("4. If not found, add the current number and index to hash map")
    
    print("\n📚 LEARNING OBJECTIVES:")
    print("- Master the hash table pattern for complement search")
    print("- Understand O(n) time vs O(n²) brute force approach")
    print("- Handle edge cases: duplicates, negatives, zeros")
    print("- Practice clean code with proper variable names")


# Reference solution (uncomment to check your work)
def two_sum_reference(nums, target):
    """
    Reference solution using hash map approach
    Time: O(n), Space: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


if __name__ == "__main__":
    main()