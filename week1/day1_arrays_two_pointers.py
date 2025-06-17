"""
=============================================================================
                      WEEK 1 - DAY 1: ARRAYS & TWO POINTERS
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. ARRAY FUNDAMENTALS
   - Arrays are collections of elements stored in contiguous memory locations
   - Access time: O(1) by index
   - Search time: O(n) for unsorted, O(log n) for sorted (with binary search)
   - Insertion/Deletion: O(n) in worst case (due to shifting elements)

2. TWO POINTER TECHNIQUE
   - Efficient pattern for solving array/string problems
   - Reduces time complexity from O(n²) to O(n) in many cases
   - Two main patterns:
     a) Opposite ends: left=0, right=len(arr)-1, move towards center
     b) Same direction: slow and fast pointers

3. WHEN TO USE TWO POINTERS:
   - Finding pairs with specific sum
   - Removing duplicates from sorted arrays
   - Checking palindromes
   - Sliding window problems
   - Merging sorted arrays

4. COMPLEXITY ANALYSIS:
   - Time: Usually O(n) instead of O(n²)
   - Space: O(1) additional space (in-place)

=============================================================================
"""

# =============================================================================
# PROBLEM 1: TWO SUM (EASY) - 30 MIN
# =============================================================================

def two_sum_brute_force(nums, target):
    """
    PROBLEM: Two Sum
    
    Given an array of integers nums and an integer target, return indices of 
    the two numbers such that they add up to target.
    
    You may assume that each input would have exactly one solution, and you 
    may not use the same element twice. You can return the answer in any order.
    
    CONSTRAINTS:
    - 2 <= nums.length <= 10^4
    - -10^9 <= nums[i] <= 10^9
    - -10^9 <= target <= 10^9
    - Only one valid answer exists
    
    EXAMPLES:
    Example 1:
        Input: nums = [2,7,11,15], target = 9
        Output: [0,1]
        Explanation: nums[0] + nums[1] = 2 + 7 = 9
    
    Example 2:
        Input: nums = [3,2,4], target = 6
        Output: [1,2]
    
    Example 3:
        Input: nums = [3,3], target = 6
        Output: [0,1]
    
    APPROACH: Brute Force (for understanding)
    TIME: O(n²), SPACE: O(1)
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_optimized(nums, target):
    """
    APPROACH: Hash Table (Optimized)
    
    Use hash table to store numbers and their indices. For each number,
    check if its complement (target - number) exists in the hash table.
    
    TIME: O(n), SPACE: O(n)
    """
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []


# =============================================================================
# PROBLEM 2: CONTAINER WITH MOST WATER (MEDIUM) - 45 MIN
# =============================================================================

def max_area(height):
    """
    PROBLEM: Container With Most Water
    
    You are given an integer array height of length n. There are n vertical 
    lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
    
    Find two lines that together with the x-axis form a container that contains 
    the most water.
    
    Return the maximum amount of water a container can store.
    
    CONSTRAINTS:
    - n == height.length
    - 2 <= n <= 10^5
    - 0 <= height[i] <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: height = [1,8,6,2,5,4,8,3,7]
        Output: 49
        Explanation: Max area = 7 * 7 = 49 (between indices 1 and 8)
    
    Example 2:
        Input: height = [1,1]
        Output: 1
    
    APPROACH: Two Pointers
    
    Start with widest container (left=0, right=n-1). The area is limited by 
    the shorter line. Move the pointer with the shorter height inward, as 
    moving the taller pointer can only decrease the area.
    
    TIME: O(n), SPACE: O(1)
    """
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height
        max_water = max(max_water, current_area)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water


# =============================================================================
# PROBLEM 3: 3SUM (MEDIUM) - 60 MIN
# =============================================================================

def three_sum(nums):
    """
    PROBLEM: 3Sum
    
    Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] 
    such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    
    Notice that the solution set must not contain duplicate triplets.
    
    CONSTRAINTS:
    - 3 <= nums.length <= 3000
    - -10^5 <= nums[i] <= 10^5
    
    EXAMPLES:
    Example 1:
        Input: nums = [-1,0,1,2,-1,-4]
        Output: [[-1,-1,2],[-1,0,1]]
        Explanation: 
        nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
        nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
        nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
        The distinct triplets are [-1,0,1] and [-1,-1,2].
    
    Example 2:
        Input: nums = [0,1,1]
        Output: []
        Explanation: The only possible triplet does not sum up to 0.
    
    Example 3:
        Input: nums = [0,0,0]
        Output: [[0,0,0]]
        Explanation: The only possible triplet sums up to 0.
    
    APPROACH: Sort + Two Pointers
    
    1. Sort the array
    2. Fix one element (i)
    3. Use two pointers to find pair that sums to -nums[i]
    4. Skip duplicates to avoid duplicate triplets
    
    TIME: O(n²), SPACE: O(1) excluding output
    """
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
            
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for second element
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # Skip duplicates for third element
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                    
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
                
    return result


# =============================================================================
# PROBLEM 4: REMOVE DUPLICATES FROM SORTED ARRAY (EASY) - 30 MIN
# =============================================================================

def remove_duplicates(nums):
    """
    PROBLEM: Remove Duplicates from Sorted Array
    
    Given an integer array nums sorted in non-decreasing order, remove the 
    duplicates in-place such that each unique element appears only once. 
    The relative order of the elements should be kept the same.
    
    Return k after placing the final result in the first k slots of nums.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 3 * 10^4
    - -100 <= nums[i] <= 100
    - nums is sorted in non-decreasing order
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,1,2]
        Output: 2, nums = [1,2,_]
        Explanation: Your function should return k = 2, with the first two 
        elements of nums being 1 and 2 respectively.
    
    Example 2:
        Input: nums = [0,0,1,1,1,2,2,3,3,4]
        Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
        Explanation: Your function should return k = 5, with the first five 
        elements of nums being 0, 1, 2, 3, and 4 respectively.
    
    APPROACH: Two Pointers (Same Direction)
    
    Use slow pointer to track position for next unique element.
    Fast pointer explores the array.
    
    TIME: O(n), SPACE: O(1)
    """
    if not nums:
        return 0
    
    slow = 0  # Position for next unique element
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1


# =============================================================================
# PROBLEM 5: TRAPPING RAIN WATER (HARD) - 45 MIN
# =============================================================================

def trap_rain_water(height):
    """
    PROBLEM: Trapping Rain Water
    
    Given n non-negative integers representing an elevation map where the width 
    of each bar is 1, compute how much water it can trap after raining.
    
    CONSTRAINTS:
    - n == height.length
    - 1 <= n <= 2 * 10^4
    - 0 <= height[i] <= 3 * 10^4
    
    EXAMPLES:
    Example 1:
        Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
        Output: 6
        Explanation: The elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
        In this case, 6 units of rain water are being trapped.
    
    Example 2:
        Input: height = [4,2,0,3,2,5]
        Output: 9
    
    APPROACH: Two Pointers with Max Height Tracking
    
    Water level at position i is determined by min(max_left, max_right) - height[i].
    Use two pointers and maintain max heights seen so far from both sides.
    
    TIME: O(n), SPACE: O(1)
    """
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water


# =============================================================================
# COMPREHENSIVE TEST CASES
# =============================================================================

def test_all_problems():
    """
    Test all implemented solutions with comprehensive test cases
    """
    print("=" * 60)
    print("           WEEK 1 - DAY 1: TESTING RESULTS")
    print("=" * 60)
    
    # Test Two Sum
    print("\n🧪 PROBLEM 1: TWO SUM")
    test_cases_two_sum = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1])
    ]
    
    for i, (nums, target, expected) in enumerate(test_cases_two_sum, 1):
        result = two_sum_optimized(nums, target)
        print(f"   Test Case {i}:")
        print(f"   Input: nums = {nums}, target = {target}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print(f"   ✅ PASS" if result == expected else f"   ❌ FAIL")
        print()
    
    # Test Container With Most Water
    print("🧪 PROBLEM 2: CONTAINER WITH MOST WATER")
    test_cases_water = [
        ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
        ([1, 1], 1),
        ([4, 3, 2, 1, 4], 16),
        ([1, 2, 1], 2)
    ]
    
    for i, (height, expected) in enumerate(test_cases_water, 1):
        result = max_area(height)
        print(f"   Test Case {i}:")
        print(f"   Input: height = {height}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print(f"   ✅ PASS" if result == expected else f"   ❌ FAIL")
        print()
    
    # Test 3Sum
    print("🧪 PROBLEM 3: 3SUM")
    test_cases_3sum = [
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([0, 1, 1], []),
        ([0, 0, 0], [[0, 0, 0]])
    ]
    
    for i, (nums, expected) in enumerate(test_cases_3sum, 1):
        result = three_sum(nums)
        print(f"   Test Case {i}:")
        print(f"   Input: nums = {nums}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print(f"   ✅ PASS" if sorted(result) == sorted(expected) else f"   ❌ FAIL")
        print()
    
    # Test Remove Duplicates
    print("🧪 PROBLEM 4: REMOVE DUPLICATES FROM SORTED ARRAY")
    test_cases_remove_dup = [
        ([1, 1, 2], 2),
        ([0, 0, 1, 1, 1, 2, 2, 3, 3, 4], 5)
    ]
    
    for i, (nums_input, expected) in enumerate(test_cases_remove_dup, 1):
        nums = nums_input.copy()
        result = remove_duplicates(nums)
        print(f"   Test Case {i}:")
        print(f"   Input: nums = {nums_input}")
        print(f"   Expected length: {expected}")
        print(f"   Got length: {result}")
        print(f"   Modified array: {nums[:result]}")
        print(f"   ✅ PASS" if result == expected else f"   ❌ FAIL")
        print()
    
    # Test Trapping Rain Water
    print("🧪 PROBLEM 5: TRAPPING RAIN WATER")
    test_cases_trap = [
        ([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6),
        ([4, 2, 0, 3, 2, 5], 9),
        ([3, 0, 2, 0, 4], 7)
    ]
    
    for i, (height, expected) in enumerate(test_cases_trap, 1):
        result = trap_rain_water(height)
        print(f"   Test Case {i}:")
        print(f"   Input: height = {height}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print(f"   ✅ PASS" if result == expected else f"   ❌ FAIL")
        print()


# =============================================================================
# EDUCATIONAL HELPER FUNCTIONS
# =============================================================================

def visualize_two_pointers_process(arr, target):
    """
    Visualize how two pointers work for finding pair sum
    Educational purpose - shows step by step process
    """
    print(f"\n📊 VISUALIZING TWO POINTERS for target sum {target}:")
    print(f"Array: {arr}")
    
    left, right = 0, len(arr) - 1
    step = 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        print(f"\nStep {step}:")
        print(f"  Left: {left} (value: {arr[left]})")
        print(f"  Right: {right} (value: {arr[right]})")
        print(f"  Sum: {current_sum}")
        
        if current_sum == target:
            print(f"  ✅ Found target! Indices: [{left}, {right}]")
            return [left, right]
        elif current_sum < target:
            print(f"  Sum too small, move left pointer →")
            left += 1
        else:
            print(f"  Sum too large, move right pointer ←")
            right -= 1
        
        step += 1
    
    print(f"  ❌ Target sum not found")
    return []


def analyze_complexity():
    """
    Demonstrates complexity improvement with two pointers
    """
    print("\n" + "=" * 60)
    print("                 COMPLEXITY ANALYSIS")
    print("=" * 60)
    print("\n📈 PROBLEM: Find pair with target sum")
    print("\n🔴 Brute Force Approach:")
    print("  - Nested loops: O(n²) time")
    print("  - Space: O(1)")
    print("  - Code: for i in range(n): for j in range(i+1, n):")
    
    print("\n🟡 Two Pointers Approach (sorted array):")
    print("  - Single pass: O(n) time")
    print("  - Space: O(1)")
    print("  - Code: while left < right: if sum == target...")
    
    print("\n🟢 Hash Table Approach:")
    print("  - Single pass: O(n) time")
    print("  - Space: O(n)")
    print("  - Code: if complement in hash_map...")
    
    print("\n💡 TRADEOFF: Two pointers needs sorted array, hash table doesn't")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*60)
    print("               EDUCATIONAL DEMONSTRATIONS")
    print("="*60)
    
    # Demonstrate two pointers process
    visualize_two_pointers_process([1, 2, 3, 4, 6], 6)
    
    # Show complexity analysis
    analyze_complexity()
    
    print("\n" + "="*60)
    print("                   DAY 1 COMPLETE")
    print("="*60)
    print("🎯 KEY TAKEAWAYS:")
    print("1. Two pointers reduce O(n²) to O(n) for many problems")
    print("2. Opposite ends pattern: palindromes, pair sums, containers")
    print("3. Same direction pattern: remove duplicates, sliding window")
    print("4. Always consider if array needs to be sorted first")
    print("5. Handle edge cases: empty arrays, single elements")
    print("6. Practice visualizing pointer movements")
    print("\n🚀 NEXT: Day 2 - Strings & Pattern Matching") 