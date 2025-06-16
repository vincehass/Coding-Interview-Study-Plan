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

# Problem 1: Two Sum - Classic example transitioning to hash table approach
def two_sum_brute_force(nums, target):
    """
    Brute force approach - helps understand the problem before optimization
    Time: O(n²), Space: O(1)
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_optimized(nums, target):
    """
    Hash table approach - demonstrates transition from two pointers to hash maps
    Time: O(n), Space: O(n)
    """
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []


# Problem 2: Container With Most Water - Pure two pointers problem
def max_area(height):
    """
    Two pointers approach - move pointer with smaller height
    
    Intuition: To maximize area, we want maximum width and height.
    Starting from maximum width, we move the pointer with smaller height
    because moving the larger height pointer can only decrease the area.
    
    Time: O(n), Space: O(1)
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


# Problem 3: 3Sum - Extension of two sum using two pointers
def three_sum(nums):
    """
    Sort array then use two pointers for each fixed element
    
    Strategy:
    1. Sort the array
    2. Fix one element (i)
    3. Use two pointers to find pair that sums to -nums[i]
    4. Skip duplicates to avoid duplicate triplets
    
    Time: O(n²), Space: O(1) excluding output
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


# Problem 4: Remove Duplicates from Sorted Array - Same direction two pointers
def remove_duplicates(nums):
    """
    Two pointers: slow pointer tracks position for next unique element
    
    Pattern: Slow and fast pointers moving in same direction
    - Fast pointer explores the array
    - Slow pointer tracks the position for next valid element
    
    Time: O(n), Space: O(1)
    """
    if not nums:
        return 0
    
    slow = 0  # Position for next unique element
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1


# Problem 5: Trapping Rain Water - Advanced two pointers with auxiliary variables
def trap_rain_water(height):
    """
    Two pointers with left_max and right_max tracking
    
    Key insight: Water level at position i is determined by
    min(max_left, max_right) - height[i]
    
    We can use two pointers and maintain max heights seen so far
    
    Time: O(n), Space: O(1)
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


# PRACTICE PROBLEMS AND TEST CASES
def test_all_problems():
    """
    Test all implemented solutions with various test cases
    """
    print("=== TESTING DAY 1 PROBLEMS ===\n")
    
    # Test Two Sum
    print("1. Two Sum Tests:")
    test_cases_two_sum = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1])
    ]
    
    for nums, target, expected in test_cases_two_sum:
        result = two_sum_optimized(nums, target)
        print(f"   Input: {nums}, Target: {target}")
        print(f"   Output: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Container With Most Water
    print("2. Container With Most Water Tests:")
    test_cases_water = [
        ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
        ([1, 1], 1),
        ([4, 3, 2, 1, 4], 16),
        ([1, 2, 1], 2)
    ]
    
    for height, expected in test_cases_water:
        result = max_area(height)
        print(f"   Input: {height}")
        print(f"   Output: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test 3Sum
    print("3. 3Sum Tests:")
    test_cases_3sum = [
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([0, 1, 1], []),
        ([0, 0, 0], [[0, 0, 0]])
    ]
    
    for nums, expected in test_cases_3sum:
        result = three_sum(nums)
        print(f"   Input: {nums}")
        print(f"   Output: {result}")
        print(f"   Expected: {expected}")
        print(f"   ✓ Correct" if sorted(result) == sorted(expected) else f"   ✗ Wrong")
        print()
    
    # Test Remove Duplicates
    print("4. Remove Duplicates Tests:")
    test_cases_remove_dup = [
        ([1, 1, 2], 2),
        ([0, 0, 1, 1, 1, 2, 2, 3, 3, 4], 5)
    ]
    
    for nums, expected in test_cases_remove_dup:
        original = nums.copy()
        result = remove_duplicates(nums)
        print(f"   Input: {original}")
        print(f"   Length: {result}, Expected: {expected}")
        print(f"   Modified array: {nums[:result]}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Trapping Rain Water
    print("5. Trapping Rain Water Tests:")
    test_cases_trap = [
        ([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6),
        ([4, 2, 0, 3, 2, 5], 9),
        ([3, 0, 2, 0, 4], 7)
    ]
    
    for height, expected in test_cases_trap:
        result = trap_rain_water(height)
        print(f"   Input: {height}")
        print(f"   Output: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()


# EDUCATIONAL HELPER FUNCTIONS
def visualize_two_pointers_process(arr, target):
    """
    Visualize how two pointers work for finding pair sum
    Educational purpose - shows step by step process
    """
    print(f"\nVisualizing Two Pointers for target sum {target}:")
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
            print(f"  ✓ Found target! Indices: [{left}, {right}]")
            return [left, right]
        elif current_sum < target:
            print(f"  Sum too small, move left pointer →")
            left += 1
        else:
            print(f"  Sum too large, move right pointer ←")
            right -= 1
        
        step += 1
    
    print(f"  Target sum not found")
    return []


# COMPLEXITY ANALYSIS EXAMPLES
def analyze_complexity():
    """
    Demonstrates complexity improvement with two pointers
    """
    print("\n=== COMPLEXITY ANALYSIS ===")
    print("\nProblem: Find pair with target sum")
    print("\nBrute Force Approach:")
    print("  - Nested loops: O(n²) time")
    print("  - Space: O(1)")
    print("  - Code: for i in range(n): for j in range(i+1, n):")
    
    print("\nTwo Pointers Approach (sorted array):")
    print("  - Single pass: O(n) time")
    print("  - Space: O(1)")
    print("  - Code: while left < right: if sum == target...")
    
    print("\nHash Table Approach:")
    print("  - Single pass: O(n) time")
    print("  - Space: O(n)")
    print("  - Code: if complement in hash_map...")
    
    print("\nTradeoff: Two pointers needs sorted array, hash table doesn't")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*60)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*60)
    
    # Demonstrate two pointers process
    visualize_two_pointers_process([1, 2, 3, 4, 6], 6)
    
    # Show complexity analysis
    analyze_complexity()
    
    print("\n" + "="*60)
    print("DAY 1 COMPLETE - KEY TAKEAWAYS:")
    print("="*60)
    print("1. Two pointers reduce O(n²) to O(n) for many problems")
    print("2. Opposite ends pattern: palindromes, pair sums, containers")
    print("3. Same direction pattern: remove duplicates, sliding window")
    print("4. Always consider if array needs to be sorted first")
    print("5. Handle edge cases: empty arrays, single elements")
    print("6. Practice visualizing pointer movements")
    print("\nNext: Day 2 - Strings & Pattern Matching") 