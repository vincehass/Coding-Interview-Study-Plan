"""
=============================================================================
                    WEEK 2 - DAY 9: BINARY SEARCH ALGORITHM
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. BINARY SEARCH FUNDAMENTALS
   - Divide and conquer algorithm on sorted data
   - Key insight: Eliminate half of search space in each step
   - Time complexity: O(log n), Space: O(1) iterative, O(log n) recursive
   - Requires sorted or monotonic property

2. BINARY SEARCH TEMPLATE
   - Standard template prevents off-by-one errors
   - Three key components:
     * Initialization: left = 0, right = len(array) - 1
     * Loop condition: while left <= right
     * Update: left = mid + 1 or right = mid - 1

3. BINARY SEARCH VARIATIONS
   - Find exact target
   - Find first/last occurrence
   - Find insertion position
   - Search in rotated arrays
   - Find peak elements
   - Search in 2D matrices

4. SEARCH SPACE THINKING
   - Binary search not just for arrays
   - Any monotonic function can be binary searched
   - Examples: sqrt(x), capacity problems, optimization problems
   - Key: Define search space and monotonic property

5. COMMON PITFALLS
   - Off-by-one errors in boundary conditions
   - Infinite loops from incorrect updates
   - Integer overflow: use mid = left + (right - left) // 2
   - Incorrect termination conditions

6. TRANSITION FROM BST
   - BST search navigates tree structure
   - Binary search navigates array indices
   - Both use comparison-based elimination
   - Both achieve O(log n) with divide-and-conquer

=============================================================================
"""

from typing import List


# Problem 1: Classic Binary Search - Foundation template
def binary_search_iterative(nums, target):
    """
    Standard binary search for exact target
    
    Template that prevents common errors
    
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Prevent overflow
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found


def binary_search_recursive(nums, target):
    """
    Recursive binary search implementation
    
    More intuitive but uses O(log n) space
    
    Time: O(log n), Space: O(log n)
    """
    def search(left, right):
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            return search(mid + 1, right)
        else:
            return search(left, mid - 1)
    
    return search(0, len(nums) - 1)


# Problem 2: Find First and Last Position - Boundary search
def search_range(nums, target):
    """
    Find first and last position of target in sorted array
    
    Use modified binary search to find boundaries
    
    Time: O(log n), Space: O(1)
    """
    def find_first(nums, target):
        left, right = 0, len(nums) - 1
        first_pos = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                first_pos = mid
                right = mid - 1  # Continue searching left
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return first_pos
    
    def find_last(nums, target):
        left, right = 0, len(nums) - 1
        last_pos = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                last_pos = mid
                left = mid + 1  # Continue searching right
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return last_pos
    
    first = find_first(nums, target)
    if first == -1:
        return [-1, -1]
    
    last = find_last(nums, target)
    return [first, last]


# Problem 3: Search in Rotated Sorted Array - Modified binary search
def search_rotated_array(nums, target):
    """
    Search in rotated sorted array
    
    Key insight: One half is always sorted
    Determine which half is sorted, then decide which half to search
    
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


def find_minimum_rotated(nums):
    """
    Find minimum element in rotated sorted array
    
    Minimum is the pivot point where rotation occurred
    
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    # Array not rotated
    if nums[left] <= nums[right]:
        return nums[left]
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Check if mid is the minimum
        if mid > 0 and nums[mid] < nums[mid - 1]:
            return nums[mid]
        
        # Check if mid + 1 is the minimum
        if mid < len(nums) - 1 and nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        
        # Decide which half to search
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid - 1
    
    return nums[0]


# Problem 4: Find Peak Element - Mountain array pattern
def find_peak_element(nums):
    """
    Find peak element where nums[i] > nums[i-1] and nums[i] > nums[i+1]
    
    Key insight: Always move towards higher neighbor
    
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] < nums[mid + 1]:
            # Peak is to the right
            left = mid + 1
        else:
            # Peak is to the left or at mid
            right = mid
    
    return left


def find_peak_in_mountain_array(arr):
    """
    Find peak in mountain array (bitonic array)
    
    Mountain array: increases then decreases
    
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < arr[mid + 1]:
            # Still in increasing part
            left = mid + 1
        else:
            # In decreasing part or at peak
            right = mid
    
    return left


# Problem 5: Search 2D Matrix - Extend binary search to 2D
def search_matrix(matrix, target):
    """
    Search in row-wise and column-wise sorted matrix
    
    Treat 2D matrix as 1D sorted array
    
    Time: O(log(m*n)), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Convert 1D index to 2D coordinates
        row = mid // n
        col = mid % n
        
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


def search_matrix_ii(matrix, target):
    """
    Search in matrix where each row and column is sorted
    
    Start from top-right (or bottom-left) corner
    
    Time: O(m + n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    row, col = 0, len(matrix[0]) - 1
    
    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1  # Move left
        else:
            row += 1  # Move down
    
    return False


# Problem 6: Square Root - Search space binary search
def my_sqrt(x):
    """
    Find integer square root using binary search
    
    Search space: [0, x], find largest k where k*k <= x
    
    Time: O(log x), Space: O(1)
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Largest integer whose square <= x


def my_sqrt_precise(x, precision=1e-6):
    """
    Find square root with given precision
    
    Uses floating-point binary search
    
    Time: O(log(x/precision)), Space: O(1)
    """
    if x < 0:
        raise ValueError("Cannot compute square root of negative number")
    
    if x < 1:
        left, right = x, 1
    else:
        left, right = 1, x
    
    while right - left > precision:
        mid = (left + right) / 2
        square = mid * mid
        
        if square < x:
            left = mid
        else:
            right = mid
    
    return (left + right) / 2


# ADVANCED PROBLEMS FOR EXTRA PRACTICE

def find_kth_smallest_in_matrix(matrix, k):
    """
    Find kth smallest element in row-wise and column-wise sorted matrix
    
    Binary search on value range, not indices
    
    Time: O(n * log(max - min)), Space: O(1)
    """
    def count_less_equal(target):
        """Count elements <= target"""
        count = 0
        row, col = len(matrix) - 1, 0
        
        while row >= 0 and col < len(matrix[0]):
            if matrix[row][col] <= target:
                count += row + 1
                col += 1
            else:
                row -= 1
        
        return count
    
    left, right = matrix[0][0], matrix[-1][-1]
    
    while left < right:
        mid = left + (right - left) // 2
        
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    
    return left


def capacity_to_ship_packages(weights, D):
    """
    Find minimum capacity to ship all packages within D days
    
    Binary search on capacity (search space optimization)
    
    Time: O(n * log(sum(weights))), Space: O(1)
    """
    def can_ship(capacity):
        """Check if can ship with given capacity"""
        days = 1
        current_weight = 0
        
        for weight in weights:
            if current_weight + weight > capacity:
                days += 1
                current_weight = weight
            else:
                current_weight += weight
        
        return days <= D
    
    left = max(weights)  # Must handle heaviest package
    right = sum(weights)  # Ship all in one day
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    
    return left


def split_array_largest_sum(nums, m):
    """
    Split array into m subarrays to minimize largest sum
    
    Binary search on answer
    
    Time: O(n * log(sum(nums))), Space: O(1)
    """
    def can_split(max_sum):
        """Check if can split with given max sum"""
        splits = 1
        current_sum = 0
        
        for num in nums:
            if current_sum + num > max_sum:
                splits += 1
                current_sum = num
            else:
                current_sum += num
        
        return splits <= m
    
    left = max(nums)  # At least one element per subarray
    right = sum(nums)  # All elements in one subarray
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    
    return left


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all binary search problems with comprehensive test cases
    """
    print("=== TESTING DAY 9 PROBLEMS ===\n")
    
    # Test Classic Binary Search
    print("1. Classic Binary Search:")
    test_array = [1, 3, 5, 7, 9, 11, 13, 15]
    
    result1 = binary_search_iterative(test_array, 7)
    result2 = binary_search_recursive(test_array, 7)
    result3 = binary_search_iterative(test_array, 6)
    
    print(f"   Array: {test_array}")
    print(f"   Search 7 (iterative): {result1} (expected: 3)")
    print(f"   Search 7 (recursive): {result2} (expected: 3)")
    print(f"   Search 6 (not found): {result3} (expected: -1)")
    print()
    
    # Test Search Range
    print("2. Find First and Last Position:")
    range_array = [5, 7, 7, 8, 8, 8, 10]
    
    range1 = search_range(range_array, 8)
    range2 = search_range(range_array, 6)
    
    print(f"   Array: {range_array}")
    print(f"   Range of 8: {range1} (expected: [3, 5])")
    print(f"   Range of 6: {range2} (expected: [-1, -1])")
    print()
    
    # Test Rotated Array Search
    print("3. Search in Rotated Array:")
    rotated = [4, 5, 6, 7, 0, 1, 2]
    
    rot1 = search_rotated_array(rotated, 0)
    rot2 = search_rotated_array(rotated, 3)
    min_rot = find_minimum_rotated(rotated)
    
    print(f"   Rotated array: {rotated}")
    print(f"   Search 0: {rot1} (expected: 4)")
    print(f"   Search 3: {rot2} (expected: -1)")
    print(f"   Minimum element: {min_rot} (expected: 0)")
    print()
    
    # Test Peak Finding
    print("4. Find Peak Element:")
    peak_array = [1, 2, 3, 1]
    mountain = [0, 1, 0]
    
    peak1 = find_peak_element(peak_array)
    peak2 = find_peak_in_mountain_array(mountain)
    
    print(f"   Peak array: {peak_array}")
    print(f"   Peak index: {peak1} (expected: 2)")
    print(f"   Mountain array: {mountain}")
    print(f"   Mountain peak: {peak2} (expected: 1)")
    print()
    
    # Test 2D Matrix Search
    print("5. Search in 2D Matrix:")
    matrix1 = [[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16]]
    matrix2 = [[1, 3, 5], [2, 4, 6], [7, 8, 9]]
    
    search1 = search_matrix_ii(matrix1, 5)
    search2 = search_matrix(matrix2, 4)
    
    print(f"   Matrix 1: {matrix1}")
    print(f"   Search 5: {search1} (expected: True)")
    print(f"   Matrix 2: {matrix2}")
    print(f"   Search 4: {search2} (expected: True)")
    print()
    
    # Test Square Root
    print("6. Square Root:")
    sqrt1 = my_sqrt(8)
    sqrt2 = my_sqrt(16)
    sqrt_precise = my_sqrt_precise(8, 0.001)
    
    print(f"   Integer sqrt(8): {sqrt1} (expected: 2)")
    print(f"   Integer sqrt(16): {sqrt2} (expected: 4)")
    print(f"   Precise sqrt(8): {sqrt_precise:.3f} (expected: ~2.828)")
    print()


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_binary_search_thinking():
    """
    Visual demonstration of binary search decision process
    """
    print("\n=== BINARY SEARCH THINKING PROCESS ===")
    
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    
    print(f"Searching for {target} in {arr}")
    
    left, right = 0, len(arr) - 1
    step = 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        print(f"\nStep {step}:")
        print(f"  Search space: indices {left} to {right}")
        print(f"  Values: {arr[left]} to {arr[right]}")
        print(f"  Mid index: {mid}, value: {arr[mid]}")
        
        if arr[mid] == target:
            print(f"  Found target at index {mid}!")
            break
        elif arr[mid] < target:
            print(f"  {arr[mid]} < {target}, search right half")
            left = mid + 1
        else:
            print(f"  {arr[mid]} > {target}, search left half")
            right = mid - 1
        
        step += 1
    
    print(f"\nEliminated half the search space in each step!")
    print(f"Total steps: {step} (vs {len(arr)} for linear search)")


def binary_search_variations_guide():
    """
    Guide for different binary search problem types
    """
    print("\n=== BINARY SEARCH PROBLEM PATTERNS ===")
    
    patterns = {
        "Exact Target": [
            "Classic binary search",
            "Search in rotated array",
            "Search in 2D matrix"
        ],
        "Boundary Search": [
            "First/last occurrence",
            "Insert position",
            "Search range"
        ],
        "Peak Finding": [
            "Find peak element",
            "Mountain array peak",
            "Local maximum"
        ],
        "Value Binary Search": [
            "Square root",
            "Kth smallest in matrix",
            "Capacity problems"
        ],
        "Answer Binary Search": [
            "Split array largest sum",
            "Minimize maximum",
            "Optimization problems"
        ]
    }
    
    for pattern, examples in patterns.items():
        print(f"\n{pattern}:")
        for example in examples:
            print(f"  • {example}")


def common_binary_search_mistakes():
    """
    Highlight common mistakes in binary search implementation
    """
    print("\n=== COMMON BINARY SEARCH MISTAKES ===")
    
    print("1. Integer Overflow:")
    print("   WRONG: mid = (left + right) // 2")
    print("   RIGHT: mid = left + (right - left) // 2")
    
    print("\n2. Infinite Loop:")
    print("   WRONG: right = mid (when left can equal right)")
    print("   RIGHT: right = mid - 1 (for exact search)")
    
    print("\n3. Off-by-One Errors:")
    print("   WRONG: while left < right (missing equal case)")
    print("   RIGHT: while left <= right (for exact search)")
    
    print("\n4. Boundary Conditions:")
    print("   WRONG: Not handling empty array")
    print("   RIGHT: Check if array is empty first")
    
    print("\n5. Search Space Definition:")
    print("   WRONG: Incorrect initial left/right values")
    print("   RIGHT: Define search space based on problem")


def complexity_analysis():
    """
    Analyze complexity of different binary search problems
    """
    print("\n=== BINARY SEARCH COMPLEXITY ANALYSIS ===")
    
    print("Time Complexities:")
    print("  Classic binary search: O(log n)")
    print("  First/last position: O(log n)")
    print("  Rotated array search: O(log n)")
    print("  2D matrix search: O(log(m*n)) or O(m+n)")
    print("  Value-based search: O(log(range) * validation_cost)")
    
    print("\nSpace Complexities:")
    print("  Iterative: O(1)")
    print("  Recursive: O(log n) for call stack")
    
    print("\nKey Insights:")
    print("  • Each comparison eliminates half the possibilities")
    print("  • Works on any monotonic function, not just arrays")
    print("  • Can search on answer space, not just input space")
    print("  • Template approach prevents implementation errors")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Demonstrate thinking process
    demonstrate_binary_search_thinking()
    
    # Show problem patterns
    binary_search_variations_guide()
    
    # Common mistakes
    common_binary_search_mistakes()
    
    # Complexity analysis
    complexity_analysis()
    
    print("\n" + "="*70)
    print("DAY 9 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. Master the standard binary search template")
    print("2. Binary search works on any monotonic property")
    print("3. Can search on values/answers, not just array indices")
    print("4. Always consider boundary cases and off-by-one errors")
    print("5. Different update strategies for different problem types")
    print("6. O(log n) efficiency by eliminating half each step")
    print("7. Extends to 2D problems and optimization problems")
    print("\nTransition: Day 9→10 - From search algorithms to advanced tree problems")
    print("- Binary search principles apply to tree navigation")
    print("- Complex tree problems often combine multiple concepts")
    print("- Advanced tree algorithms build on search foundations")
    print("\nNext: Day 10 - Advanced Tree Problems") 