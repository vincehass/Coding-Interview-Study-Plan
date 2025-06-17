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


# =============================================================================
# PROBLEM 1: BINARY SEARCH (EASY) - 30 MIN
# =============================================================================

def binary_search_basic(nums, target):
    """
    PROBLEM: Binary Search
    
    Given an array of integers nums which is sorted in ascending order, and an integer target, 
    write a function to search target in nums. If target exists, then return its index. 
    Otherwise, return -1.
    
    You must write an algorithm with O(log n) runtime complexity.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^4
    - -10^4 < nums[i], target < 10^4
    - All the integers in nums are unique
    - nums is sorted in ascending order
    
    EXAMPLES:
    Example 1:
        Input: nums = [-1,0,3,5,9,12], target = 9
        Output: 4
        Explanation: 9 exists in nums and its index is 4
    
    Example 2:
        Input: nums = [-1,0,3,5,9,12], target = 2
        Output: -1
        Explanation: 2 does not exist in nums so return -1
    
    APPROACH: Classic Binary Search
    
    Divide search space in half at each step
    
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


# =============================================================================
# PROBLEM 2: FIRST BAD VERSION (EASY) - 30 MIN
# =============================================================================

def first_bad_version(n):
    """
    PROBLEM: First Bad Version
    
    You are a product manager and currently leading a team to develop a new product. 
    Unfortunately, the latest version of your product fails the quality check. Since each 
    version is developed based on the previous version, all the versions after a bad version are also bad.
    
    Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, 
    which causes all the following ones to be bad.
    
    You are given an API bool isBadVersion(version) which returns whether version is bad. 
    Implement a function to find the first bad version. You should minimize the number of calls to the API.
    
    CONSTRAINTS:
    - 1 <= bad <= n <= 2^31 - 1
    
    EXAMPLES:
    Example 1:
        Input: n = 5, bad = 4
        Output: 4
        Explanation:
        call isBadVersion(3) -> false
        call isBadVersion(5) -> true
        call isBadVersion(4) -> true
        Then 4 is the first bad version.
    
    Example 2:
        Input: n = 1, bad = 1
        Output: 1
    
    APPROACH: Binary Search for First Occurrence
    
    Find the leftmost position where condition becomes true
    
    TIME: O(log n), SPACE: O(1)
    """
    def isBadVersion(version):
        # This is a mock function - in real problem it's provided
        return version >= 4  # Assuming bad version starts at 4
    
    left, right = 1, n
    
    while left < right:
        mid = (left + right) // 2
        
        if isBadVersion(mid):
            right = mid  # Could be the first bad, search left
        else:
            left = mid + 1  # Not bad, search right
    
    return left


# =============================================================================
# PROBLEM 3: SEARCH INSERT POSITION (EASY) - 30 MIN
# =============================================================================

def search_insert(nums, target):
    """
    PROBLEM: Search Insert Position
    
    Given a sorted array of distinct integers and a target value, return the index if the 
    target is found. If not, return the index where it would be if it were inserted in order.
    
    You must write an algorithm with O(log n) runtime complexity.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^4
    - -10^4 <= nums[i] <= 10^4
    - nums contains distinct values sorted in ascending order
    - -10^4 <= target <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,3,5,6], target = 5
        Output: 2
    
    Example 2:
        Input: nums = [1,3,5,6], target = 2
        Output: 1
    
    Example 3:
        Input: nums = [1,3,5,6], target = 7
        Output: 4
    
    APPROACH: Binary Search for Insertion Point
    
    Find the leftmost position where target can be inserted
    
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums)
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left


# =============================================================================
# PROBLEM 4: FIND PEAK ELEMENT (MEDIUM) - 45 MIN
# =============================================================================

def find_peak_element(nums):
    """
    PROBLEM: Find Peak Element
    
    A peak element is an element that is strictly greater than its neighbors.
    
    Given a 0-indexed integer array nums, find a peak element, and return its index. 
    If the array contains multiple peaks, return the index to any of the peaks.
    
    You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always 
    considered to be strictly greater than a neighbor that is outside the array.
    
    You must write an algorithm that runs in O(log n) time.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 1000
    - -2^31 <= nums[i] <= 2^31 - 1
    - nums[i] != nums[i + 1] for all valid i
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,2,3,1]
        Output: 2
        Explanation: 3 is a peak element and your function should return the index number 2
    
    Example 2:
        Input: nums = [1,2,1,3,5,6,4]
        Output: 5
        Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6
    
    APPROACH: Binary Search on Slope
    
    Move towards the increasing slope to find a peak
    
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] < nums[mid + 1]:
            # Slope is increasing, peak is to the right
            left = mid + 1
        else:
            # Slope is decreasing, peak is to the left (or at mid)
            right = mid
    
    return left


# =============================================================================
# PROBLEM 5: SEARCH IN ROTATED SORTED ARRAY (MEDIUM) - 45 MIN
# =============================================================================

def search_rotated_array(nums, target):
    """
    PROBLEM: Search in Rotated Sorted Array
    
    There is an integer array nums sorted in ascending order (with distinct values).
    
    Prior to being passed to your function, nums is possibly rotated at an unknown pivot 
    index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., 
    nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] 
    might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
    
    Given the array nums after the possible rotation and an integer target, return the index 
    of target if it is in nums, or -1 if it is not in nums.
    
    You must write an algorithm with O(log n) runtime complexity.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 5000
    - -10^4 <= nums[i] <= 10^4
    - All values of nums are unique
    - nums is an ascending array that is possibly rotated
    - -10^4 <= target <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: nums = [4,5,6,7,0,1,2], target = 0
        Output: 4
    
    Example 2:
        Input: nums = [4,5,6,7,0,1,2], target = 3
        Output: -1
    
    Example 3:
        Input: nums = [1], target = 0
        Output: -1
    
    APPROACH: Modified Binary Search
    
    Determine which half is sorted, then decide which half to search
    
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
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


# =============================================================================
# PROBLEM 6: FIND MINIMUM IN ROTATED SORTED ARRAY (MEDIUM) - 45 MIN
# =============================================================================

def find_min_rotated(nums):
    """
    PROBLEM: Find Minimum in Rotated Sorted Array
    
    Suppose an array of length n sorted in ascending order is rotated between 1 and n times. 
    For example, the array nums = [0,1,2,4,5,6,7] might become:
    - [4,5,6,7,0,1,2] if it was rotated 4 times
    - [0,1,2,4,5,6,7] if it was rotated 7 times
    
    Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array 
    [a[n-1], a[0], a[1], a[2], ..., a[n-2]].
    
    Given the sorted rotated array nums of unique elements, return the minimum element of this array.
    
    You must write an algorithm that runs in O(log n) time.
    
    CONSTRAINTS:
    - n == nums.length
    - 1 <= n <= 5000
    - -5000 <= nums[i] <= 5000
    - All the integers of nums are unique
    - nums is sorted and rotated between 1 and n times
    
    EXAMPLES:
    Example 1:
        Input: nums = [3,4,5,1,2]
        Output: 1
        Explanation: The original array was [1,2,3,4,5] rotated 3 times
    
    Example 2:
        Input: nums = [4,5,6,7,0,1,2]
        Output: 0
        Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times
    
    Example 3:
        Input: nums = [11,13,15,17]
        Output: 11
        Explanation: The original array was [11,13,15,17] and it was rotated 4 times
    
    APPROACH: Binary Search for Rotation Point
    
    The minimum element is at the rotation point
    
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    
    # Array is not rotated
    if nums[left] <= nums[right]:
        return nums[left]
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid
    
    return nums[left]


# =============================================================================
# PROBLEM 7: SEARCH A 2D MATRIX (MEDIUM) - 45 MIN
# =============================================================================

def search_matrix(matrix, target):
    """
    PROBLEM: Search a 2D Matrix
    
    Write an efficient algorithm that searches for a value target in an m x n integer matrix. 
    This matrix has the following properties:
    - Integers in each row are sorted from left to right
    - The first integer of each row is greater than the last integer of the previous row
    
    CONSTRAINTS:
    - m == matrix.length
    - n == matrix[i].length
    - 1 <= m, n <= 100
    - -10^4 <= matrix[i][j], target <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
        Output: true
    
    Example 2:
        Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
        Output: false
    
    APPROACH: Treat as Sorted 1D Array
    
    Use binary search on conceptual 1D array
    
    TIME: O(log(m*n)), SPACE: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_value = matrix[mid // n][mid % n]
        
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


# =============================================================================
# PROBLEM 8: FIND FIRST AND LAST POSITION (MEDIUM) - 45 MIN
# =============================================================================

def search_range(nums, target):
    """
    PROBLEM: Find First and Last Position of Element in Sorted Array
    
    Given an array of integers nums sorted in non-decreasing order, find the starting 
    and ending position of a given target value.
    
    If target is not found in the array, return [-1, -1].
    
    You must write an algorithm with O(log n) runtime complexity.
    
    CONSTRAINTS:
    - 0 <= nums.length <= 10^5
    - -10^9 <= nums[i] <= 10^9
    - nums is a non-decreasing array
    - -10^9 <= target <= 10^9
    
    EXAMPLES:
    Example 1:
        Input: nums = [5,7,7,8,8,10], target = 8
        Output: [3,4]
    
    Example 2:
        Input: nums = [5,7,7,8,8,10], target = 6
        Output: [-1,-1]
    
    Example 3:
        Input: nums = [], target = 0
        Output: [-1,-1]
    
    APPROACH: Two Binary Searches
    
    Find leftmost and rightmost positions separately
    
    TIME: O(log n), SPACE: O(1)
    """
    def find_leftmost(nums, target):
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left
    
    def find_rightmost(nums, target):
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left - 1
    
    if not nums:
        return [-1, -1]
    
    left_pos = find_leftmost(nums, target)
    
    # Target not found
    if left_pos == len(nums) or nums[left_pos] != target:
        return [-1, -1]
    
    right_pos = find_rightmost(nums, target)
    
    return [left_pos, right_pos]


# =============================================================================
# PROBLEM 9: SQRT(X) (EASY) - 30 MIN
# =============================================================================

def my_sqrt(x):
    """
    PROBLEM: Sqrt(x)
    
    Given a non-negative integer x, return the square root of x rounded down to the nearest integer. 
    The returned integer should be non-negative as well.
    
    You must not use any built-in exponent function or operator.
    
    CONSTRAINTS:
    - 0 <= x <= 2^31 - 1
    
    EXAMPLES:
    Example 1:
        Input: x = 4
        Output: 2
        Explanation: The square root of 4 is 2, so we return 2
    
    Example 2:
        Input: x = 8
        Output: 2
        Explanation: The square root of 8 is 2.828..., and since we round it down to the nearest integer, 2 is returned
    
    APPROACH: Binary Search on Answer
    
    Search for the largest integer whose square is ≤ x
    
    TIME: O(log x), SPACE: O(1)
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Return the largest valid value


# =============================================================================
# PROBLEM 10: VALID PERFECT SQUARE (EASY) - 30 MIN
# =============================================================================

def is_perfect_square(num):
    """
    PROBLEM: Valid Perfect Square
    
    Given a positive integer num, return true if num is a perfect square or false otherwise.
    
    A perfect square is an integer that is the square of an integer. In other words, 
    it is the product of some integer with itself.
    
    You must not use any built-in library function, such as sqrt.
    
    CONSTRAINTS:
    - 1 <= num <= 2^31 - 1
    
    EXAMPLES:
    Example 1:
        Input: num = 16
        Output: true
        Explanation: We return true because 4 * 4 = 16 and 4 is an integer
    
    Example 2:
        Input: num = 14
        Output: false
        Explanation: We return false because 3.742 * 3.742 = 14 and 3.742 is not an integer
    
    APPROACH: Binary Search
    
    Search for an integer whose square equals num
    
    TIME: O(log num), SPACE: O(1)
    """
    if num < 2:
        return True
    
    left, right = 2, num // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == num:
            return True
        elif square < num:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


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