"""
=============================================================================
                     WEEK 1 - DAY 3: HASH TABLES & SETS
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. HASH TABLE FUNDAMENTALS
   - Key-value data structure with O(1) average operations
   - Hash function maps keys to indices in underlying array
   - Collision handling: chaining (linked lists) vs open addressing
   - Load factor affects performance: keep < 0.75 for good performance

2. HASH TABLE OPERATIONS
   - Insert: O(1) average, O(n) worst case
   - Search: O(1) average, O(n) worst case
   - Delete: O(1) average, O(n) worst case
   - Space complexity: O(n)

3. WHEN TO USE HASH TABLES
   - Fast lookups by key
   - Frequency counting
   - Checking for duplicates
   - Caching/memoization
   - Two sum type problems

4. PYTHON IMPLEMENTATIONS
   - dict: Hash table implementation
   - set: Hash table for unique elements only
   - collections.defaultdict: dict with default values
   - collections.Counter: dict for counting

5. HASH TABLE PATTERNS
   - Frequency counting: count occurrences
   - Two sum pattern: complement lookup
   - Sliding window with hash map: character frequencies
   - Grouping: group by computed key

6. TRANSITION FROM PREVIOUS TOPICS
   - Arrays + Hash: Two sum optimization
   - Strings + Hash: Anagram detection, character frequency
   - Next: Hash tables support advanced data structures

=============================================================================
"""

from collections import defaultdict, Counter
import heapq


# =============================================================================
# PROBLEM 1: TWO SUM (EASY) - 30 MIN
# =============================================================================

def two_sum_hash(nums, target):
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
    
    APPROACH: Hash Table (Optimized)
    
    Use hash table to store numbers and their indices. For each number,
    check if its complement (target - number) exists in the hash table.
    
    TIME: O(n), SPACE: O(n)
    """
    num_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    
    return []


# =============================================================================
# PROBLEM 2: SUBARRAY SUM EQUALS K (MEDIUM) - 45 MIN
# =============================================================================

def subarray_sum(nums, k):
    """
    PROBLEM: Subarray Sum Equals K
    
    Given an array of integers nums and an integer k, return the total number 
    of subarrays whose sum equals to k.
    
    A subarray is a contiguous non-empty sequence of elements within an array.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 2 * 10^4
    - -1000 <= nums[i] <= 1000
    - -10^7 <= k <= 10^7
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,1,1], k = 2
        Output: 2
        Explanation: Subarrays [1,1] and [1,1] have sum 2
    
    Example 2:
        Input: nums = [1,2,3], k = 3
        Output: 2
        Explanation: Subarrays [1,2] and [3] have sum 3
    
    APPROACH: Prefix Sum with Hash Map
    
    Key insight: If prefix_sum[j] - prefix_sum[i] = k, then
    subarray from i+1 to j has sum k
    
    Algorithm:
    1. Maintain running prefix sum
    2. For each position, check if (prefix_sum - k) exists in hash map
    3. Count occurrences of each prefix sum
    
    TIME: O(n), SPACE: O(n)
    """
    count = 0
    prefix_sum = 0
    sum_count = defaultdict(int)
    sum_count[0] = 1  # Empty prefix has sum 0
    
    for num in nums:
        prefix_sum += num
        
        # Check if there's a prefix sum such that current - that = k
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        # Add current prefix sum to map
        sum_count[prefix_sum] += 1
    
    return count


# =============================================================================
# PROBLEM 3: TOP K FREQUENT ELEMENTS (MEDIUM) - 45 MIN
# =============================================================================

def top_k_frequent(nums, k):
    """
    PROBLEM: Top K Frequent Elements
    
    Given an integer array nums and an integer k, return the k most frequent elements.
    You may return the answer in any order.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    - k is in the range [1, the number of unique elements in the array]
    - It's guaranteed that the answer is unique
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,1,1,2,2,3], k = 2
        Output: [1,2]
    
    Example 2:
        Input: nums = [1], k = 1
        Output: [1]
    
    APPROACH 1: Hash Map + Min Heap
    
    1. Count frequencies with hash map
    2. Use min-heap of size k to find top k elements
    
    TIME: O(n log k), SPACE: O(n)
    """
    # Count frequencies
    freq_map = Counter(nums)
    
    # Use min-heap to maintain top k elements
    heap = []
    
    for num, freq in freq_map.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]


def top_k_frequent_bucket_sort(nums, k):
    """
    APPROACH 2: Bucket Sort (Optimized)
    
    Since frequency is bounded by array length, we can use bucket sort
    
    TIME: O(n), SPACE: O(n)
    """
    freq_map = Counter(nums)
    
    # Create buckets for each possible frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    # Place elements in buckets by frequency
    for num, freq in freq_map.items():
        buckets[freq].append(num)
    
    # Collect top k elements from highest frequency buckets
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result


# =============================================================================
# PROBLEM 4: FIRST MISSING POSITIVE (HARD) - 60 MIN
# =============================================================================

def first_missing_positive(nums):
    """
    PROBLEM: First Missing Positive
    
    Given an unsorted integer array nums, return the smallest missing positive integer.
    
    You must implement an algorithm that runs in O(n) time and uses constant extra space.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 5 * 10^5
    - -2^31 <= nums[i] <= 2^31 - 1
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,2,0]
        Output: 3
        Explanation: The numbers in the range [1,2] are all in the array
    
    Example 2:
        Input: nums = [3,4,-1,1]
        Output: 2
        Explanation: 1 is in the array but 2 is missing
    
    Example 3:
        Input: nums = [7,8,9,11,12]
        Output: 1
        Explanation: The smallest positive integer 1 is missing
    
    APPROACH 1: Hash Set (Simple)
    
    Use hash set for O(1) lookups
    
    TIME: O(n), SPACE: O(n)
    """
    num_set = set(nums)
    
    positive = 1
    while positive in num_set:
        positive += 1
    
    return positive


def first_missing_positive_in_place(nums):
    """
    APPROACH 2: In-place Array as Hash Table (Optimal)
    
    Key insight: Answer is in range [1, n+1] where n = len(nums)
    Use array indices as hash keys by placing each number at index (number-1)
    
    TIME: O(n), SPACE: O(1)
    """
    n = len(nums)
    
    # Place each number at its correct position
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap nums[i] to its correct position
            correct_pos = nums[i] - 1
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
    
    # Find first position where number doesn't match index + 1
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1


# =============================================================================
# PROBLEM 5: INTERSECTION OF TWO ARRAYS (EASY) - 30 MIN
# =============================================================================

def intersection(nums1, nums2):
    """
    PROBLEM: Intersection of Two Arrays
    
    Given two integer arrays nums1 and nums2, return an array of their intersection.
    Each element in the result must be unique and you may return the result in any order.
    
    CONSTRAINTS:
    - 1 <= nums1.length, nums2.length <= 1000
    - 0 <= nums1[i], nums2[i] <= 1000
    
    EXAMPLES:
    Example 1:
        Input: nums1 = [1,2,2,1], nums2 = [2,2]
        Output: [2]
    
    Example 2:
        Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
        Output: [9,4] (or [4,9])
    
    APPROACH 1: Two Sets
    
    Convert both arrays to sets and find intersection
    
    TIME: O(n + m), SPACE: O(n + m)
    """
    return list(set(nums1) & set(nums2))


def intersection_hash_table(nums1, nums2):
    """
    APPROACH 2: Hash Table
    
    Use hash table to track elements from first array,
    then check elements from second array
    
    TIME: O(n + m), SPACE: O(min(n, m))
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    seen = set(nums1)
    result = set()
    
    for num in nums2:
        if num in seen:
            result.add(num)
    
    return list(result)


# =============================================================================
# PROBLEM 6: 4SUM (MEDIUM) - 60 MIN
# =============================================================================

def four_sum(nums, target):
    """
    PROBLEM: 4Sum
    
    Given an array nums of n integers, return an array of all the unique quadruplets
    [nums[a], nums[b], nums[c], nums[d]] such that:
    - 0 <= a, b, c, d < n
    - a, b, c, d are distinct
    - nums[a] + nums[b] + nums[c] + nums[d] == target
    
    You may return the answer in any order.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 200
    - -10^9 <= nums[i] <= 10^9
    - -10^9 <= target <= 10^9
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,0,-1,0,-2,2], target = 0
        Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
    
    Example 2:
        Input: nums = [2,2,2,2,2], target = 8
        Output: [[2,2,2,2]]
    
    APPROACH: Sort + Two Pointers (Extension of 3Sum)
    
    1. Sort the array
    2. Fix first two elements with nested loops
    3. Use two pointers for remaining two elements
    4. Skip duplicates to avoid duplicate quadruplets
    
    TIME: O(n³), SPACE: O(1) excluding output
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 3):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        for j in range(i + 1, n - 2):
            # Skip duplicates for second element
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            
            left, right = j + 1, n - 1
            
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                
                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    # Skip duplicates for third element
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    # Skip duplicates for fourth element
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
    
    return result


# =============================================================================
# PROBLEM 7: LONGEST CONSECUTIVE SEQUENCE (MEDIUM) - 45 MIN
# =============================================================================

def longest_consecutive_sequence(nums):
    """
    PROBLEM: Longest Consecutive Sequence
    
    Given an unsorted array of integers nums, return the length of the longest 
    consecutive elements sequence.
    
    You must write an algorithm that runs in O(n) time.
    
    CONSTRAINTS:
    - 0 <= nums.length <= 10^5
    - -10^9 <= nums[i] <= 10^9
    
    EXAMPLES:
    Example 1:
        Input: nums = [100,4,200,1,3,2]
        Output: 4
        Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
    
    Example 2:
        Input: nums = [0,3,7,2,5,8,4,6,0,1]
        Output: 9
    
    APPROACH: Hash Set
    
    1. Put all numbers in hash set for O(1) lookup
    2. For each number, check if it's the start of a sequence
    3. If it is, count consecutive numbers
    
    TIME: O(n), SPACE: O(n)
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        # Check if this is the start of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            longest = max(longest, current_length)
    
    return longest


# =============================================================================
# PROBLEM 8: GROUP SHIFTED STRINGS (MEDIUM) - 45 MIN
# =============================================================================

def group_shifted_strings(strings):
    """
    PROBLEM: Group Shifted Strings
    
    We can shift a string by shifting each of its letters to its successive letter.
    For example, "abc" can be shifted to be "bcd".
    
    We can keep shifting the string to form a sequence:
    "abc" -> "bcd" -> ... -> "xyz"
    
    Given an array of strings strings, group all strings that belong to the same shifting sequence.
    You may return the answer in any order.
    
    CONSTRAINTS:
    - 1 <= strings.length <= 200
    - 1 <= strings[i].length <= 50
    - strings[i] consists of lowercase English letters
    
    EXAMPLES:
    Example 1:
        Input: strings = ["abc","bcd","acef","xyz","az","ba","a","z"]
        Output: [["acef"],["a","z"],["abc","bcd","xyz"],["az","ba"]]
    
    Example 2:
        Input: strings = ["a"]
        Output: [["a"]]
    
    APPROACH: Hash Map with Shift Pattern
    
    Create a pattern for each string that represents its shift sequence.
    Use the pattern as key to group strings.
    
    TIME: O(n * m) where n = number of strings, m = average length
    SPACE: O(n * m)
    """
    def get_shift_pattern(s):
        """Get shift pattern for string (differences between consecutive chars)"""
        if len(s) <= 1:
            return tuple()
        
        pattern = []
        for i in range(1, len(s)):
            # Calculate difference, handle wrap-around
            diff = (ord(s[i]) - ord(s[i-1])) % 26
            pattern.append(diff)
        
        return tuple(pattern)
    
    groups = defaultdict(list)
    
    for string in strings:
        pattern = get_shift_pattern(string)
        groups[pattern].append(string)
    
    return list(groups.values())


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all hash table problems with comprehensive test cases
    """
    print("=== TESTING DAY 3 PROBLEMS ===\n")
    
    # Test Two Sum Hash
    print("1. Two Sum with Hash Table:")
    two_sum_tests = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1])
    ]
    
    for nums, target, expected in two_sum_tests:
        result = two_sum_hash(nums, target)
        print(f"   Input: {nums}, Target: {target}")
        print(f"   Output: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Subarray Sum Equals K
    print("2. Subarray Sum Equals K:")
    subarray_tests = [
        ([1, 1, 1], 2, 2),
        ([1, 2, 3], 3, 2),
        ([1, -1, 0], 0, 3)
    ]
    
    for nums, k, expected in subarray_tests:
        result = subarray_sum(nums, k)
        print(f"   Input: {nums}, k: {k}")
        print(f"   Output: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Top K Frequent Elements
    print("3. Top K Frequent Elements:")
    top_k_tests = [
        ([1, 1, 1, 2, 2, 3], 2, [1, 2]),
        ([1], 1, [1])
    ]
    
    for nums, k, expected in top_k_tests:
        result1 = top_k_frequent(nums, k)
        result2 = top_k_frequent_bucket_sort(nums, k)
        print(f"   Input: {nums}, k: {k}")
        print(f"   Heap approach: {sorted(result1)}")
        print(f"   Bucket sort: {sorted(result2)}")
        print(f"   Expected: {sorted(expected)}")
        success = sorted(result1) == sorted(expected) and sorted(result2) == sorted(expected)
        print(f"   ✓ Correct" if success else f"   ✗ Wrong")
        print()
    
    # Test First Missing Positive
    print("4. First Missing Positive:")
    missing_tests = [
        ([1, 2, 0], 3),
        ([3, 4, -1, 1], 2),
        ([7, 8, 9, 11, 12], 1)
    ]
    
    for nums, expected in missing_tests:
        original = nums.copy()
        result1 = first_missing_positive(nums)
        result2 = first_missing_positive_in_place(nums.copy())
        print(f"   Input: {original}")
        print(f"   Hash set: {result1}, In-place: {result2}, Expected: {expected}")
        print(f"   ✓ Correct" if result1 == result2 == expected else f"   ✗ Wrong")
        print()
    
    # Test Array Intersection
    print("5. Intersection of Two Arrays:")
    intersection_tests = [
        ([1, 2, 2, 1], [2, 2], [2]),
        ([4, 9, 5], [9, 4, 9, 8, 4], [9, 4])
    ]
    
    for nums1, nums2, expected in intersection_tests:
        result1 = intersection(nums1, nums2)
        result2 = intersection_hash_table(nums1, nums2)
        print(f"   nums1: {nums1}, nums2: {nums2}")
        print(f"   Set ops: {sorted(result1)}")
        print(f"   Manual: {sorted(result2)}")
        print(f"   Expected: {sorted(expected)}")
        success = sorted(result1) == sorted(expected) and sorted(result2) == sorted(expected)
        print(f"   ✓ Correct" if success else f"   ✗ Wrong")
        print()


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_hash_collisions():
    """
    Demonstrate hash collision concepts
    """
    print("\n=== HASH COLLISION DEMONSTRATION ===")
    
    print("In Python, hash collisions are handled automatically,")
    print("but understanding them helps with complexity analysis.")
    
    # Show hash values for demonstration
    print("\nHash values (may vary between runs):")
    test_keys = ["apple", "banana", "cherry", 123, 456]
    for key in test_keys:
        print(f"  hash('{key}'): {hash(key) % 100}")  # Mod 100 for readability
    
    print("\nCollision handling strategies:")
    print("1. Chaining: Store colliding elements in linked lists")
    print("2. Open addressing: Find next available slot")
    print("3. Python uses open addressing with random probing")


def hash_table_performance_analysis():
    """
    Analyze hash table performance characteristics
    """
    print("\n=== HASH TABLE PERFORMANCE ANALYSIS ===")
    
    print("Time Complexities:")
    print("  Average case: O(1) for insert, search, delete")
    print("  Worst case: O(n) when all keys hash to same bucket")
    print("  Best case: O(1) with good hash function and low load factor")
    
    print("\nLoad Factor Impact:")
    print("  Load factor = number of elements / table size")
    print("  Recommended: < 0.75 for good performance")
    print("  Python automatically resizes when needed")
    
    print("\nSpace Complexity:")
    print("  O(n) where n is number of key-value pairs")
    print("  Additional overhead for hash table structure")


def hash_function_properties():
    """
    Explain properties of good hash functions
    """
    print("\n=== HASH FUNCTION PROPERTIES ===")
    
    print("Properties of good hash functions:")
    print("1. Deterministic: Same input always produces same output")
    print("2. Uniform distribution: Spreads keys evenly across buckets")
    print("3. Fast computation: O(1) or O(k) where k is key length")
    print("4. Avalanche effect: Small input changes cause large output changes")
    
    print("\nPython's hash function:")
    print("  - Uses SipHash algorithm for strings")
    print("  - Includes random seed for security")
    print("  - Different hash values between program runs")


def common_hash_table_patterns():
    """
    Demonstrate common patterns with hash tables
    """
    print("\n=== COMMON HASH TABLE PATTERNS ===")
    
    # Pattern 1: Frequency counting
    print("1. Frequency Counting Pattern:")
    arr = [1, 2, 3, 2, 1, 3, 1]
    freq = Counter(arr)
    print(f"   Array: {arr}")
    print(f"   Frequencies: {dict(freq)}")
    
    # Pattern 2: Complement lookup (Two Sum)
    print("\n2. Complement Lookup Pattern (Two Sum):")
    nums = [2, 7, 11, 15]
    target = 9
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            print(f"   Found pair: indices {seen[complement]} and {i}, values {complement} and {num}")
            break
        seen[num] = i
    
    # Pattern 3: Grouping by computed key
    print("\n3. Grouping Pattern:")
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    groups = defaultdict(list)
    for word in words:
        key = ''.join(sorted(word))
        groups[key].append(word)
    print(f"   Words: {words}")
    print(f"   Anagram groups: {dict(groups)}")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Demonstrate hash collisions
    demonstrate_hash_collisions()
    
    # Performance analysis
    hash_table_performance_analysis()
    
    # Hash function properties
    hash_function_properties()
    
    # Common patterns
    common_hash_table_patterns()
    
    print("\n" + "="*70)
    print("DAY 3 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. Hash tables provide O(1) average case for insert/search/delete")
    print("2. Perfect for frequency counting and complement lookups")
    print("3. Two sum pattern: store complements for O(n) solution")
    print("4. Prefix sum + hash map: solve subarray sum problems")
    print("5. Sets for uniqueness checking and intersections")
    print("6. Counter for frequency-based problems")
    print("7. defaultdict avoids key existence checking")
    print("\nTransition: Day 3→4 - From hash tables to linked lists")
    print("- Hash tables support pointer-based data structures")
    print("- Fast lookups help with linked list problems (cycle detection)")
    print("- Next: Linear data structures with pointer manipulation")
    print("\nNext: Day 4 - Linked Lists") 