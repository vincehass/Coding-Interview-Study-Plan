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


# Problem 1: Two Sum Revisited - Core hash table pattern
def two_sum_hash(nums, target):
    """
    Find indices of two numbers that add up to target
    
    Hash table approach: Store complements and their indices
    This is the fundamental hash table pattern for pair problems
    
    Time: O(n), Space: O(n)
    """
    num_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    
    return []


# Problem 2: Subarray Sum Equals K - Prefix sum with hash table
def subarray_sum(nums, k):
    """
    Count number of continuous subarrays whose sum equals k
    
    Key insight: If prefix_sum[j] - prefix_sum[i] = k, then
    subarray from i+1 to j has sum k
    
    Algorithm:
    1. Maintain running prefix sum
    2. For each position, check if (prefix_sum - k) exists in hash map
    3. Count occurrences of each prefix sum
    
    Time: O(n), Space: O(n)
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


# Problem 3: Top K Frequent Elements - Hash table + heap
def top_k_frequent(nums, k):
    """
    Find k most frequent elements
    
    Approach 1: Hash map + heap
    1. Count frequencies with hash map
    2. Use min-heap of size k to find top k
    
    Time: O(n log k), Space: O(n)
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
    Alternative approach using bucket sort
    
    Since frequency is bounded by array length, we can use bucket sort
    Time: O(n), Space: O(n)
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


# Problem 4: First Missing Positive - Hash set for O(1) lookups
def first_missing_positive(nums):
    """
    Find smallest missing positive integer
    
    Approach: Use hash set for O(1) lookups
    1. Add all numbers to set
    2. Check consecutive positive integers starting from 1
    
    Time: O(n), Space: O(n)
    """
    num_set = set(nums)
    
    positive = 1
    while positive in num_set:
        positive += 1
    
    return positive


def first_missing_positive_in_place(nums):
    """
    In-place approach using array as hash table
    
    Key insight: Answer is in range [1, n+1] where n = len(nums)
    Use array indices as hash keys by placing each number at index (number-1)
    
    Time: O(n), Space: O(1)
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


# Problem 5: Intersection of Two Arrays - Set operations
def intersection(nums1, nums2):
    """
    Find intersection of two arrays (unique elements)
    
    Approach: Convert to sets and use intersection
    Time: O(n + m), Space: O(min(n, m))
    """
    return list(set(nums1) & set(nums2))


def intersection_hash_table(nums1, nums2):
    """
    Manual implementation using hash table
    Shows explicit hash table logic
    """
    seen = set(nums1)
    result = set()
    
    for num in nums2:
        if num in seen:
            result.add(num)
    
    return list(result)


# ADVANCED PROBLEMS FOR DEEPER UNDERSTANDING

def four_sum(nums, target):
    """
    Find all unique quadruplets that sum to target
    
    Extension of two sum using hash maps
    Time: O(n²), Space: O(n²)
    """
    nums.sort()
    n = len(nums)
    result = []
    
    for i in range(n - 3):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
            
        for j in range(i + 1, n - 2):
            # Skip duplicates for second element
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            
            # Use two pointers for remaining two elements
            left, right = j + 1, n - 1
            
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                
                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    # Skip duplicates
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
    
    return result


def longest_consecutive_sequence(nums):
    """
    Find length of longest consecutive elements sequence
    
    Hash set approach: O(n) time instead of O(n log n) sorting
    
    Key insight: Only start counting from beginning of sequence
    Time: O(n), Space: O(n)
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        # Only start counting if this is the beginning of sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            max_length = max(max_length, current_length)
    
    return max_length


def group_shifted_strings(strings):
    """
    Group strings that are shifts of each other
    
    Example: "abc" and "bcd" are shifts (each character shifted by 1)
    
    Key insight: Compute shift pattern as hash key
    Time: O(n * m) where n = number of strings, m = average length
    Space: O(n * m)
    """
    def get_shift_pattern(s):
        """Get pattern of character shifts"""
        if len(s) <= 1:
            return tuple()
        
        pattern = []
        for i in range(1, len(s)):
            # Calculate shift from previous character
            shift = (ord(s[i]) - ord(s[i-1])) % 26
            pattern.append(shift)
        
        return tuple(pattern)
    
    groups = defaultdict(list)
    
    for s in strings:
        pattern = get_shift_pattern(s)
        groups[pattern].append(s)
    
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