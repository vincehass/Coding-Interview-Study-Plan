"""
=============================================================================
                        WEEK 1 SOLUTION FILE
                     COMPLETE SOLUTIONS & VARIANTS
                           Meta Interview Preparation
=============================================================================

This file contains complete solutions for all Week 1 practice problems with
multiple approaches, variants, and comprehensive test cases.

TOPICS COVERED:
- Arrays & Two Pointers
- Strings & Patterns  
- Hash Tables & Sets
- Linked Lists
- Stacks & Queues

=============================================================================
"""

from collections import defaultdict, Counter, deque
from typing import List, Optional
import heapq

# =============================================================================
# ARRAYS & TWO POINTERS SOLUTIONS
# =============================================================================

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    PROBLEM: Two Sum
    TIME: O(n), SPACE: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def two_sum_sorted(nums: List[int], target: int) -> List[int]:
    """
    VARIANT: Two Sum on sorted array using two pointers
    TIME: O(n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

def two_sum_all_pairs(nums: List[int], target: int) -> List[List[int]]:
    """
    VARIANT: Find all pairs that sum to target
    TIME: O(n), SPACE: O(n)
    """
    seen = {}
    result = []
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            for j in seen[complement]:
                result.append([j, i])
        
        if num not in seen:
            seen[num] = []
        seen[num].append(i)
    
    return result

def three_sum(nums: List[int]) -> List[List[int]]:
    """
    PROBLEM: 3Sum
    TIME: O(n²), SPACE: O(1) excluding output
    """
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
            
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                    
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return result

def three_sum_closest(nums: List[int], target: int) -> int:
    """
    VARIANT: 3Sum Closest
    TIME: O(n²), SPACE: O(1)
    """
    nums.sort()
    closest_sum = float('inf')
    
    for i in range(len(nums) - 2):
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum
            
            if current_sum < target:
                left += 1
            elif current_sum > target:
                right -= 1
            else:
                return current_sum
    
    return closest_sum

def four_sum(nums: List[int], target: int) -> List[List[int]]:
    """
    VARIANT: 4Sum
    TIME: O(n³), SPACE: O(1) excluding output
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i-1]:
            continue
            
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j-1]:
                continue
                
            left, right = j + 1, n - 1
            
            while left < right:
                total = nums[i] + nums[j] + nums[left] + nums[right]
                
                if total == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                        
                    left += 1
                    right -= 1
                elif total < target:
                    left += 1
                else:
                    right -= 1
    
    return result

def container_with_most_water(height: List[int]) -> int:
    """
    PROBLEM: Container With Most Water
    TIME: O(n), SPACE: O(1)
    """
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        current_area = width * min(height[left], height[right])
        max_area = max(max_area, current_area)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

def trapping_rain_water(height: List[int]) -> int:
    """
    VARIANT: Trapping Rain Water
    TIME: O(n), SPACE: O(1)
    """
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water_trapped = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water_trapped += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water_trapped += right_max - height[right]
            right -= 1
    
    return water_trapped

# =============================================================================
# STRINGS & PATTERNS SOLUTIONS
# =============================================================================

def longest_substring_without_repeating(s: str) -> int:
    """
    PROBLEM: Longest Substring Without Repeating Characters
    TIME: O(n), SPACE: O(min(m,n))
    """
    char_index = {}
    left = 0
    max_length = 0
    
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        
        char_index[char] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length

def longest_substring_k_distinct(s: str, k: int) -> int:
    """
    VARIANT: Longest Substring with At Most K Distinct Characters
    TIME: O(n), SPACE: O(k)
    """
    if k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_length = 0
    
    for right, char in enumerate(s):
        char_count[char] = char_count.get(char, 0) + 1
        
        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1
            if char_count[left_char] == 0:
                del char_count[left_char]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

def longest_repeating_character_replacement(s: str, k: int) -> int:
    """
    VARIANT: Longest Repeating Character Replacement
    TIME: O(n), SPACE: O(26)
    """
    char_count = {}
    left = 0
    max_count = 0
    max_length = 0
    
    for right, char in enumerate(s):
        char_count[char] = char_count.get(char, 0) + 1
        max_count = max(max_count, char_count[char])
        
        # If window size - max_count > k, shrink window
        if right - left + 1 - max_count > k:
            left_char = s[left]
            char_count[left_char] -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    PROBLEM: Group Anagrams
    TIME: O(N * K * log K), SPACE: O(N * K)
    """
    anagram_groups = defaultdict(list)
    
    for s in strs:
        key = ''.join(sorted(s))
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())

def group_anagrams_counting(strs: List[str]) -> List[List[str]]:
    """
    VARIANT: Group Anagrams using character counting
    TIME: O(N * K), SPACE: O(N * K)
    """
    anagram_groups = defaultdict(list)
    
    for s in strs:
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        key = tuple(count)
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())

def valid_parentheses(s: str) -> bool:
    """
    PROBLEM: Valid Parentheses
    TIME: O(n), SPACE: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return len(stack) == 0

def min_remove_to_make_valid(s: str) -> str:
    """
    VARIANT: Minimum Remove to Make Valid Parentheses
    TIME: O(n), SPACE: O(n)
    """
    # First pass: remove invalid closing parentheses
    first_pass = []
    open_count = 0
    
    for char in s:
        if char == '(':
            first_pass.append(char)
            open_count += 1
        elif char == ')' and open_count > 0:
            first_pass.append(char)
            open_count -= 1
        elif char != ')':
            first_pass.append(char)
    
    # Second pass: remove extra opening parentheses
    result = []
    open_needed = 0
    
    for char in reversed(first_pass):
        if char == ')':
            result.append(char)
            open_needed += 1
        elif char == '(' and open_needed > 0:
            result.append(char)
            open_needed -= 1
        elif char != '(':
            result.append(char)
    
    return ''.join(reversed(result))

# =============================================================================
# HASH TABLES & SETS SOLUTIONS
# =============================================================================

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    PROBLEM: Top K Frequent Elements
    TIME: O(n log k), SPACE: O(n)
    """
    counter = Counter(nums)
    return heapq.nlargest(k, counter.keys(), key=counter.get)

def top_k_frequent_bucket_sort(nums: List[int], k: int) -> List[int]:
    """
    VARIANT: Top K Frequent using bucket sort
    TIME: O(n), SPACE: O(n)
    """
    counter = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in counter.items():
        buckets[freq].append(num)
    
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result

def top_k_frequent_words(words: List[str], k: int) -> List[str]:
    """
    VARIANT: Top K Frequent Words (lexicographically sorted)
    TIME: O(n log k), SPACE: O(n)
    """
    counter = Counter(words)
    
    # Use heap with custom comparison
    heap = []
    for word, freq in counter.items():
        heapq.heappush(heap, (-freq, word))
    
    result = []
    for _ in range(k):
        freq, word = heapq.heappop(heap)
        result.append(word)
    
    return result

def longest_consecutive(nums: List[int]) -> int:
    """
    PROBLEM: Longest Consecutive Sequence
    TIME: O(n), SPACE: O(n)
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            longest = max(longest, current_length)
    
    return longest

def longest_consecutive_with_path(nums: List[int]) -> List[int]:
    """
    VARIANT: Return the actual longest consecutive sequence
    TIME: O(n), SPACE: O(n)
    """
    if not nums:
        return []
    
    num_set = set(nums)
    longest_seq = []
    
    for num in num_set:
        if num - 1 not in num_set:
            current_seq = [num]
            current_num = num
            
            while current_num + 1 in num_set:
                current_num += 1
                current_seq.append(current_num)
            
            if len(current_seq) > len(longest_seq):
                longest_seq = current_seq
    
    return longest_seq

# =============================================================================
# LINKED LISTS SOLUTIONS
# =============================================================================

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    PROBLEM: Reverse Linked List (Iterative)
    TIME: O(n), SPACE: O(1)
    """
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev

def reverse_linked_list_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    VARIANT: Reverse Linked List (Recursive)
    TIME: O(n), SPACE: O(n)
    """
    if not head or not head.next:
        return head
    
    new_head = reverse_linked_list_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return new_head

def reverse_list_range(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    VARIANT: Reverse Linked List II (between positions)
    TIME: O(n), SPACE: O(1)
    """
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    # Move to position before left
    for _ in range(left - 1):
        prev = prev.next
    
    # Reverse the sublist
    current = prev.next
    for _ in range(right - left):
        next_temp = current.next
        current.next = next_temp.next
        next_temp.next = prev.next
        prev.next = next_temp
    
    return dummy.next

def merge_two_sorted_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    PROBLEM: Merge Two Sorted Lists
    TIME: O(n + m), SPACE: O(1)
    """
    dummy = ListNode(0)
    current = dummy
    
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    current.next = list1 or list2
    return dummy.next

def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    VARIANT: Merge K Sorted Lists
    TIME: O(N log k), SPACE: O(1)
    """
    if not lists:
        return None
    
    while len(lists) > 1:
        merged_lists = []
        
        for i in range(0, len(lists), 2):
            list1 = lists[i]
            list2 = lists[i + 1] if i + 1 < len(lists) else None
            merged_lists.append(merge_two_sorted_lists(list1, list2))
        
        lists = merged_lists
    
    return lists[0]

def has_cycle(head: Optional[ListNode]) -> bool:
    """
    PROBLEM: Linked List Cycle
    TIME: O(n), SPACE: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    VARIANT: Linked List Cycle II (find start of cycle)
    TIME: O(n), SPACE: O(1)
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect cycle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Phase 2: Find start of cycle
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

# =============================================================================
# STACKS & QUEUES SOLUTIONS
# =============================================================================

def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    PROBLEM: Daily Temperatures
    TIME: O(n), SPACE: O(n)
    """
    result = [0] * len(temperatures)
    stack = []
    
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    
    return result

def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    VARIANT: Next Greater Element I
    TIME: O(n + m), SPACE: O(n)
    """
    next_greater = {}
    stack = []
    
    # Build next greater mapping for nums2
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # For remaining elements in stack, no greater element exists
    while stack:
        next_greater[stack.pop()] = -1
    
    # Build result for nums1
    return [next_greater[num] for num in nums1]

def next_greater_element_circular(nums: List[int]) -> List[int]:
    """
    VARIANT: Next Greater Element II (circular array)
    TIME: O(n), SPACE: O(n)
    """
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # Process the array twice to handle circular nature
    for i in range(2 * n):
        while stack and nums[stack[-1]] < nums[i % n]:
            result[stack.pop()] = nums[i % n]
        
        if i < n:
            stack.append(i)
    
    return result

def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
    """
    PROBLEM: Sliding Window Maximum
    TIME: O(n), SPACE: O(k)
    """
    if not nums or k == 0:
        return []
    
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements from back
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result if window is of size k
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

def sliding_window_minimum(nums: List[int], k: int) -> List[int]:
    """
    VARIANT: Sliding Window Minimum
    TIME: O(n), SPACE: O(k)
    """
    if not nums or k == 0:
        return []
    
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove larger elements from back
        while dq and nums[dq[-1]] > nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result if window is of size k
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Create linked list from list of values"""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """Convert linked list to list of values"""
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result

def create_cycle(head: Optional[ListNode], pos: int) -> Optional[ListNode]:
    """Create cycle in linked list at given position"""
    if not head or pos < 0:
        return head
    
    cycle_start = None
    current = head
    index = 0
    
    while current.next:
        if index == pos:
            cycle_start = current
        current = current.next
        index += 1
    
    if cycle_start:
        current.next = cycle_start
    
    return head

# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def test_all_week1_solutions():
    """Comprehensive test suite for all Week 1 solutions"""
    
    print("=" * 80)
    print("                    WEEK 1 COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Test Two Sum variants
    print("\n🧪 TESTING TWO SUM VARIANTS")
    print("-" * 50)
    
    test_cases_two_sum = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1]),
        ([-1, -2, -3, -4, -5], -8, [2, 4]),
        ([0, 4, 3, 0], 0, [0, 3]),
        ([1, 2, 3, 4, 5], 8, [2, 4]),
        ([5, 5, 11], 10, [0, 1])
    ]
    
    for i, (nums, target, expected) in enumerate(test_cases_two_sum, 1):
        result = two_sum(nums.copy(), target)
        print(f"Test {i}: nums={nums}, target={target}")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  ✅ PASS" if result == expected else f"  ❌ FAIL")
    
    # Test all pairs variant
    print(f"\nTesting Two Sum All Pairs:")
    all_pairs_result = two_sum_all_pairs([1, 2, 3, 2, 1], 3)
    print(f"  All pairs that sum to 3: {all_pairs_result}")
    
    # Test Three Sum
    print("\n🧪 TESTING THREE SUM VARIANTS")
    print("-" * 50)
    
    test_cases_three_sum = [
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([0, 1, 1], []),
        ([0, 0, 0], [[0, 0, 0]]),
        ([1, 2, -2, -1], []),
        ([-2, 0, 1, 1, 2], [[-2, 0, 2], [-2, 1, 1]])
    ]
    
    for i, (nums, expected) in enumerate(test_cases_three_sum, 1):
        result = three_sum(nums.copy())
        result_set = set(tuple(sorted(trip)) for trip in result)
        expected_set = set(tuple(sorted(trip)) for trip in expected)
        
        print(f"Test {i}: nums={nums}")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  ✅ PASS" if result_set == expected_set else f"  ❌ FAIL")
    
    # Test Three Sum Closest
    print(f"\nTesting Three Sum Closest:")
    closest_result = three_sum_closest([-1, 2, 1, -4], 1)
    print(f"  Closest sum to 1: {closest_result} (Expected: 2)")
    
    # Test Container With Most Water
    print("\n🧪 TESTING CONTAINER WITH MOST WATER")
    print("-" * 50)
    
    test_cases_container = [
        ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
        ([1, 1], 1),
        ([4, 3, 2, 1, 4], 16),
        ([1, 2, 1], 2),
        ([2, 1], 1)
    ]
    
    for i, (height, expected) in enumerate(test_cases_container, 1):
        result = container_with_most_water(height)
        print(f"Test {i}: height={height}")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  ✅ PASS" if result == expected else f"  ❌ FAIL")
    
    # Test Trapping Rain Water
    print(f"\nTesting Trapping Rain Water:")
    rain_result = trapping_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
    print(f"  Water trapped: {rain_result} (Expected: 6)")
    
    # Test String Problems
    print("\n🧪 TESTING STRING PROBLEMS")
    print("-" * 50)
    
    test_cases_substring = [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("", 0),
        ("abcdef", 6),
        ("aab", 2)
    ]
    
    for i, (s, expected) in enumerate(test_cases_substring, 1):
        result = longest_substring_without_repeating(s)
        print(f"Test {i}: s='{s}'")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  ✅ PASS" if result == expected else f"  ❌ FAIL")
    
    # Test Group Anagrams
    print(f"\nTesting Group Anagrams:")
    anagram_result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    print(f"  Grouped anagrams: {anagram_result}")
    
    # Test Valid Parentheses
    print("\n🧪 TESTING PARENTHESES PROBLEMS")
    print("-" * 50)
    
    test_cases_parentheses = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True),
        ("", True)
    ]
    
    for i, (s, expected) in enumerate(test_cases_parentheses, 1):
        result = valid_parentheses(s)
        print(f"Test {i}: s='{s}'")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  ✅ PASS" if result == expected else f"  ❌ FAIL")
    
    # Test Hash Table Problems
    print("\n🧪 TESTING HASH TABLE PROBLEMS")
    print("-" * 50)
    
    # Top K Frequent
    top_k_result = top_k_frequent([1, 1, 1, 2, 2, 3], 2)
    print(f"Top 2 frequent elements: {top_k_result} (Expected: [1, 2])")
    
    # Longest Consecutive
    consecutive_result = longest_consecutive([100, 4, 200, 1, 3, 2])
    print(f"Longest consecutive sequence: {consecutive_result} (Expected: 4)")
    
    # Test Linked List Problems
    print("\n🧪 TESTING LINKED LIST PROBLEMS")
    print("-" * 50)
    
    # Reverse Linked List
    head = create_linked_list([1, 2, 3, 4, 5])
    reversed_head = reverse_linked_list(head)
    reversed_list = linked_list_to_list(reversed_head)
    print(f"Reversed [1,2,3,4,5]: {reversed_list} (Expected: [5,4,3,2,1])")
    
    # Merge Two Sorted Lists
    list1 = create_linked_list([1, 2, 4])
    list2 = create_linked_list([1, 3, 4])
    merged = merge_two_sorted_lists(list1, list2)
    merged_list = linked_list_to_list(merged)
    print(f"Merged [1,2,4] and [1,3,4]: {merged_list} (Expected: [1,1,2,3,4,4])")
    
    # Has Cycle
    cycle_head = create_linked_list([3, 2, 0, -4])
    create_cycle(cycle_head, 1)  # Create cycle at position 1
    has_cycle_result = has_cycle(cycle_head)
    print(f"Has cycle: {has_cycle_result} (Expected: True)")
    
    # Test Stack/Queue Problems
    print("\n🧪 TESTING STACK/QUEUE PROBLEMS")
    print("-" * 50)
    
    # Daily Temperatures
    temps_result = daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])
    print(f"Daily temperatures: {temps_result} (Expected: [1,1,4,2,1,1,0,0])")
    
    # Sliding Window Maximum
    window_max_result = sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3)
    print(f"Sliding window maximum: {window_max_result} (Expected: [3,3,5,5,6,7])")
    
    print("\n" + "=" * 80)
    print("                    TESTING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_all_week1_solutions() 