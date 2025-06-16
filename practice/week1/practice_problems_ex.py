"""
=============================================================================
                        WEEK 1 PRACTICE PROBLEMS
                           Meta Interview Preparation
=============================================================================

This file contains practice problems for Week 1. Work through these problems
independently to reinforce your learning from the main study materials.

INSTRUCTIONS:
1. Read each problem statement and constraints carefully
2. Understand the examples and expected outputs
3. Write your solution in the designated space
4. Test your solution with the provided test cases
5. Compare with the reference implementation when stuck

=============================================================================
"""

# ARRAYS & TWO POINTERS PRACTICE

def two_sum_practice(nums, target):
    """
    PROBLEM: Two Sum
    
    DESCRIPTION:
    Given an array of integers `nums` and an integer `target`, return indices of 
    the two numbers such that they add up to `target`.
    
    You may assume that each input would have exactly one solution, and you may 
    not use the same element twice.
    
    You can return the answer in any order.
    
    CONSTRAINTS:
    - 2 <= nums.length <= 10^4
    - -10^9 <= nums[i] <= 10^9
    - -10^9 <= target <= 10^9
    - Only one valid answer exists
    
    EXAMPLES:
    Example 1:
        Input: nums = [2,7,11,15], target = 9
        Output: [0,1]
        Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
    
    Example 2:
        Input: nums = [3,2,4], target = 6
        Output: [1,2]
    
    Example 3:
        Input: nums = [3,3], target = 6
        Output: [0,1]
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


def three_sum_practice(nums):
    """
    PROBLEM: 3Sum
    
    DESCRIPTION:
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
    
    EXPECTED TIME COMPLEXITY: O(n²)
    EXPECTED SPACE COMPLEXITY: O(1) excluding output array
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


def container_with_most_water_practice(height):
    """
    PROBLEM: Container With Most Water
    
    DESCRIPTION:
    You are given an integer array `height` of length n. There are n vertical 
    lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
    
    Find two lines that together with the x-axis form a container that can contain 
    the most water.
    
    Return the maximum amount of water a container can store.
    
    Notice that you may not slant the container.
    
    CONSTRAINTS:
    - n == height.length
    - 2 <= n <= 10^5
    - 0 <= height[i] <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: height = [1,8,6,2,5,4,8,3,7]
        Output: 49
        Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
        In this case, the max area of water the container can contain is 49.
    
    Example 2:
        Input: height = [1,1]
        Output: 1
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


# STRINGS & PATTERNS PRACTICE

def longest_substring_without_repeating_practice(s):
    """
    PROBLEM: Longest Substring Without Repeating Characters
    
    DESCRIPTION:
    Given a string s, find the length of the longest substring without repeating characters.
    
    CONSTRAINTS:
    - 0 <= s.length <= 5 * 10^4
    - s consists of English letters, digits, symbols and spaces.
    
    EXAMPLES:
    Example 1:
        Input: s = "abcabcbb"
        Output: 3
        Explanation: The answer is "abc", with the length of 3.
    
    Example 2:
        Input: s = "bbbbb"
        Output: 1
        Explanation: The answer is "b", with the length of 1.
    
    Example 3:
        Input: s = "pwwkew"
        Output: 3
        Explanation: The answer is "wke", with the length of 3.
        Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(min(m,n)) where m is size of charset
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


def group_anagrams_practice(strs):
    """
    PROBLEM: Group Anagrams
    
    DESCRIPTION:
    Given an array of strings strs, group the anagrams together. You can return 
    the answer in any order.
    
    An Anagram is a word or phrase formed by rearranging the letters of a different 
    word or phrase, typically using all the original letters exactly once.
    
    CONSTRAINTS:
    - 1 <= strs.length <= 10^4
    - 0 <= strs[i].length <= 100
    - strs[i] consists of lowercase English letters only.
    
    EXAMPLES:
    Example 1:
        Input: strs = ["eat","tea","tan","ate","nat","bat"]
        Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
    
    Example 2:
        Input: strs = [""]
        Output: [[""]]
    
    Example 3:
        Input: strs = ["a"]
        Output: [["a"]]
    
    EXPECTED TIME COMPLEXITY: O(N * K * log K) where N is length of strs, K is max length of string
    EXPECTED SPACE COMPLEXITY: O(N * K)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


def valid_parentheses_practice(s):
    """
    PROBLEM: Valid Parentheses
    
    DESCRIPTION:
    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', 
    determine if the input string is valid.
    
    An input string is valid if:
    1. Open brackets must be closed by the same type of brackets.
    2. Open brackets must be closed in the correct order.
    3. Every close bracket has a corresponding open bracket of the same type.
    
    CONSTRAINTS:
    - 1 <= s.length <= 10^4
    - s consists of parentheses only '()[]{}'.
    
    EXAMPLES:
    Example 1:
        Input: s = "()"
        Output: true
    
    Example 2:
        Input: s = "()[]{}"
        Output: true
    
    Example 3:
        Input: s = "(]"
        Output: false
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


# HASH TABLES & SETS PRACTICE

def top_k_frequent_practice(nums, k):
    """
    PROBLEM: Top K Frequent Elements
    
    DESCRIPTION:
    Given an integer array nums and an integer k, return the k most frequent elements. 
    You may return the answer in any order.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^5
    - k is in the range [1, the number of unique elements in the array].
    - It's guaranteed that the answer is unique.
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,1,1,2,2,3], k = 2
        Output: [1,2]
    
    Example 2:
        Input: nums = [1], k = 1
        Output: [1]
    
    FOLLOW-UP: Your algorithm's time complexity must be better than O(n log n), 
    where n is the array's size.
    
    EXPECTED TIME COMPLEXITY: O(n log k) using heap, O(n) using bucket sort
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


def longest_consecutive_practice(nums):
    """
    PROBLEM: Longest Consecutive Sequence
    
    DESCRIPTION:
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
        Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. 
        Therefore its length is 4.
    
    Example 2:
        Input: nums = [0,3,7,2,5,8,4,6,0,1]
        Output: 9
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


# LINKED LISTS PRACTICE

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_linked_list_practice(head):
    """
    PROBLEM: Reverse Linked List
    
    DESCRIPTION:
    Given the head of a singly linked list, reverse the list, and return the reversed list.
    
    CONSTRAINTS:
    - The number of nodes in the list is the range [0, 5000].
    - -5000 <= Node.val <= 5000
    
    EXAMPLES:
    Example 1:
        Input: head = [1,2,3,4,5]
        Output: [5,4,3,2,1]
    
    Example 2:
        Input: head = [1,2]
        Output: [2,1]
    
    Example 3:
        Input: head = []
        Output: []
    
    FOLLOW-UP: A linked list can be reversed either iteratively or recursively. 
    Could you implement both?
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1) iterative, O(n) recursive
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


def merge_two_sorted_lists_practice(list1, list2):
    """
    PROBLEM: Merge Two Sorted Lists
    
    DESCRIPTION:
    You are given the heads of two sorted linked lists list1 and list2.
    
    Merge the two lists in a one sorted list. The list should be made by splicing 
    together the nodes of the first two lists.
    
    Return the head of the merged linked list.
    
    CONSTRAINTS:
    - The number of nodes in both lists is in the range [0, 50].
    - -100 <= Node.val <= 100
    - Both list1 and list2 are sorted in non-decreasing order.
    
    EXAMPLES:
    Example 1:
        Input: list1 = [1,2,4], list2 = [1,3,4]
        Output: [1,1,2,3,4,4]
    
    Example 2:
        Input: list1 = [], list2 = []
        Output: []
    
    Example 3:
        Input: list1 = [], list2 = [0]
        Output: [0]
    
    EXPECTED TIME COMPLEXITY: O(n + m)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


def has_cycle_practice(head):
    """
    PROBLEM: Linked List Cycle
    
    DESCRIPTION:
    Given head, the head of a linked list, determine if the linked list has a cycle in it.
    
    There is a cycle in a linked list if there is some node in the list that can be 
    reached again by continuously following the next pointer. Internally, pos is used 
    to denote the index of the node that tail's next pointer is connected to. Note that 
    pos is not passed as a parameter.
    
    Return true if there is a cycle in the linked list. Otherwise, return false.
    
    CONSTRAINTS:
    - The number of the nodes in the list is in the range [0, 10^4].
    - -10^5 <= Node.val <= 10^5
    - pos is -1 or a valid index in the linked-list.
    
    EXAMPLES:
    Example 1:
        Input: head = [3,2,0,-4], pos = 1
        Output: true
        Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
    
    Example 2:
        Input: head = [1,2], pos = 0
        Output: true
        Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.
    
    Example 3:
        Input: head = [1], pos = -1
        Output: false
        Explanation: There is no cycle in the linked list.
    
    FOLLOW-UP: Can you solve it using O(1) (i.e. constant) memory?
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(1) using Floyd's algorithm
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


# STACKS & QUEUES PRACTICE

def daily_temperatures_practice(temperatures):
    """
    PROBLEM: Daily Temperatures
    
    DESCRIPTION:
    Given an array of integers temperatures represents the daily temperatures, return 
    an array answer such that answer[i] is the number of days you have to wait after 
    the ith day to get a warmer temperature. If there is no future day for which this 
    is possible, keep answer[i] == 0 instead.
    
    CONSTRAINTS:
    - 1 <= temperatures.length <= 10^5
    - 30 <= temperatures[i] <= 100
    
    EXAMPLES:
    Example 1:
        Input: temperatures = [73,74,75,71,69,72,76,73]
        Output: [1,1,4,2,1,1,0,0]
        Explanation:
        For temperature 73, we need to wait 1 day to get 74.
        For temperature 74, we need to wait 1 day to get 75.
        For temperature 75, we need to wait 4 days to get 76.
        And so on...
    
    Example 2:
        Input: temperatures = [30,40,50,60]
        Output: [1,1,1,0]
    
    Example 3:
        Input: temperatures = [30,60,90]
        Output: [1,1,0]
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


def sliding_window_maximum_practice(nums, k):
    """
    PROBLEM: Sliding Window Maximum
    
    DESCRIPTION:
    You are given an array of integers nums, there is a sliding window of size k 
    which is moving from the very left of the array to the very right. You can only 
    see the k numbers in the window. Each time the sliding window moves right by one position.
    
    Return the max sliding window.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    - 1 <= k <= nums.length
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
        Output: [3,3,5,5,6,7]
        Explanation: 
        Window position                Max
        ---------------               -----
        [1  3  -1] -3  5  3  6  7       3
         1 [3  -1  -3] 5  3  6  7       3
         1  3 [-1  -3  5] 3  6  7       5
         1  3  -1 [-3  5  3] 6  7       5
         1  3  -1  -3 [5  3  6] 7       6
         1  3  -1  -3  5 [3  6  7]      7
    
    Example 2:
        Input: nums = [1], k = 1
        Output: [1]
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(k)
    
    YOUR SOLUTION:
    """
    # Write your solution here
    pass


# TEST CASES FOR VERIFICATION

def test_week1_practice():
    """Test your solutions with comprehensive test cases"""
    
    print("=== TESTING WEEK 1 PRACTICE SOLUTIONS ===\n")
    
    # Test Two Sum
    print("1. Two Sum:")
    test_cases = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1])
    ]
    for nums, target, expected in test_cases:
        result = two_sum_practice(nums, target)
        print(f"   Input: nums={nums}, target={target}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Three Sum  
    print("2. Three Sum:")
    test_cases = [
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([0, 1, 1], []),
        ([0, 0, 0], [[0, 0, 0]])
    ]
    for nums, expected in test_cases:
        result = three_sum_practice(nums)
        print(f"   Input: {nums}")
        print(f"   Output: {result}")
        print(f"   Expected: {expected}")
        print()
    
    # Test Container With Most Water
    print("3. Container With Most Water:")
    test_cases = [
        ([1,8,6,2,5,4,8,3,7], 49),
        ([1,1], 1)
    ]
    for height, expected in test_cases:
        result = container_with_most_water_practice(height)
        print(f"   Input: height={height}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Longest Substring
    print("4. Longest Substring Without Repeating:")
    test_cases = [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3)
    ]
    for s, expected in test_cases:
        result = longest_substring_without_repeating_practice(s)
        print(f"   Input: s={s}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Group Anagrams
    print("5. Group Anagrams:")
    test_cases = [
        (["eat","tea","tan","ate","nat","bat"], [["bat"],["nat","tan"],["ate","eat","tea"]]),
        ([""], [[""]]),
        (["a"], [["a"]])
    ]
    for strs, expected in test_cases:
        result = group_anagrams_practice(strs)
        print(f"   Input: strs={strs}")
        print(f"   Output: {result}")
        print(f"   Expected: {expected}")
        print()
    
    # Test Valid Parentheses
    print("6. Valid Parentheses:")
    test_cases = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False)
    ]
    for s, expected in test_cases:
        result = valid_parentheses_practice(s)
        print(f"   Input: s={s}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Top K Frequent
    print("7. Top K Frequent:")
    test_cases = [
        ([1,1,1,2,2,3], 2, [1,2]),
        ([1], 1, [1])
    ]
    for nums, k, expected in test_cases:
        result = top_k_frequent_practice(nums, k)
        print(f"   Input: nums={nums}, k={k}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Longest Consecutive
    print("8. Longest Consecutive:")
    test_cases = [
        ([100,4,200,1,3,2], 4),
        ([0,3,7,2,5,8,4,6,0,1], 9)
    ]
    for nums, expected in test_cases:
        result = longest_consecutive_practice(nums)
        print(f"   Input: nums={nums}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Add more comprehensive tests for all problems...
    print("Continue implementing and testing other problems...")


# REFERENCE SOLUTIONS (Uncomment when you want to check your work)

def two_sum_reference(nums, target):
    """Reference solution for Two Sum"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


def three_sum_reference(nums):
    """Reference solution for Three Sum"""
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


def container_with_most_water_reference(height):
    """Reference solution for Container With Most Water"""
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


# Add more reference solutions...


if __name__ == "__main__":
    print("Week 1 Practice Problems")
    print("========================")
    print("Read each problem statement carefully.")
    print("Understand constraints and examples.")
    print("Write your solution in the designated space.")
    print("Run test_week1_practice() to check your solutions.")
    print()
    
    # Uncomment to run tests
    # test_week1_practice() 