"""
=============================================================================
                        WEEK 2 PRACTICE PROBLEMS
                     TREES & BINARY SEARCH
                           Meta Interview Preparation
=============================================================================

This file contains practice problems for Week 2. Work through these problems
independently to reinforce your learning from the main study materials.

INSTRUCTIONS:
1. Read each problem statement and constraints carefully
2. Understand the examples and expected outputs
3. Write your solution in the designated space
4. Test your solution with the provided test cases
5. Compare with the reference implementation when stuck

=============================================================================
"""

from collections import deque
from typing import List, Optional
import heapq


# TreeNode definition
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# BINARY TREES PRACTICE

def inorder_traversal_practice(root):
    """
    PROBLEM: Binary Tree Inorder Traversal
    
    DESCRIPTION:
    Given the root of a binary tree, return the inorder traversal of its nodes' values.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 100].
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [1,null,2,3]
        Output: [1,3,2]
    
    Example 2:
        Input: root = []
        Output: []
    
    Example 3:
        Input: root = [1]
        Output: [1]
    
    FOLLOW-UP: Recursive solution is trivial, could you do it iteratively?
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h) where h is height
    
    YOUR SOLUTION:
    """
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result


def max_depth_practice(root):
    """
    PROBLEM: Maximum Depth of Binary Tree
    
    DESCRIPTION:
    Given the root of a binary tree, return its maximum depth.
    
    A binary tree's maximum depth is the number of nodes along the longest path 
    from the root node down to the farthest leaf node.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4].
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [3,9,20,null,null,15,7]
        Output: 3
    
    Example 2:
        Input: root = [1,null,2]
        Output: 2
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h)
    
    YOUR SOLUTION:
    """
    if not root:
        return 0
    
    left_depth = max_depth_practice(root.left)
    right_depth = max_depth_practice(root.right)
    
    return max(left_depth, right_depth) + 1


def is_same_tree_practice(p, q):
    """
    PROBLEM: Same Tree
    
    DESCRIPTION:
    Given the roots of two binary trees p and q, write a function to check if they are the same or not.
    
    Two binary trees are considered the same if they are structurally identical, 
    and the nodes have the same value.
    
    CONSTRAINTS:
    - The number of nodes in both trees is in the range [0, 100].
    - -10^4 <= Node.val <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: p = [1,2,3], q = [1,2,3]
        Output: true
    
    Example 2:
        Input: p = [1,2], q = [1,null,2]
        Output: false
    
    Example 3:
        Input: p = [1,2,1], q = [1,1,2]
        Output: false
    
    EXPECTED TIME COMPLEXITY: O(min(m,n))
    EXPECTED SPACE COMPLEXITY: O(min(m,n))
    
    YOUR SOLUTION:
    """
    if not p and not q:
        return True
    
    if not p or not q:
        return False
    
    if p.val != q.val:
        return False
    
    return (is_same_tree_practice(p.left, q.left) and 
            is_same_tree_practice(p.right, q.right))


def has_path_sum_practice(root, targetSum):
    """
    PROBLEM: Path Sum
    
    DESCRIPTION:
    Given the root of a binary tree and an integer targetSum, return true if the tree 
    has a root-to-leaf path such that adding up all the values along the path equals targetSum.
    
    A leaf is a node with no children.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 5000].
    - -1000 <= Node.val <= 1000
    - -1000 <= targetSum <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
        Output: true
        Explanation: The root-to-leaf path with the target sum is shown.
    
    Example 2:
        Input: root = [1,2,3], targetSum = 5
        Output: false
        Explanation: There two root-to-leaf paths in the tree:
        (1 --> 2): The sum is 3.
        (1 --> 3): The sum is 4.
        There is no root-to-leaf path with sum = 5.
    
    Example 3:
        Input: root = [], targetSum = 0
        Output: false
        Explanation: Since the tree is empty, there are no root-to-leaf paths.
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h)
    
    YOUR SOLUTION:
    """
    if not root:
        return False
    
    # If it's a leaf node, check if the remaining sum equals the node value
    if not root.left and not root.right:
        return targetSum == root.val
    
    # Recursively check left and right subtrees with updated target sum
    return (has_path_sum_practice(root.left, targetSum - root.val) or
            has_path_sum_practice(root.right, targetSum - root.val))


# BINARY SEARCH TREES PRACTICE

def is_valid_bst_practice(root):
    """
    PROBLEM: Validate Binary Search Tree
    
    DESCRIPTION:
    Given the root of a binary tree, determine if it is a valid binary search tree (BST).
    
    A valid BST is defined as follows:
    - The left subtree of a node contains only nodes with keys less than the node's key.
    - The right subtree of a node contains only nodes with keys greater than the node's key.
    - Both the left and right subtrees must also be binary search trees.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 10^4].
    - -2^31 <= Node.val <= 2^31 - 1
    
    EXAMPLES:
    Example 1:
        Input: root = [2,1,3]
        Output: true
    
    Example 2:
        Input: root = [5,1,4,null,null,3,6]
        Output: false
        Explanation: The root node's value is 5 but its right child's value is 4.
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h)
    
    YOUR SOLUTION:
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))


def search_bst_practice(root, val):
    """
    PROBLEM: Search in a Binary Search Tree
    
    DESCRIPTION:
    You are given the root of a binary search tree (BST) and an integer val.
    
    Find the node in the BST that the node's value equals val and return the subtree 
    rooted with that node. If such a node does not exist, return null.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 5000].
    - 1 <= Node.val <= 10^7
    - root is a binary search tree.
    - 1 <= val <= 10^7
    
    EXAMPLES:
    Example 1:
        Input: root = [4,2,7,1,3], val = 2
        Output: [2,1,3]
    
    Example 2:
        Input: root = [4,2,7,1,3], val = 5
        Output: []
    
    EXPECTED TIME COMPLEXITY: O(h) where h is height
    EXPECTED SPACE COMPLEXITY: O(1) iterative, O(h) recursive
    
    YOUR SOLUTION:
    """
    if not root:
        return None
    
    if root.val == val:
        return root
    elif val < root.val:
        return search_bst_practice(root.left, val)
    else:
        return search_bst_practice(root.right, val)


def insert_into_bst_practice(root, val):
    """
    PROBLEM: Insert into a Binary Search Tree
    
    DESCRIPTION:
    You are given the root node of a binary search tree (BST) and a value to insert into the tree. 
    Return the root node of the BST after the insertion. It is guaranteed that the new value does 
    not exist in the original BST.
    
    Notice that there may exist multiple valid ways for the insertion, as long as the tree 
    remains a BST after insertion. You can return any of them.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4].
    - -10^8 <= Node.val <= 10^8
    - All the values Node.val are unique.
    - -10^8 <= val <= 10^8
    - It's guaranteed that val does not exist in the original BST.
    
    EXAMPLES:
    Example 1:
        Input: root = [4,2,7,1,3], val = 5
        Output: [4,2,7,1,3,5]
    
    Example 2:
        Input: root = [40,20,60,10,30,50,70], val = 25
        Output: [40,20,60,10,30,50,70,null,null,25]
    
    Example 3:
        Input: root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
        Output: [4,2,7,1,3,5]
    
    EXPECTED TIME COMPLEXITY: O(h)
    EXPECTED SPACE COMPLEXITY: O(h)
    
    YOUR SOLUTION:
    """
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = insert_into_bst_practice(root.left, val)
    else:
        root.right = insert_into_bst_practice(root.right, val)
    
    return root


# BINARY SEARCH ON ARRAYS PRACTICE

def binary_search_practice(nums, target):
    """
    PROBLEM: Binary Search
    
    DESCRIPTION:
    Given a sorted (in ascending order) integer array nums of n elements and a target value, 
    write a function to search target in nums. If target exists, then return its index, 
    otherwise return -1.
    
    CONSTRAINTS:
    - You may assume that all elements in nums are unique.
    - n will be in the range [1, 10000].
    - The value of each element in nums will be in the range [-9999, 9999].
    
    EXAMPLES:
    Example 1:
        Input: nums = [-1,0,3,5,9,12], target = 9
        Output: 4
        Explanation: 9 exists in nums and its index is 4
    
    Example 2:
        Input: nums = [-1,0,3,5,9,12], target = 2
        Output: -1
        Explanation: 2 does not exist in nums so return -1
    
    EXPECTED TIME COMPLEXITY: O(log n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def search_range_practice(nums, target):
    """
    PROBLEM: Find First and Last Position of Element in Sorted Array
    
    DESCRIPTION:
    Given an array of integers nums sorted in non-decreasing order, find the starting 
    and ending position of a given target value.
    
    If target is not found in the array, return [-1, -1].
    
    You must write an algorithm with O(log n) runtime complexity.
    
    CONSTRAINTS:
    - 0 <= nums.length <= 10^5
    - -10^9 <= nums[i] <= 10^9
    - nums is a non-decreasing array.
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
    
    EXPECTED TIME COMPLEXITY: O(log n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
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


def search_rotated_array_practice(nums, target):
    """
    PROBLEM: Search in Rotated Sorted Array
    
    DESCRIPTION:
    There is an integer array nums sorted in ascending order (with distinct values).
    
    Prior to being passed to your function, nums is possibly rotated at an unknown 
    pivot index k (1 <= k < nums.length) such that the resulting array is 
    [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
    For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
    
    Given the array nums after the possible rotation and an integer target, return the 
    index of target if it is in nums, or -1 if it is not in nums.
    
    You must write an algorithm with O(log n) runtime complexity.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 5000
    - -10^4 <= nums[i] <= 10^4
    - All values of nums are unique.
    - nums is an ascending array that is possibly rotated.
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
    
    EXPECTED TIME COMPLEXITY: O(log n)
    EXPECTED SPACE COMPLEXITY: O(1)
    
    YOUR SOLUTION:
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


# HEAP / PRIORITY QUEUE PRACTICE

def find_kth_largest_practice(nums, k):
    """
    PROBLEM: Kth Largest Element in an Array
    
    DESCRIPTION:
    Given an integer array nums and an integer k, return the kth largest element in the array.
    
    Note that it is the kth largest element in the sorted order, not the kth distinct element.
    
    Can you solve it without sorting?
    
    CONSTRAINTS:
    - 1 <= k <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: nums = [3,2,1,5,6,4], k = 2
        Output: 5
    
    Example 2:
        Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
        Output: 4
    
    EXPECTED TIME COMPLEXITY: O(n log k) using heap, O(n) average using quickselect
    EXPECTED SPACE COMPLEXITY: O(k) using heap
    
    YOUR SOLUTION:
    """
    import heapq
    
    # Use a min heap of size k
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]


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
    
    EXPECTED TIME COMPLEXITY: O(n log k)
    EXPECTED SPACE COMPLEXITY: O(n)
    
    YOUR SOLUTION:
    """
    from collections import Counter
    
    # Count frequencies
    counter = Counter(nums)
    
    # Use heap to find top k elements
    return heapq.nlargest(k, counter.keys(), key=counter.get)


class MedianFinder_Practice:
    """
    PROBLEM: Find Median from Data Stream
    
    DESCRIPTION:
    The median is the middle value in an ordered integer list. If the size of the list 
    is even, there is no middle value and the median is the mean of the two middle values.
    
    Implement the MedianFinder class:
    - MedianFinder() initializes the MedianFinder object.
    - void addNum(int num) adds the integer num from the data stream to the data structure.
    - double findMedian() returns the median of all elements so far.
    
    CONSTRAINTS:
    - -10^5 <= num <= 10^5
    - There will be at least one element in the data structure before calling findMedian.
    - At most 5 * 10^4 calls will be made to addNum and findMedian.
    
    EXAMPLES:
    Example 1:
        Input: 
        ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
        [[], [1], [2], [], [3], []]
        Output:
        [null, null, null, 1.5, null, 2.0]
    
    EXPECTED TIME COMPLEXITY: O(log n) for addNum, O(1) for findMedian
    EXPECTED SPACE COMPLEXITY: O(n)
    """

    def __init__(self):
        # Use two heaps: max heap for smaller half, min heap for larger half
        self.small = []  # max heap (use negative values)
        self.large = []  # min heap

    def addNum(self, num: int) -> None:
        # Always add to small first
        heapq.heappush(self.small, -num)
        
        # Ensure all elements in small are <= all elements in large
        if self.small and self.large and -self.small[0] > self.large[0]:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        
        # Balance the heaps
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        else:
            return (-self.small[0] + self.large[0]) / 2.0


# HELPER FUNCTIONS

def create_tree_from_list(values):
    """Helper function to create binary tree from level-order list"""
    if not values or values[0] is None:
        return None
    
    root = TreeNode(values[0])
    queue = deque([root])
    index = 1
    
    while queue and index < len(values):
        node = queue.popleft()
        
        # Add left child
        if index < len(values) and values[index] is not None:
            node.left = TreeNode(values[index])
            queue.append(node.left)
        index += 1
        
        # Add right child
        if index < len(values) and values[index] is not None:
            node.right = TreeNode(values[index])
            queue.append(node.right)
        index += 1
    
    return root


def tree_to_list(root):
    """Helper function to convert tree to level-order list"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    
    # Remove trailing None values
    while result and result[-1] is None:
        result.pop()
    
    return result


# TEST CASES FOR VERIFICATION

def test_week2_practice():
    """Test your solutions with comprehensive test cases"""
    
    print("=== TESTING WEEK 2 PRACTICE SOLUTIONS ===\n")
    
    # Test Inorder Traversal
    print("1. Binary Tree Inorder Traversal:")
    test_cases = [
        ([1, None, 2, 3], [1, 3, 2]),
        ([], []),
        ([1], [1])
    ]
    for tree_list, expected in test_cases:
        root = create_tree_from_list(tree_list)
        result = inorder_traversal_practice(root)
        print(f"   Input: {tree_list}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Maximum Depth
    print("2. Maximum Depth of Binary Tree:")
    test_cases = [
        ([3, 9, 20, None, None, 15, 7], 3),
        ([1, None, 2], 2),
        ([], 0)
    ]
    for tree_list, expected in test_cases:
        root = create_tree_from_list(tree_list)
        result = max_depth_practice(root)
        print(f"   Input: {tree_list}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test BST Validation
    print("3. Validate Binary Search Tree:")
    test_cases = [
        ([2, 1, 3], True),
        ([5, 1, 4, None, None, 3, 6], False)
    ]
    for tree_list, expected in test_cases:
        root = create_tree_from_list(tree_list)
        result = is_valid_bst_practice(root)
        print(f"   Input: {tree_list}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Binary Search
    print("4. Binary Search:")
    test_cases = [
        ([-1, 0, 3, 5, 9, 12], 9, 4),
        ([-1, 0, 3, 5, 9, 12], 2, -1)
    ]
    for nums, target, expected in test_cases:
        result = binary_search_practice(nums, target)
        print(f"   Input: nums={nums}, target={target}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    # Test Kth Largest
    print("5. Kth Largest Element:")
    test_cases = [
        ([3, 2, 1, 5, 6, 4], 2, 5),
        ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 4)
    ]
    for nums, k, expected in test_cases:
        result = find_kth_largest_practice(nums, k)
        print(f"   Input: nums={nums}, k={k}")
        print(f"   Output: {result} (Expected: {expected})")
        print(f"   ✓ Correct: {result == expected}")
        print()
    
    print("Continue implementing and testing other problems...")


# REFERENCE SOLUTIONS (Uncomment when you want to check your work)

def inorder_traversal_reference(root):
    """Reference solution for Inorder Traversal"""
    result = []
    
    def inorder(node):
        if not node:
            return
        inorder(node.left)
        result.append(node.val)
        inorder(node.right)
    
    inorder(root)
    return result


def max_depth_reference(root):
    """Reference solution for Maximum Depth"""
    if not root:
        return 0
    
    left_depth = max_depth_reference(root.left)
    right_depth = max_depth_reference(root.right)
    
    return 1 + max(left_depth, right_depth)


def is_valid_bst_reference(root):
    """Reference solution for Validate BST"""
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))


def binary_search_reference(nums, target):
    """Reference solution for Binary Search"""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def find_kth_largest_reference(nums, k):
    """Reference solution for Kth Largest Element"""
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]


# Add more reference solutions...


if __name__ == "__main__":
    print("Week 2 Practice Problems")
    print("========================")
    print("Read each problem statement carefully.")
    print("Understand constraints and examples.")
    print("Write your solution in the designated space.")
    print("Run test_week2_practice() to check your solutions.")
    print()
    
    # Uncomment to run tests
    # test_week2_practice() 