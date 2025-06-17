"""
=============================================================================
                        WEEK 2 SOLUTION FILE
                     COMPLETE SOLUTIONS & VARIANTS
                           Meta Interview Preparation
=============================================================================

This file contains complete solutions for all Week 2 practice problems with
multiple approaches, variants, and comprehensive test cases.

TOPICS COVERED:
- Binary Trees
- Binary Search Trees
- Binary Search on Arrays
- Heaps & Priority Queues

=============================================================================
"""

from collections import deque, defaultdict
from typing import List, Optional, Tuple
import heapq


# TreeNode definition
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# =============================================================================
# BINARY TREES SOLUTIONS
# =============================================================================

def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    PROBLEM: Binary Tree Inorder Traversal (Recursive)
    TIME: O(n), SPACE: O(h)
    """
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result

def inorder_traversal_iterative(root: Optional[TreeNode]) -> List[int]:
    """
    VARIANT: Inorder Traversal (Iterative)
    TIME: O(n), SPACE: O(h)
    """
    result = []
    stack = []
    current = root
    
    while stack or current:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
    return result

def preorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    VARIANT: Preorder Traversal (Root -> Left -> Right)
    TIME: O(n), SPACE: O(h)
    """
    result = []
    
    def preorder(node):
        if node:
            result.append(node.val)
            preorder(node.left)
            preorder(node.right)
    
    preorder(root)
    return result

def postorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    VARIANT: Postorder Traversal (Left -> Right -> Root)
    TIME: O(n), SPACE: O(h)
    """
    result = []
    
    def postorder(node):
        if node:
            postorder(node.left)
            postorder(node.right)
            result.append(node.val)
    
    postorder(root)
    return result

def level_order_traversal(root: Optional[TreeNode]) -> List[List[int]]:
    """
    VARIANT: Level Order Traversal (BFS)
    TIME: O(n), SPACE: O(w) where w is max width
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result

def max_depth(root: Optional[TreeNode]) -> int:
    """
    PROBLEM: Maximum Depth of Binary Tree
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return max(left_depth, right_depth) + 1

def min_depth(root: Optional[TreeNode]) -> int:
    """
    VARIANT: Minimum Depth of Binary Tree
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return 0
    
    if not root.left and not root.right:
        return 1
    
    if not root.left:
        return min_depth(root.right) + 1
    
    if not root.right:
        return min_depth(root.left) + 1
    
    return min(min_depth(root.left), min_depth(root.right)) + 1

def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    PROBLEM: Same Tree
    TIME: O(min(m,n)), SPACE: O(min(m,n))
    """
    if not p and not q:
        return True
    
    if not p or not q:
        return False
    
    if p.val != q.val:
        return False
    
    return (is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))

def is_symmetric(root: Optional[TreeNode]) -> bool:
    """
    VARIANT: Symmetric Tree
    TIME: O(n), SPACE: O(h)
    """
    def is_mirror(left, right):
        if not left and not right:
            return True
        
        if not left or not right:
            return False
        
        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))
    
    if not root:
        return True
    
    return is_mirror(root.left, root.right)

def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
    """
    PROBLEM: Path Sum
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return False
    
    if not root.left and not root.right:
        return target_sum == root.val
    
    return (has_path_sum(root.left, target_sum - root.val) or
            has_path_sum(root.right, target_sum - root.val))

def path_sum_all_paths(root: Optional[TreeNode], target_sum: int) -> List[List[int]]:
    """
    VARIANT: Path Sum II - Return all paths
    TIME: O(n²), SPACE: O(n²)
    """
    result = []
    
    def dfs(node, remaining, path):
        if not node:
            return
        
        path.append(node.val)
        
        if not node.left and not node.right and remaining == node.val:
            result.append(path.copy())
        else:
            dfs(node.left, remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)
        
        path.pop()
    
    dfs(root, target_sum, [])
    return result

def diameter_of_tree(root: Optional[TreeNode]) -> int:
    """
    VARIANT: Diameter of Binary Tree
    TIME: O(n), SPACE: O(h)
    """
    max_diameter = 0
    
    def height(node):
        nonlocal max_diameter
        
        if not node:
            return 0
        
        left_height = height(node.left)
        right_height = height(node.right)
        
        max_diameter = max(max_diameter, left_height + right_height)
        
        return max(left_height, right_height) + 1
    
    height(root)
    return max_diameter

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    VARIANT: Lowest Common Ancestor of Binary Tree
    TIME: O(n), SPACE: O(h)
    """
    if not root or root == p or root == q:
        return root
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root
    
    return left or right

# =============================================================================
# BINARY SEARCH TREES SOLUTIONS
# =============================================================================

def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    PROBLEM: Validate Binary Search Tree
    TIME: O(n), SPACE: O(h)
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

def is_valid_bst_inorder(root: Optional[TreeNode]) -> bool:
    """
    VARIANT: Validate BST using inorder traversal
    TIME: O(n), SPACE: O(h)
    """
    inorder_vals = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            inorder_vals.append(node.val)
            inorder(node.right)
    
    inorder(root)
    
    for i in range(1, len(inorder_vals)):
        if inorder_vals[i] <= inorder_vals[i-1]:
            return False
    
    return True

def search_bst(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    """
    PROBLEM: Search in a Binary Search Tree
    TIME: O(h), SPACE: O(1) iterative, O(h) recursive
    """
    if not root:
        return None
    
    if root.val == val:
        return root
    elif val < root.val:
        return search_bst(root.left, val)
    else:
        return search_bst(root.right, val)

def search_bst_iterative(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    """
    VARIANT: Search BST (Iterative)
    TIME: O(h), SPACE: O(1)
    """
    while root:
        if root.val == val:
            return root
        elif val < root.val:
            root = root.left
        else:
            root = root.right
    
    return None

def insert_into_bst(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    """
    PROBLEM: Insert into a Binary Search Tree
    TIME: O(h), SPACE: O(h)
    """
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    else:
        root.right = insert_into_bst(root.right, val)
    
    return root

def delete_from_bst(root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
    """
    VARIANT: Delete Node in a BST
    TIME: O(h), SPACE: O(h)
    """
    if not root:
        return None
    
    if key < root.val:
        root.left = delete_from_bst(root.left, key)
    elif key > root.val:
        root.right = delete_from_bst(root.right, key)
    else:
        # Node to delete found
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        else:
            # Node has two children
            # Find inorder successor (smallest in right subtree)
            min_node = root.right
            while min_node.left:
                min_node = min_node.left
            
            root.val = min_node.val
            root.right = delete_from_bst(root.right, min_node.val)
    
    return root

def kth_smallest_in_bst(root: Optional[TreeNode], k: int) -> int:
    """
    VARIANT: Kth Smallest Element in BST
    TIME: O(h + k), SPACE: O(h)
    """
    def inorder(node):
        if not node:
            return None
        
        left_result = inorder(node.left)
        if left_result is not None:
            return left_result
        
        nonlocal k
        k -= 1
        if k == 0:
            return node.val
        
        return inorder(node.right)
    
    return inorder(root)

def bst_to_sorted_list(root: Optional[TreeNode]) -> List[int]:
    """
    VARIANT: Convert BST to Sorted Array
    TIME: O(n), SPACE: O(n)
    """
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result

def lca_bst(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    VARIANT: Lowest Common Ancestor of BST
    TIME: O(h), SPACE: O(1)
    """
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
    
    return None

# =============================================================================
# BINARY SEARCH ON ARRAYS SOLUTIONS
# =============================================================================

def binary_search(nums: List[int], target: int) -> int:
    """
    PROBLEM: Binary Search
    TIME: O(log n), SPACE: O(1)
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

def binary_search_recursive(nums: List[int], target: int) -> int:
    """
    VARIANT: Binary Search (Recursive)
    TIME: O(log n), SPACE: O(log n)
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

def search_range(nums: List[int], target: int) -> List[int]:
    """
    PROBLEM: Find First and Last Position of Element in Sorted Array
    TIME: O(log n), SPACE: O(1)
    """
    def find_first(nums, target):
        left, right = 0, len(nums) - 1
        first_pos = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                first_pos = mid
                right = mid - 1
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
                left = mid + 1
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

def search_insert_position(nums: List[int], target: int) -> int:
    """
    VARIANT: Search Insert Position
    TIME: O(log n), SPACE: O(1)
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
    
    return left

def search_rotated_array(nums: List[int], target: int) -> int:
    """
    PROBLEM: Search in Rotated Sorted Array
    TIME: O(log n), SPACE: O(1)
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

def find_minimum_rotated_array(nums: List[int]) -> int:
    """
    VARIANT: Find Minimum in Rotated Sorted Array
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]

def find_peak_element(nums: List[int]) -> int:
    """
    VARIANT: Find Peak Element
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left

def sqrt_integer(x: int) -> int:
    """
    VARIANT: Sqrt(x) using Binary Search
    TIME: O(log x), SPACE: O(1)
    """
    if x < 2:
        return x
    
    left, right = 2, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right

# =============================================================================
# HEAP & PRIORITY QUEUE SOLUTIONS
# =============================================================================

def find_kth_largest(nums: List[int], k: int) -> int:
    """
    PROBLEM: Kth Largest Element in an Array
    TIME: O(n log k), SPACE: O(k)
    """
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]

def find_kth_largest_quickselect(nums: List[int], k: int) -> int:
    """
    VARIANT: Kth Largest using Quickselect
    TIME: O(n) average, O(n²) worst, SPACE: O(1)
    """
    def partition(left, right, pivot_idx):
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if nums[i] < nums[right]:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1
        
        nums[store_idx], nums[right] = nums[right], nums[store_idx]
        return store_idx
    
    def quickselect(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        import random
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(left, right, pivot_idx)
        
        if k_smallest == pivot_idx:
            return nums[k_smallest]
        elif k_smallest < pivot_idx:
            return quickselect(left, pivot_idx - 1, k_smallest)
        else:
            return quickselect(pivot_idx + 1, right, k_smallest)
    
    return quickselect(0, len(nums) - 1, len(nums) - k)

def top_k_frequent_heap(nums: List[int], k: int) -> List[int]:
    """
    PROBLEM: Top K Frequent Elements
    TIME: O(n log k), SPACE: O(n)
    """
    from collections import Counter
    
    counter = Counter(nums)
    return heapq.nlargest(k, counter.keys(), key=counter.get)

def top_k_frequent_bucket_sort(nums: List[int], k: int) -> List[int]:
    """
    VARIANT: Top K Frequent using Bucket Sort
    TIME: O(n), SPACE: O(n)
    """
    from collections import Counter
    
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

class MedianFinder:
    """
    PROBLEM: Find Median from Data Stream
    TIME: O(log n) for addNum, O(1) for findMedian
    SPACE: O(n)
    """

    def __init__(self):
        self.small = []  # max heap (use negative values)
        self.large = []  # min heap

    def addNum(self, num: int) -> None:
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

def merge_k_sorted_lists_heap(lists: List[List[int]]) -> List[int]:
    """
    VARIANT: Merge K Sorted Lists using Heap
    TIME: O(N log k), SPACE: O(k)
    """
    heap = []
    result = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    while heap:
        val, list_idx, element_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same list
        if element_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][element_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, element_idx + 1))
    
    return result

def sliding_window_median(nums: List[int], k: int) -> List[float]:
    """
    VARIANT: Sliding Window Median
    TIME: O(n log k), SPACE: O(k)
    """
    from bisect import bisect_left, insort
    
    window = []
    result = []
    
    for i, num in enumerate(nums):
        # Add element to window
        insort(window, num)
        
        # Remove element outside window
        if len(window) > k:
            window.pop(bisect_left(window, nums[i - k]))
        
        # Calculate median
        if len(window) == k:
            if k % 2 == 1:
                result.append(float(window[k // 2]))
            else:
                result.append((window[k // 2 - 1] + window[k // 2]) / 2.0)
    
    return result

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_tree_from_list(values: List[Optional[int]]) -> Optional[TreeNode]:
    """Create binary tree from level-order list"""
    if not values or values[0] is None:
        return None
    
    root = TreeNode(values[0])
    queue = deque([root])
    index = 1
    
    while queue and index < len(values):
        node = queue.popleft()
        
        if index < len(values) and values[index] is not None:
            node.left = TreeNode(values[index])
            queue.append(node.left)
        index += 1
        
        if index < len(values) and values[index] is not None:
            node.right = TreeNode(values[index])
            queue.append(node.right)
        index += 1
    
    return root

def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    """Convert tree to level-order list"""
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

def print_tree(root: Optional[TreeNode], level: int = 0, prefix: str = "Root: ") -> None:
    """Print tree structure"""
    if root is not None:
        print("  " * level + prefix + str(root.val))
        if root.left is not None or root.right is not None:
            if root.left:
                print_tree(root.left, level + 1, "L--- ")
            else:
                print("  " * (level + 1) + "L--- None")
            if root.right:
                print_tree(root.right, level + 1, "R--- ")
            else:
                print("  " * (level + 1) + "R--- None")

# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def test_all_week2_solutions():
    """Comprehensive test suite for all Week 2 solutions"""
    
    print("=" * 80)
    print("                    WEEK 2 COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Test Binary Tree Traversals
    print("\n🧪 TESTING BINARY TREE TRAVERSALS")
    print("-" * 50)
    
    # Create test tree: [1,null,2,3]
    tree1 = create_tree_from_list([1, None, 2, 3])
    
    inorder_result = inorder_traversal(tree1)
    print(f"Inorder [1,null,2,3]: {inorder_result} (Expected: [1,3,2])")
    
    inorder_iter_result = inorder_traversal_iterative(tree1)
    print(f"Inorder Iterative: {inorder_iter_result} (Expected: [1,3,2])")
    
    preorder_result = preorder_traversal(tree1)
    print(f"Preorder: {preorder_result} (Expected: [1,2,3])")
    
    postorder_result = postorder_traversal(tree1)
    print(f"Postorder: {postorder_result} (Expected: [3,2,1])")
    
    # Test complete binary tree
    tree2 = create_tree_from_list([1, 2, 3, 4, 5, 6, 7])
    level_order_result = level_order_traversal(tree2)
    print(f"Level Order [1,2,3,4,5,6,7]: {level_order_result}")
    print(f"Expected: [[1], [2, 3], [4, 5, 6, 7]]")
    
    # Test Tree Properties
    print("\n🧪 TESTING TREE PROPERTIES")
    print("-" * 50)
    
    test_cases_depth = [
        ([3, 9, 20, None, None, 15, 7], 3),
        ([1, None, 2], 2),
        ([], 0),
        ([1], 1)
    ]
    
    for i, (tree_list, expected) in enumerate(test_cases_depth, 1):
        tree = create_tree_from_list(tree_list)
        result = max_depth(tree)
        print(f"Max Depth Test {i}: {tree_list}")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  ✅ PASS" if result == expected else f"  ❌ FAIL")
    
    # Test Same Tree
    tree_a = create_tree_from_list([1, 2, 3])
    tree_b = create_tree_from_list([1, 2, 3])
    tree_c = create_tree_from_list([1, None, 2])
    
    same_result1 = is_same_tree(tree_a, tree_b)
    same_result2 = is_same_tree(tree_a, tree_c)
    print(f"Same Tree [1,2,3] == [1,2,3]: {same_result1} (Expected: True)")
    print(f"Same Tree [1,2,3] == [1,null,2]: {same_result2} (Expected: False)")
    
    # Test Symmetric Tree
    symmetric_tree = create_tree_from_list([1, 2, 2, 3, 4, 4, 3])
    asymmetric_tree = create_tree_from_list([1, 2, 2, None, 3, None, 3])
    
    symmetric_result1 = is_symmetric(symmetric_tree)
    symmetric_result2 = is_symmetric(asymmetric_tree)
    print(f"Symmetric [1,2,2,3,4,4,3]: {symmetric_result1} (Expected: True)")
    print(f"Symmetric [1,2,2,null,3,null,3]: {symmetric_result2} (Expected: False)")
    
    # Test Path Sum
    path_sum_tree = create_tree_from_list([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])
    path_sum_result = has_path_sum(path_sum_tree, 22)
    print(f"Has Path Sum 22: {path_sum_result} (Expected: True)")
    
    # Test BST Operations
    print("\n🧪 TESTING BST OPERATIONS")
    print("-" * 50)
    
    # Valid BST
    valid_bst = create_tree_from_list([2, 1, 3])
    invalid_bst = create_tree_from_list([5, 1, 4, None, None, 3, 6])
    
    valid_result1 = is_valid_bst(valid_bst)
    valid_result2 = is_valid_bst(invalid_bst)
    print(f"Valid BST [2,1,3]: {valid_result1} (Expected: True)")
    print(f"Valid BST [5,1,4,null,null,3,6]: {valid_result2} (Expected: False)")
    
    # Search BST
    search_tree = create_tree_from_list([4, 2, 7, 1, 3])
    search_result = search_bst(search_tree, 2)
    search_iterative_result = search_bst_iterative(search_tree, 2)
    print(f"Search BST for 2: Found node with value {search_result.val if search_result else None}")
    print(f"Search BST Iterative for 2: Found node with value {search_iterative_result.val if search_iterative_result else None}")
    
    # Insert into BST
    insert_tree = create_tree_from_list([4, 2, 7, 1, 3])
    insert_result = insert_into_bst(insert_tree, 5)
    print(f"Insert 5 into BST: Tree modified successfully")
    
    # Kth Smallest in BST
    kth_tree = create_tree_from_list([3, 1, 4, None, 2])
    kth_result = kth_smallest_in_bst(kth_tree, 1)
    print(f"1st smallest in BST: {kth_result} (Expected: 1)")
    
    # Test Binary Search on Arrays
    print("\n🧪 TESTING BINARY SEARCH ON ARRAYS")
    print("-" * 50)
    
    test_cases_binary_search = [
        ([-1, 0, 3, 5, 9, 12], 9, 4),
        ([-1, 0, 3, 5, 9, 12], 2, -1),
        ([5], 5, 0),
        ([1, 3], 3, 1)
    ]
    
    for i, (nums, target, expected) in enumerate(test_cases_binary_search, 1):
        result = binary_search(nums, target)
        recursive_result = binary_search_recursive(nums, target)
        print(f"Binary Search Test {i}: nums={nums}, target={target}")
        print(f"  Iterative - Expected: {expected}, Got: {result}")
        print(f"  Recursive - Expected: {expected}, Got: {recursive_result}")
        print(f"  ✅ PASS" if result == expected and recursive_result == expected else f"  ❌ FAIL")
    
    # Test Search Range
    range_result = search_range([5, 7, 7, 8, 8, 10], 8)
    print(f"Search Range for 8 in [5,7,7,8,8,10]: {range_result} (Expected: [3,4])")
    
    # Test Search Insert Position
    insert_pos_result = search_insert_position([1, 3, 5, 6], 5)
    print(f"Search Insert Position for 5: {insert_pos_result} (Expected: 2)")
    
    # Test Search in Rotated Array
    rotated_result = search_rotated_array([4, 5, 6, 7, 0, 1, 2], 0)
    print(f"Search in Rotated Array for 0: {rotated_result} (Expected: 4)")
    
    # Test Find Minimum in Rotated Array
    min_rotated_result = find_minimum_rotated_array([3, 4, 5, 1, 2])
    print(f"Find Minimum in Rotated Array: {min_rotated_result} (Expected: 1)")
    
    # Test Heap Operations
    print("\n🧪 TESTING HEAP OPERATIONS")
    print("-" * 50)
    
    # Kth Largest Element
    kth_largest_result = find_kth_largest([3, 2, 1, 5, 6, 4], 2)
    kth_largest_quickselect_result = find_kth_largest_quickselect([3, 2, 1, 5, 6, 4], 2)
    print(f"Kth Largest (heap): {kth_largest_result} (Expected: 5)")
    print(f"Kth Largest (quickselect): {kth_largest_quickselect_result} (Expected: 5)")
    
    # Top K Frequent
    top_k_heap_result = top_k_frequent_heap([1, 1, 1, 2, 2, 3], 2)
    top_k_bucket_result = top_k_frequent_bucket_sort([1, 1, 1, 2, 2, 3], 2)
    print(f"Top K Frequent (heap): {top_k_heap_result}")
    print(f"Top K Frequent (bucket): {top_k_bucket_result}")
    
    # Test Median Finder
    print("\nTesting Median Finder:")
    median_finder = MedianFinder()
    median_finder.addNum(1)
    median_finder.addNum(2)
    print(f"Median after adding 1,2: {median_finder.findMedian()} (Expected: 1.5)")
    median_finder.addNum(3)
    print(f"Median after adding 3: {median_finder.findMedian()} (Expected: 2.0)")
    
    # Test Complex Scenarios
    print("\n🧪 TESTING COMPLEX SCENARIOS")
    print("-" * 50)
    
    # Diameter of Tree
    diameter_tree = create_tree_from_list([1, 2, 3, 4, 5])
    diameter_result = diameter_of_tree(diameter_tree)
    print(f"Diameter of Tree: {diameter_result} (Expected: 3)")
    
    # Path Sum II (All Paths)
    path_sum_tree = create_tree_from_list([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1])
    all_paths_result = path_sum_all_paths(path_sum_tree, 22)
    print(f"All Path Sum 22: {len(all_paths_result)} paths found")
    
    # BST to Sorted List
    bst_tree = create_tree_from_list([2, 1, 3])
    sorted_list_result = bst_to_sorted_list(bst_tree)
    print(f"BST to Sorted List: {sorted_list_result} (Expected: [1,2,3])")
    
    print("\n" + "=" * 80)
    print("                    TESTING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_all_week2_solutions() 