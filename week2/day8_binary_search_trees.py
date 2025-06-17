"""
=============================================================================
                   WEEK 2 - DAY 8: BINARY SEARCH TREES (BST)
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. BINARY SEARCH TREE PROPERTIES
   - BST Property: For any node N:
     * All nodes in left subtree < N.val
     * All nodes in right subtree > N.val
     * Both left and right subtrees are also BSTs
   - In-order traversal gives sorted sequence
   - Search, insert, delete in O(h) time where h is height

2. BST VS GENERAL BINARY TREE
   - General tree: No ordering constraint
   - BST: Strict ordering enables efficient operations
   - Balanced BST: O(log n) operations (AVL, Red-Black)
   - Unbalanced BST: O(n) worst case (degenerates to linked list)

3. COMMON BST OPERATIONS
   - Search: Navigate left/right based on comparison
   - Insert: Find position, add as leaf
   - Delete: Three cases (leaf, one child, two children)
   - Minimum/Maximum: Leftmost/rightmost node
   - Successor/Predecessor: Next/previous in sorted order

4. BST VALIDATION
   - Naive approach: Check each node's immediate children
   - Correct approach: Maintain valid range for each node
   - In-order traversal should produce sorted sequence

5. BALANCED BST CONCEPTS
   - Height-balanced: Height difference ≤ 1 between subtrees
   - Self-balancing trees: Automatically maintain balance
   - Rotation operations for rebalancing
   - Examples: AVL trees, Red-Black trees

6. TRANSITION FROM DAY 7
   - Tree traversals (especially in-order) crucial for BST
   - Recursive patterns from general trees apply
   - Add ordering constraint for efficiency
   - Range-based thinking for validation

=============================================================================
"""

from typing import List, Optional


# Use TreeNode from Day 7
class TreeNode:
    """Standard binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"


# =============================================================================
# PROBLEM 1: VALIDATE BINARY SEARCH TREE (MEDIUM) - 45 MIN
# =============================================================================

def is_valid_bst_wrong(root):
    """
    EDUCATIONAL: INCORRECT approach - Only checks immediate children
    
    This fails for cases like [5, 1, 4, null, null, 3, 6]
    where 3 < 5 but is in right subtree
    
    Educational purpose: Shows common mistake
    """
    if not root:
        return True
    
    # Check immediate children only (WRONG!)
    if root.left and root.left.val >= root.val:
        return False
    if root.right and root.right.val <= root.val:
        return False
    
    return (is_valid_bst_wrong(root.left) and 
            is_valid_bst_wrong(root.right))


def is_valid_bst_correct(root):
    """
    PROBLEM: Validate Binary Search Tree
    
    Given the root of a binary tree, determine if it is a valid binary search tree (BST).
    
    A valid BST is defined as follows:
    - The left subtree of a node contains only nodes with keys less than the node's key
    - The right subtree of a node contains only nodes with keys greater than the node's key
    - Both the left and right subtrees must also be binary search trees
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 10^4]
    - -2^31 <= Node.val <= 2^31 - 1
    
    EXAMPLES:
    Example 1:
        Input: root = [2,1,3]
        Output: true
    
    Example 2:
        Input: root = [5,1,4,null,null,3,6]
        Output: false
        Explanation: The root node's value is 5 but its right child's value is 4
    
    APPROACH: Range Validation (Correct)
    
    Each node must be within (min_val, max_val) range
    
    TIME: O(n), SPACE: O(h)
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        # Check if current node violates BST property
        if node.val <= min_val or node.val >= max_val:
            return False
        
        # Recursively validate with updated ranges
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))


def is_valid_bst_inorder(root):
    """
    APPROACH: In-order Traversal Should Be Sorted
    
    TIME: O(n), SPACE: O(h)
    """
    def inorder(node):
        if not node:
            return []
        
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    values = inorder(root)
    
    # Check if sorted and no duplicates
    for i in range(1, len(values)):
        if values[i] <= values[i-1]:
            return False
    
    return True


def is_valid_bst_inorder_optimized(root):
    """
    APPROACH: Optimized In-order (Track Previous Value)
    
    Track previous value without storing all values
    
    TIME: O(n), SPACE: O(h)
    """
    def inorder(node):
        nonlocal prev
        if not node:
            return True
        
        # Traverse left
        if not inorder(node.left):
            return False
        
        # Check current node
        if prev is not None and node.val <= prev:
            return False
        prev = node.val
        
        # Traverse right
        return inorder(node.right)
    
    prev = None
    return inorder(root)


# =============================================================================
# PROBLEM 2: SEARCH IN A BINARY SEARCH TREE (EASY) - 30 MIN
# =============================================================================

def search_bst_recursive(root, val):
    """
    PROBLEM: Search in a Binary Search Tree
    
    You are given the root of a binary search tree (BST) and an integer val.
    
    Find the node in the BST that the node's value equals val and return the subtree 
    rooted with that node. If such a node does not exist, return null.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 5000]
    - 1 <= Node.val <= 10^7
    - root is a binary search tree
    - 1 <= val <= 10^7
    
    EXAMPLES:
    Example 1:
        Input: root = [4,2,7,1,3], val = 2
        Output: [2,1,3]
    
    Example 2:
        Input: root = [4,2,7,1,3], val = 5
        Output: []
    
    APPROACH: Recursive Search
    
    Navigate left/right based on comparison
    
    TIME: O(h), SPACE: O(h)
    """
    if not root or root.val == val:
        return root
    
    if val < root.val:
        return search_bst_recursive(root.left, val)
    else:
        return search_bst_recursive(root.right, val)


def search_bst_iterative(root, val):
    """
    APPROACH: Iterative Search
    
    More memory efficient than recursive
    
    TIME: O(h), SPACE: O(1)
    """
    current = root
    
    while current:
        if val == current.val:
            return current
        elif val < current.val:
            current = current.left
        else:
            current = current.right
    
    return None


# =============================================================================
# PROBLEM 3: INSERT INTO A BINARY SEARCH TREE (MEDIUM) - 45 MIN
# =============================================================================

def insert_into_bst_recursive(root, val):
    """
    PROBLEM: Insert into a Binary Search Tree
    
    You are given the root node of a binary search tree (BST) and a value to insert into the tree. 
    Return the root node of the BST after the insertion. It is guaranteed that the new value does 
    not exist in the original BST.
    
    Notice that there may exist multiple valid ways for the insertion, as long as the tree remains 
    a BST after insertion. You can return any of them.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4]
    - -10^8 <= Node.val <= 10^8
    - All the values Node.val are unique
    - -10^8 <= val <= 10^8
    - It's guaranteed that val does not exist in the original BST
    
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
    
    APPROACH: Recursive Insertion
    
    Find correct position and insert as leaf
    
    TIME: O(h), SPACE: O(h)
    """
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = insert_into_bst_recursive(root.left, val)
    else:
        root.right = insert_into_bst_recursive(root.right, val)
    
    return root


def insert_into_bst_iterative(root, val):
    """
    APPROACH: Iterative Insertion
    
    Find parent and insert as appropriate child
    
    TIME: O(h), SPACE: O(1)
    """
    new_node = TreeNode(val)
    
    if not root:
        return new_node
    
    current = root
    while True:
        if val < current.val:
            if not current.left:
                current.left = new_node
                break
            current = current.left
        else:
            if not current.right:
                current.right = new_node
                break
            current = current.right
    
    return root


# =============================================================================
# PROBLEM 4: DELETE NODE IN A BST (MEDIUM) - 45 MIN
# =============================================================================

def delete_node_bst(root, key):
    """
    PROBLEM: Delete Node in a BST
    
    Given a root node reference of a BST and a key, delete the node with the given key in the BST. 
    Return the root node reference (possibly updated) of the BST.
    
    Basically, the deletion can be divided into two stages:
    1. Search for a node to remove
    2. If the node is found, delete the node
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4]
    - -10^5 <= Node.val <= 10^5
    - Each node has a unique value
    - root is a valid binary search tree
    - -10^5 <= key <= 10^5
    
    EXAMPLES:
    Example 1:
        Input: root = [5,3,6,2,4,null,7], key = 3
        Output: [5,4,6,2,null,null,7]
        Explanation: Given key to delete is 3. So we find the node with value 3 and delete it.
    
    Example 2:
        Input: root = [5,3,6,2,4,null,7], key = 0
        Output: [5,3,6,2,4,null,7]
        Explanation: The tree does not contain a node with value = 0.
    
    Example 3:
        Input: root = [], key = 0
        Output: []
    
    APPROACH: Three Cases for Deletion
    
    1. Leaf node: simply remove
    2. One child: replace with child
    3. Two children: replace with inorder successor
    
    TIME: O(h), SPACE: O(h)
    """
    if not root:
        return None
    
    if key < root.val:
        root.left = delete_node_bst(root.left, key)
    elif key > root.val:
        root.right = delete_node_bst(root.right, key)
    else:
        # Node to delete found
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        else:
            # Node has two children: find inorder successor
            min_larger = find_min(root.right)
            root.val = min_larger.val
            root.right = delete_node_bst(root.right, min_larger.val)
    
    return root


def find_min(root):
    """Helper: Find minimum value node in BST"""
    while root.left:
        root = root.left
    return root


def find_max(root):
    """Helper: Find maximum value node in BST"""
    while root.right:
        root = root.right
    return root


# =============================================================================
# PROBLEM 5: KTH SMALLEST ELEMENT IN A BST (MEDIUM) - 45 MIN
# =============================================================================

def kth_smallest_recursive(root, k):
    """
    PROBLEM: Kth Smallest Element in a BST
    
    Given the root of a binary search tree, and an integer k, return the kth smallest 
    value (1-indexed) of all the values of the nodes in the tree.
    
    CONSTRAINTS:
    - The number of nodes in the tree is n
    - 1 <= k <= n <= 10^4
    - 0 <= Node.val <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: root = [3,1,4,null,2], k = 1
        Output: 1
    
    Example 2:
        Input: root = [5,3,6,2,4,null,null,1], k = 3
        Output: 3
    
    APPROACH 1: Inorder Traversal (Recursive)
    
    Inorder traversal of BST gives sorted order
    
    TIME: O(n), SPACE: O(n)
    """
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    sorted_vals = inorder(root)
    return sorted_vals[k - 1]


def kth_smallest_optimized(root, k):
    """
    APPROACH 2: Early Termination Inorder
    
    Stop as soon as we find the kth element
    
    TIME: O(h + k), SPACE: O(h)
    """
    def inorder(node):
        nonlocal count, result
        if not node or result is not None:
            return
        
        inorder(node.left)
        
        count += 1
        if count == k:
            result = node.val
            return
        
        inorder(node.right)
    
    count = 0
    result = None
    inorder(root)
    return result


def kth_smallest_iterative(root, k):
    """
    APPROACH 3: Iterative Inorder with Stack
    
    Most space-efficient approach
    
    TIME: O(h + k), SPACE: O(h)
    """
    stack = []
    current = root
    count = 0
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        count += 1
        
        if count == k:
            return current.val
        
        # Move to right subtree
        current = current.right
    
    return None


# =============================================================================
# PROBLEM 6: LOWEST COMMON ANCESTOR OF A BST (EASY) - 30 MIN
# =============================================================================

def lowest_common_ancestor_bst(root, p, q):
    """
    PROBLEM: Lowest Common Ancestor of a Binary Search Tree
    
    Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
    
    According to the definition of LCA on Wikipedia: "The lowest common ancestor is defined between 
    two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow 
    a node to be a descendant of itself)."
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [2, 10^5]
    - -10^9 <= Node.val <= 10^9
    - All Node.val are unique
    - p != q
    - p and q will exist in the BST
    
    EXAMPLES:
    Example 1:
        Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
        Output: 6
        Explanation: The LCA of nodes 2 and 8 is 6
    
    Example 2:
        Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
        Output: 2
        Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself
    
    APPROACH: Iterative Using BST Property
    
    Use BST property to navigate towards LCA
    
    TIME: O(h), SPACE: O(1)
    """
    current = root
    
    while current:
        # If both nodes are in left subtree
        if p.val < current.val and q.val < current.val:
            current = current.left
        # If both nodes are in right subtree
        elif p.val > current.val and q.val > current.val:
            current = current.right
        else:
            # Found LCA (one node on each side or current is one of the nodes)
            return current
    
    return None


def lowest_common_ancestor_bst_recursive(root, p, q):
    """
    APPROACH: Recursive Using BST Property
    
    TIME: O(h), SPACE: O(h)
    """
    if not root:
        return None
    
    # If both nodes are in left subtree
    if p.val < root.val and q.val < root.val:
        return lowest_common_ancestor_bst_recursive(root.left, p, q)
    
    # If both nodes are in right subtree
    if p.val > root.val and q.val > root.val:
        return lowest_common_ancestor_bst_recursive(root.right, p, q)
    
    # Found LCA
    return root


# =============================================================================
# PROBLEM 7: CONVERT SORTED ARRAY TO BST (EASY) - 30 MIN
# =============================================================================

def sorted_array_to_bst(nums):
    """
    PROBLEM: Convert Sorted Array to Binary Search Tree
    
    Given an integer array nums where the elements are sorted in ascending order, 
    convert it to a height-balanced binary search tree.
    
    A height-balanced binary tree is a binary tree in which the depth of the two 
    subtrees of every node never differs by more than one.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^4
    - -10^4 <= nums[i] <= 10^4
    - nums is sorted in a strictly increasing order
    
    EXAMPLES:
    Example 1:
        Input: nums = [-10,-3,0,5,9]
        Output: [0,-3,9,-10,null,5]
        Explanation: [0,-10,5,null,-3,null,9] is also accepted
    
    Example 2:
        Input: nums = [1,3]
        Output: [3,1] or [1,null,3]
    
    APPROACH: Recursive Middle Element as Root
    
    Choose middle element as root to ensure balance
    
    TIME: O(n), SPACE: O(log n)
    """
    def build_bst(left, right):
        if left > right:
            return None
        
        # Choose middle element as root
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        
        # Recursively build left and right subtrees
        root.left = build_bst(left, mid - 1)
        root.right = build_bst(mid + 1, right)
        
        return root
    
    return build_bst(0, len(nums) - 1)


# =============================================================================
# PROBLEM 8: RECOVER BINARY SEARCH TREE (HARD) - 60 MIN
# =============================================================================

def recover_bst(root):
    """
    PROBLEM: Recover Binary Search Tree
    
    You are given the root of a binary search tree (BST), where the values of exactly 
    two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [2, 1000]
    - -2^31 <= Node.val <= 2^31 - 1
    
    EXAMPLES:
    Example 1:
        Input: root = [1,3,null,null,2]
        Output: [3,1,null,null,2]
        Explanation: 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid.
    
    Example 2:
        Input: root = [3,1,4,null,null,2]
        Output: [2,1,4,null,null,3]
        Explanation: 2 cannot be in the right subtree of 3 because 2 < 3. Swapping 2 and 3 makes the BST valid.
    
    APPROACH: Inorder Traversal to Find Violations
    
    In correct BST, inorder traversal is sorted.
    Find the two nodes that violate this property.
    
    TIME: O(n), SPACE: O(h)
    """
    def inorder(node):
        nonlocal first, second, prev
        
        if not node:
            return
        
        inorder(node.left)
        
        # Check for violation
        if prev and prev.val > node.val:
            if not first:
                first = prev  # First violation
            second = node  # Second violation (or update)
        
        prev = node
        inorder(node.right)
    
    first = second = prev = None
    inorder(root)
    
    # Swap the values
    if first and second:
        first.val, second.val = second.val, first.val


# =============================================================================
# PROBLEM 9: BINARY SEARCH TREE TO GREATER SUM TREE (MEDIUM) - 45 MIN
# =============================================================================

def bst_to_greater_tree(root):
    """
    PROBLEM: Binary Search Tree to Greater Sum Tree
    
    Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that 
    every key of the original BST is changed to the original key plus the sum of all keys 
    greater than the original key in BST.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4]
    - -10^4 <= Node.val <= 10^4
    - All the values in the tree are unique
    - root is guaranteed to be a valid binary search tree
    
    EXAMPLES:
    Example 1:
        Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
        Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
    
    Example 2:
        Input: root = [0,null,1]
        Output: [1,null,1]
    
    APPROACH: Reverse Inorder Traversal
    
    Traverse right → root → left to process larger values first
    
    TIME: O(n), SPACE: O(h)
    """
    def reverse_inorder(node):
        nonlocal running_sum
        
        if not node:
            return
        
        # Traverse right subtree first (larger values)
        reverse_inorder(node.right)
        
        # Update current node
        running_sum += node.val
        node.val = running_sum
        
        # Traverse left subtree
        reverse_inorder(node.left)
    
    running_sum = 0
    reverse_inorder(root)
    return root


# =============================================================================
# PROBLEM 10: RANGE SUM OF BST (EASY) - 30 MIN
# =============================================================================

def range_sum_bst(root, low, high):
    """
    PROBLEM: Range Sum of BST
    
    Given the root node of a binary search tree and two integers low and high, 
    return the sum of values of all nodes with a value in the inclusive range [low, high].
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 2 * 10^4]
    - 1 <= Node.val <= 10^5
    - 1 <= low <= high <= 10^5
    - All Node.val are unique
    
    EXAMPLES:
    Example 1:
        Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
        Output: 32
        Explanation: Nodes 7, 10, and 15 are in the range [7, 15]. 7 + 10 + 15 = 32.
    
    Example 2:
        Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
        Output: 23
        Explanation: Nodes 6, 7, and 10 are in the range [6, 10]. 6 + 7 + 10 = 23.
    
    APPROACH: Optimized DFS Using BST Property
    
    Prune search space using BST property
    
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return 0
    
    total = 0
    
    # If current node is in range, add its value
    if low <= root.val <= high:
        total += root.val
    
    # Recursively search left subtree if current value > low
    if root.val > low:
        total += range_sum_bst(root.left, low, high)
    
    # Recursively search right subtree if current value < high
    if root.val < high:
        total += range_sum_bst(root.right, low, high)
    
    return total


# =============================================================================
# PROBLEM 11: TRIM A BINARY SEARCH TREE (MEDIUM) - 45 MIN
# =============================================================================

def trim_bst(root, low, high):
    """
    PROBLEM: Trim a Binary Search Tree
    
    Given the root of a binary search tree and the lowest and highest boundaries as low and high, 
    trim the tree so that all its elements lies in [low, high]. Trimming the tree should not 
    change the relative structure of the elements that will remain in the tree.
    
    It can be proven that there is a unique answer.
    
    Return the root of the trimmed binary search tree. Note that the root may change depending 
    on the given bounds.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 10^4]
    - 0 <= Node.val <= 10^4
    - The value of each node in the tree is unique
    - root is guaranteed to be a valid binary search tree
    - 0 <= low <= high <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: root = [1,0,2], low = 1, high = 2
        Output: [1,null,2]
    
    Example 2:
        Input: root = [3,0,4,null,2,null,null,1], low = 1, high = 3
        Output: [3,2,null,1]
    
    APPROACH: Recursive Trimming
    
    Remove nodes outside range, reconnect valid subtrees
    
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return None
    
    # If current node is less than low, trim left subtree and return right
    if root.val < low:
        return trim_bst(root.right, low, high)
    
    # If current node is greater than high, trim right subtree and return left
    if root.val > high:
        return trim_bst(root.left, low, high)
    
    # Current node is in range, trim both subtrees
    root.left = trim_bst(root.left, low, high)
    root.right = trim_bst(root.right, low, high)
    
    return root


# UTILITY FUNCTIONS FOR TESTING
def create_bst_from_list(values):
    """Create BST by inserting values in order"""
    if not values:
        return None
    
    root = TreeNode(values[0])
    for val in values[1:]:
        insert_into_bst_recursive(root, val)
    
    return root


def inorder_traversal(root):
    """Get in-order traversal for testing"""
    if not root:
        return []
    
    return (inorder_traversal(root.left) + 
            [root.val] + 
            inorder_traversal(root.right))


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all BST problems with comprehensive test cases
    """
    print("=== TESTING DAY 8 PROBLEMS ===\n")
    
    # Create test BST:     5
    #                    /   \
    #                   3     7
    #                  / \   / \
    #                 2   4 6   8
    test_bst = create_bst_from_list([5, 3, 7, 2, 4, 6, 8])
    
    print("1. BST Validation Tests:")
    print(f"   Valid BST: {is_valid_bst_correct(test_bst)} (expected: True)")
    
    # Create invalid BST manually
    invalid_bst = TreeNode(5)
    invalid_bst.left = TreeNode(1)
    invalid_bst.right = TreeNode(4)
    invalid_bst.right.left = TreeNode(3)  # 3 < 5 but in right subtree
    invalid_bst.right.right = TreeNode(6)
    
    print(f"   Invalid BST: {is_valid_bst_correct(invalid_bst)} (expected: False)")
    print(f"   Wrong approach: {is_valid_bst_wrong(invalid_bst)} (expected: True - shows error)")
    print()
    
    # Test Search
    print("2. BST Search Tests:")
    search_result = search_bst_iterative(test_bst, 4)
    search_none = search_bst_iterative(test_bst, 10)
    
    print(f"   Search for 4: {search_result.val if search_result else None} (expected: 4)")
    print(f"   Search for 10: {search_none} (expected: None)")
    print()
    
    # Test Insert
    print("3. BST Insert Tests:")
    test_bst_copy = create_bst_from_list([5, 3, 7, 2, 4, 6, 8])
    insert_into_bst_recursive(test_bst_copy, 1)
    insert_into_bst_recursive(test_bst_copy, 9)
    
    inorder_after_insert = inorder_traversal(test_bst_copy)
    print(f"   After inserting 1 and 9: {inorder_after_insert}")
    print(f"   Expected: [1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print()
    
    # Test Delete
    print("4. BST Delete Tests:")
    test_bst_delete = create_bst_from_list([5, 3, 7, 2, 4, 6, 8])
    
    # Delete leaf
    delete_node_bst(test_bst_delete, 2)
    print(f"   After deleting 2: {inorder_traversal(test_bst_delete)}")
    
    # Delete node with one child
    delete_node_bst(test_bst_delete, 3)
    print(f"   After deleting 3: {inorder_traversal(test_bst_delete)}")
    
    # Delete node with two children
    delete_node_bst(test_bst_delete, 5)
    print(f"   After deleting 5: {inorder_traversal(test_bst_delete)}")
    print()
    
    # Test Kth Smallest
    print("5. Kth Smallest Element Tests:")
    test_bst_k = create_bst_from_list([3, 1, 4, None, 2])
    
    k1 = kth_smallest_optimized(test_bst, 1)
    k3 = kth_smallest_optimized(test_bst, 3)
    k5 = kth_smallest_optimized(test_bst, 5)
    
    print(f"   1st smallest: {k1} (expected: 2)")
    print(f"   3rd smallest: {k3} (expected: 4)")
    print(f"   5th smallest: {k5} (expected: 5)")
    print()
    
    # Test Array to BST
    print("6. Sorted Array to BST:")
    sorted_array = [-10, -3, 0, 5, 9]
    bst_from_array = sorted_array_to_bst(sorted_array)
    inorder_result = inorder_traversal(bst_from_array)
    
    print(f"   Input array: {sorted_array}")
    print(f"   BST inorder: {inorder_result}")
    print(f"   ✓ Arrays match: {sorted_array == inorder_result}")
    print()
    
    # Test Range Sum
    print("7. Range Sum BST:")
    range_sum = range_sum_bst(test_bst, 3, 7)
    print(f"   Sum of values in range [3, 7]: {range_sum}")
    print(f"   Expected: 3+4+5+6+7 = 25")


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_bst_property():
    """
    Visual demonstration of BST property and its benefits
    """
    print("\n=== BST PROPERTY DEMONSTRATION ===")
    
    print("BST Structure:     5")
    print("                 /   \\")
    print("                3     7")
    print("               / \\   / \\")
    print("              2   4 6   8")
    
    print("\nIn-order traversal: [2, 3, 4, 5, 6, 7, 8] (sorted!)")
    
    print("\nSearch for 6:")
    print("  1. Start at 5: 6 > 5, go right")
    print("  2. At 7: 6 < 7, go left")
    print("  3. At 6: found!")
    print("  Steps: 3 (vs 7 for linear search)")
    
    print("\nInsert 1:")
    print("  1. Start at 5: 1 < 5, go left")
    print("  2. At 3: 1 < 3, go left")
    print("  3. At 2: 1 < 2, go left")
    print("  4. No left child, insert 1 as left child of 2")


def bst_vs_array_comparison():
    """
    Compare BST operations with array operations
    """
    print("\n=== BST VS ARRAY COMPARISON ===")
    
    operations = [
        ("Search", "O(log n)", "O(n) unsorted, O(log n) sorted"),
        ("Insert", "O(log n)", "O(1) end, O(n) middle"),
        ("Delete", "O(log n)", "O(n) find + shift"),
        ("Min/Max", "O(log n)", "O(n) unsorted, O(1) sorted"),
        ("Range Query", "O(log n + k)", "O(n)"),
        ("In-order", "O(n)", "O(n log n) sort first")
    ]
    
    print("Operation   | BST (balanced) | Array")
    print("------------|----------------|-------")
    for op, bst_time, array_time in operations:
        print(f"{op:11} | {bst_time:14} | {array_time}")
    
    print("\nBST Advantages:")
    print("  • Dynamic size")
    print("  • Efficient insertion/deletion")
    print("  • Natural ordering")
    print("  • Range queries")
    
    print("\nArray Advantages:")
    print("  • Cache locality")
    print("  • Random access")
    print("  • Less memory overhead")


def balanced_vs_unbalanced():
    """
    Demonstrate importance of balance in BST
    """
    print("\n=== BALANCED VS UNBALANCED BST ===")
    
    print("Balanced BST (height = log n):")
    print("      4")
    print("    /   \\")
    print("   2     6")
    print("  / \\   / \\")
    print(" 1   3 5   7")
    print("Height: 3, Operations: O(log n)")
    
    print("\nUnbalanced BST (height = n):")
    print("1")
    print(" \\")
    print("  2")
    print("   \\")
    print("    3")
    print("     \\")
    print("      4")
    print("Height: 4, Operations: O(n)")
    
    print("\nBalance is crucial for BST efficiency!")
    print("Self-balancing trees (AVL, Red-Black) maintain O(log n)")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Demonstrate BST property
    demonstrate_bst_property()
    
    # Compare with arrays
    bst_vs_array_comparison()
    
    # Show balance importance
    balanced_vs_unbalanced()
    
    print("\n" + "="*70)
    print("DAY 8 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. BST property enables O(log n) operations in balanced trees")
    print("2. In-order traversal of BST gives sorted sequence")
    print("3. Validation requires range checking, not just local comparison")
    print("4. Three deletion cases: leaf, one child, two children")
    print("5. BST operations follow left/right navigation pattern")
    print("6. Balance is crucial - unbalanced BST degrades to O(n)")
    print("7. Range-based problems leverage BST ordering efficiently")
    print("\nTransition: Day 8→9 - From BST to Binary Search Algorithm")
    print("- BST uses binary search principle on tree structure")
    print("- Binary search applies to arrays and search spaces")
    print("- Both use divide-and-conquer with comparison-based decisions")
    print("\nNext: Day 9 - Binary Search Algorithm") 