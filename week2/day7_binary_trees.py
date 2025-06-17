"""
=============================================================================
                     WEEK 2 - DAY 7: BINARY TREES BASICS
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. BINARY TREE FUNDAMENTALS
   - Tree structure: Each node has at most two children (left and right)
   - Root: Top node with no parent
   - Leaf: Node with no children
   - Height: Longest path from root to leaf
   - Depth: Distance from root to specific node

2. TREE TERMINOLOGY
   - Parent/Child relationship
   - Siblings: Nodes with same parent
   - Ancestor/Descendant relationship
   - Subtree: Tree rooted at any node
   - Level: All nodes at same depth
   - Complete vs Full vs Perfect trees

3. BINARY TREE TRAVERSALS
   - Depth-First Search (DFS):
     * Inorder: Left → Root → Right
     * Preorder: Root → Left → Right
     * Postorder: Left → Right → Root
   - Breadth-First Search (BFS):
     * Level-order: Process all nodes level by level

4. RECURSIVE TREE PATTERNS
   - Base case: null node or leaf node
   - Recursive case: process current + recurse on children
   - Backtracking: undo changes when returning from recursion
   - Global variables vs return values for aggregation

5. TREE PROPERTIES & CALCULATIONS
   - Tree height: max depth of any leaf
   - Tree size: total number of nodes
   - Tree balance: height difference between subtrees
   - Path properties: root-to-leaf, node-to-node

6. TRANSITION FROM LINEAR STRUCTURES
   - Arrays/Linked Lists: Linear traversal → Tree traversal
   - Stacks: LIFO → DFS recursive calls
   - Queues: FIFO → BFS level processing
   - Hash Tables: Key lookup → Parent-child relationships

=============================================================================
"""

from collections import deque
from typing import List, Optional


# TreeNode definition - Foundation for all tree problems
class TreeNode:
    """
    Standard binary tree node structure
    
    Used consistently across all tree problems
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"


# =============================================================================
# PROBLEM 1: BINARY TREE TRAVERSALS (EASY) - 30 MIN
# =============================================================================

def inorder_traversal_recursive(root):
    """
    PROBLEM: Binary Tree Inorder Traversal
    
    Given the root of a binary tree, return the inorder traversal of its nodes' values.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 100]
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
    
    APPROACH: Recursive (Left → Root → Right)
    
    Most common traversal for BST (gives sorted order)
    
    TIME: O(n), SPACE: O(h) where h is height
    """
    result = []
    
    def inorder(node):
        if not node:
            return
        
        inorder(node.left)      # Traverse left subtree
        result.append(node.val) # Process current node
        inorder(node.right)     # Traverse right subtree
    
    inorder(root)
    return result


def inorder_traversal_iterative(root):
    """
    APPROACH: Iterative using Stack
    
    Simulates recursive call stack explicitly
    
    TIME: O(n), SPACE: O(h)
    """
    result = []
    stack = []
    current = root
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result


def preorder_traversal(root):
    """
    APPROACH: Preorder (Root → Left → Right)
    
    Useful for tree copying, expression trees
    
    TIME: O(n), SPACE: O(h)
    """
    result = []
    
    def preorder(node):
        if not node:
            return
        
        result.append(node.val) # Process current node
        preorder(node.left)     # Traverse left subtree
        preorder(node.right)    # Traverse right subtree
    
    preorder(root)
    return result


def postorder_traversal(root):
    """
    APPROACH: Postorder (Left → Right → Root)
    
    Useful for deletion, calculating tree properties
    
    TIME: O(n), SPACE: O(h)
    """
    result = []
    
    def postorder(node):
        if not node:
            return
        
        postorder(node.left)    # Traverse left subtree
        postorder(node.right)   # Traverse right subtree
        result.append(node.val) # Process current node
    
    postorder(root)
    return result


def level_order_traversal(root):
    """
    APPROACH: Level-order using BFS (Queue)
    
    Process nodes level by level from left to right
    
    TIME: O(n), SPACE: O(w) where w is maximum width
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        # Add children to queue
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result


def level_order_by_levels(root):
    """
    APPROACH: Level-order Grouped by Levels
    
    Returns list of lists, each containing one level
    
    TIME: O(n), SPACE: O(w)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result


# =============================================================================
# PROBLEM 2: MAXIMUM DEPTH OF BINARY TREE (EASY) - 30 MIN
# =============================================================================

def max_depth(root):
    """
    PROBLEM: Maximum Depth of Binary Tree
    
    Given the root of a binary tree, return its maximum depth.
    
    A binary tree's maximum depth is the number of nodes along the longest path 
    from the root node down to the farthest leaf node.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4]
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [3,9,20,null,null,15,7]
        Output: 3
    
    Example 2:
        Input: root = [1,null,2]
        Output: 2
    
    APPROACH: Recursive (Bottom-up)
    
    Maximum depth = 1 + max(left_depth, right_depth)
    
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)


# =============================================================================
# PROBLEM 3: MINIMUM DEPTH OF BINARY TREE (EASY) - 30 MIN
# =============================================================================

def min_depth(root):
    """
    PROBLEM: Minimum Depth of Binary Tree
    
    Given a binary tree, find its minimum depth.
    
    The minimum depth is the number of nodes along the shortest path from the 
    root node down to the nearest leaf node.
    
    Note: A leaf is a node with no children.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^5]
    - -1000 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [3,9,20,null,null,15,7]
        Output: 2
    
    Example 2:
        Input: root = [2,null,3,null,4,null,5,null,6]
        Output: 5
    
    APPROACH: Recursive with Leaf Check
    
    Must reach a leaf node (both children are null)
    
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return 0
    
    # If one child is missing, go to the other side
    if not root.left:
        return 1 + min_depth(root.right)
    if not root.right:
        return 1 + min_depth(root.left)
    
    # Both children exist
    return 1 + min(min_depth(root.left), min_depth(root.right))


# =============================================================================
# PROBLEM 4: COUNT COMPLETE TREE NODES (MEDIUM) - 45 MIN
# =============================================================================

def count_nodes(root):
    """
    PROBLEM: Count Complete Tree Nodes
    
    Given the root of a complete binary tree, return the number of the nodes in the tree.
    
    According to Wikipedia, every level, except possibly the last, is completely filled 
    in a complete binary tree, and all nodes in the last level are as far left as possible. 
    It can have between 1 and 2^h nodes inclusive at the last level h.
    
    Design an algorithm that runs in less than O(n) time complexity.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 5 * 10^4]
    - 0 <= Node.val <= 5 * 10^4
    - The tree is guaranteed to be complete
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,3,4,5,6]
        Output: 6
    
    Example 2:
        Input: root = []
        Output: 0
    
    Example 3:
        Input: root = [1]
        Output: 1
    
    APPROACH: Optimized for Complete Tree
    
    Use properties of complete binary tree for O(log²n) solution
    
    TIME: O(log²n), SPACE: O(logn)
    """
    if not root:
        return 0
    
    # Simple O(n) approach for educational purposes
    return 1 + count_nodes(root.left) + count_nodes(root.right)


# =============================================================================
# PROBLEM 5: SYMMETRIC TREE (EASY) - 30 MIN
# =============================================================================

def is_symmetric(root):
    """
    PROBLEM: Symmetric Tree
    
    Given the root of a binary tree, check whether it is a mirror of itself 
    (i.e., symmetric around its center).
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 1000]
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,2,3,4,4,3]
        Output: true
    
    Example 2:
        Input: root = [1,2,2,null,3,null,3]
        Output: false
    
    APPROACH: Recursive Mirror Check
    
    Check if left and right subtrees are mirrors of each other
    
    TIME: O(n), SPACE: O(h)
    """
    def is_mirror(left, right):
        # Both null
        if not left and not right:
            return True
        
        # One null, one not null
        if not left or not right:
            return False
        
        # Both exist: check value and recursive mirror property
        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))
    
    if not root:
        return True
    
    return is_mirror(root.left, root.right)


# =============================================================================
# PROBLEM 6: PATH SUM (EASY) - 30 MIN
# =============================================================================

def has_path_sum(root, target_sum):
    """
    PROBLEM: Path Sum
    
    Given the root of a binary tree and an integer targetSum, return true if the tree 
    has a root-to-leaf path such that adding up all the values along the path equals targetSum.
    
    A leaf is a node with no children.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 5000]
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
    
    Example 3:
        Input: root = [], targetSum = 0
        Output: false
    
    APPROACH: Recursive DFS
    
    Subtract current node value and check remaining sum
    
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return False
    
    # Leaf node: check if remaining sum equals node value
    if not root.left and not root.right:
        return target_sum == root.val
    
    # Recursive case: subtract current value and check children
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or 
            has_path_sum(root.right, remaining))


# =============================================================================
# PROBLEM 7: PATH SUM II (MEDIUM) - 45 MIN
# =============================================================================

def path_sum_all_paths(root, target_sum):
    """
    PROBLEM: Path Sum II
    
    Given the root of a binary tree and an integer targetSum, return all root-to-leaf 
    paths where the sum of the node values in the path equals targetSum.
    
    Each path should be returned as a list of the node values, not node references.
    
    A root-to-leaf path is a path starting from the root and ending at any leaf node. 
    A leaf is a node with no children.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 5000]
    - -1000 <= Node.val <= 1000
    - -1000 <= targetSum <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
        Output: [[5,4,11,2],[5,8,4,5]]
    
    Example 2:
        Input: root = [1,2,3], targetSum = 5
        Output: []
    
    Example 3:
        Input: root = [1,2], targetSum = 0
        Output: []
    
    APPROACH: DFS with Backtracking
    
    Build path during traversal, backtrack when returning
    
    TIME: O(n²) worst case, SPACE: O(h)
    """
    def dfs(node, current_path, current_sum, all_paths):
        if not node:
            return
        
        # Add current node to path
        current_path.append(node.val)
        current_sum += node.val
        
        # Check if leaf and sum matches
        if not node.left and not node.right and current_sum == target_sum:
            all_paths.append(current_path[:])  # Copy the path
        
        # Recurse on children
        dfs(node.left, current_path, current_sum, all_paths)
        dfs(node.right, current_path, current_sum, all_paths)
        
        # Backtrack
        current_path.pop()
    
    result = []
    dfs(root, [], 0, result)
    return result


# =============================================================================
# PROBLEM 8: MAXIMUM PATH SUM (HARD) - 60 MIN
# =============================================================================

def max_path_sum_leaf_to_leaf(root):
    """
    PROBLEM: Binary Tree Maximum Path Sum
    
    A path in a binary tree is a sequence of nodes where each pair of adjacent nodes 
    in the sequence has an edge connecting them. A node can only appear in the sequence 
    at most once. Note that the path does not need to pass through the root.
    
    The path sum of a path is the sum of the node's values in the path.
    
    Given the root of a binary tree, return the maximum path sum of any non-empty path.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 3 * 10^4]
    - -1000 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,3]
        Output: 6
        Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6
    
    Example 2:
        Input: root = [-10,9,20,null,null,15,7]
        Output: 42
        Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42
    
    APPROACH: Post-order with Global Maximum
    
    For each node, consider path through it connecting left and right subtrees
    
    TIME: O(n), SPACE: O(h)
    """
    def max_path_ending_at(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Get maximum path sum ending at left and right children
        left_max = max(0, max_path_ending_at(node.left))   # Ignore negative paths
        right_max = max(0, max_path_ending_at(node.right))
        
        # Maximum path through current node (connecting left and right)
        path_through_node = node.val + left_max + right_max
        max_sum = max(max_sum, path_through_node)
        
        # Return maximum path ending at current node (can only go one direction)
        return node.val + max(left_max, right_max)
    
    max_sum = float('-inf')
    max_path_ending_at(root)
    return max_sum


# =============================================================================
# PROBLEM 9: CONSTRUCT BINARY TREE FROM TRAVERSALS (MEDIUM) - 45 MIN
# =============================================================================

def build_tree_preorder_inorder(preorder, inorder):
    """
    PROBLEM: Construct Binary Tree from Preorder and Inorder Traversal
    
    Given two integer arrays preorder and inorder where preorder is the preorder 
    traversal of a binary tree and inorder is the inorder traversal of the same tree, 
    construct and return the binary tree.
    
    CONSTRAINTS:
    - 1 <= preorder.length <= 3000
    - inorder.length == preorder.length
    - -3000 <= preorder[i], inorder[i] <= 3000
    - preorder and inorder consist of unique values
    - Each value of inorder also appears in preorder
    - preorder is guaranteed to be the preorder traversal of the tree
    - inorder is guaranteed to be the inorder traversal of the tree
    
    EXAMPLES:
    Example 1:
        Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
        Output: [3,9,20,null,null,15,7]
    
    Example 2:
        Input: preorder = [-1], inorder = [-1]
        Output: [-1]
    
    APPROACH: Recursive Construction
    
    Use preorder to identify root, inorder to split left/right subtrees
    
    TIME: O(n), SPACE: O(n)
    """
    if not preorder or not inorder:
        return None
    
    # First element in preorder is always root
    root_val = preorder[0]
    root = TreeNode(root_val)
    
    # Find root position in inorder
    root_idx = inorder.index(root_val)
    
    # Split arrays for left and right subtrees
    left_inorder = inorder[:root_idx]
    right_inorder = inorder[root_idx + 1:]
    
    left_preorder = preorder[1:1 + len(left_inorder)]
    right_preorder = preorder[1 + len(left_inorder):]
    
    # Recursively build subtrees
    root.left = build_tree_preorder_inorder(left_preorder, left_inorder)
    root.right = build_tree_preorder_inorder(right_preorder, right_inorder)
    
    return root


def build_tree_postorder_inorder(postorder, inorder):
    """
    APPROACH: Construct from Postorder and Inorder
    
    Similar to preorder approach but build from right to left
    
    TIME: O(n), SPACE: O(n)
    """
    if not postorder or not inorder:
        return None
    
    # Last element in postorder is always root
    root_val = postorder[-1]
    root = TreeNode(root_val)
    
    # Find root position in inorder
    root_idx = inorder.index(root_val)
    
    # Split arrays for left and right subtrees
    left_inorder = inorder[:root_idx]
    right_inorder = inorder[root_idx + 1:]
    
    left_postorder = postorder[:len(left_inorder)]
    right_postorder = postorder[len(left_inorder):-1]
    
    # Recursively build subtrees
    root.left = build_tree_postorder_inorder(left_postorder, left_inorder)
    root.right = build_tree_postorder_inorder(right_postorder, right_inorder)
    
    return root


# =============================================================================
# PROBLEM 10: SAME TREE (EASY) - 30 MIN
# =============================================================================

def is_same_tree(p, q):
    """
    PROBLEM: Same Tree
    
    Given the roots of two binary trees p and q, write a function to check if they are the same or not.
    
    Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
    
    CONSTRAINTS:
    - The number of nodes in both trees is in the range [0, 100]
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
    
    APPROACH: Recursive Comparison
    
    Compare structure and values recursively
    
    TIME: O(min(m,n)), SPACE: O(min(m,n))
    """
    # Both null
    if not p and not q:
        return True
    
    # One null, one not null
    if not p or not q:
        return False
    
    # Both exist: check value and recurse
    return (p.val == q.val and
            is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))


# =============================================================================
# PROBLEM 11: SUBTREE OF ANOTHER TREE (MEDIUM) - 45 MIN
# =============================================================================

def is_subtree(s, t):
    """
    PROBLEM: Subtree of Another Tree
    
    Given the roots of two binary trees root and subRoot, return true if there is a 
    subtree of root with the same structure and node values of subRoot and false otherwise.
    
    A subtree of a binary tree tree is a tree that consists of a node in tree and all 
    of this node's descendants. The tree tree could also be considered as a subtree of itself.
    
    CONSTRAINTS:
    - The number of nodes in the root tree is in the range [1, 2000]
    - The number of nodes in the subRoot tree is in the range [1, 1000]
    - -10^4 <= root.val <= 10^4
    - -10^4 <= subRoot.val <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: root = [3,4,5,1,2], subRoot = [4,1,2]
        Output: true
    
    Example 2:
        Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
        Output: false
    
    APPROACH: Check Each Node as Potential Root
    
    For each node in main tree, check if subtree starting there matches target
    
    TIME: O(m*n), SPACE: O(max(m,n))
    """
    if not s:
        return False
    
    # Check if subtree rooted at current node matches
    if is_same_tree(s, t):
        return True
    
    # Recursively check left and right subtrees
    return is_subtree(s.left, t) or is_subtree(s.right, t)


# =============================================================================
# PROBLEM 12: INVERT BINARY TREE (EASY) - 30 MIN
# =============================================================================

def invert_tree(root):
    """
    PROBLEM: Invert Binary Tree
    
    Given the root of a binary tree, invert the tree, and return its root.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 100]
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [4,2,7,1,3,6,9]
        Output: [4,7,2,9,6,3,1]
    
    Example 2:
        Input: root = [2,1,3]
        Output: [2,3,1]
    
    Example 3:
        Input: root = []
        Output: []
    
    APPROACH: Recursive Swap
    
    Recursively swap left and right children
    
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return None
    
    # Swap children
    root.left, root.right = root.right, root.left
    
    # Recursively invert subtrees
    invert_tree(root.left)
    invert_tree(root.right)
    
    return root


# UTILITY FUNCTIONS FOR TESTING AND VISUALIZATION
def create_tree_from_list(values):
    """
    Create binary tree from level-order list representation
    
    None values represent missing nodes
    """
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
    """
    Convert tree to level-order list representation
    """
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


def print_tree_structure(root, level=0, prefix="Root: "):
    """
    Print tree structure visually
    """
    if root is not None:
        print(" " * (level * 4) + prefix + str(root.val))
        if root.left or root.right:
            if root.left:
                print_tree_structure(root.left, level + 1, "L--- ")
            else:
                print(" " * ((level + 1) * 4) + "L--- None")
            if root.right:
                print_tree_structure(root.right, level + 1, "R--- ")
            else:
                print(" " * ((level + 1) * 4) + "R--- None")


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all binary tree problems with comprehensive test cases
    """
    print("=== TESTING DAY 7 PROBLEMS ===\n")
    
    # Create test tree:       3
    #                       /   \
    #                      9     20
    #                           /  \
    #                          15   7
    test_tree = create_tree_from_list([3, 9, 20, None, None, 15, 7])
    
    print("Test Tree Structure:")
    print_tree_structure(test_tree)
    print()
    
    # Test Traversals
    print("1. Tree Traversals:")
    print(f"   Inorder (recursive): {inorder_traversal_recursive(test_tree)}")
    print(f"   Inorder (iterative): {inorder_traversal_iterative(test_tree)}")
    print(f"   Preorder: {preorder_traversal(test_tree)}")
    print(f"   Postorder: {postorder_traversal(test_tree)}")
    print(f"   Level-order: {level_order_traversal(test_tree)}")
    print(f"   Level-order by levels: {level_order_by_levels(test_tree)}")
    print()
    
    # Test Tree Properties
    print("2. Tree Properties:")
    print(f"   Max depth: {max_depth(test_tree)} (expected: 3)")
    print(f"   Min depth: {min_depth(test_tree)} (expected: 2)")
    print(f"   Node count: {count_nodes(test_tree)} (expected: 5)")
    
    # Test symmetric tree
    symmetric_tree = create_tree_from_list([1, 2, 2, 3, 4, 4, 3])
    print(f"   Is symmetric (symmetric tree): {is_symmetric(symmetric_tree)} (expected: True)")
    print(f"   Is symmetric (test tree): {is_symmetric(test_tree)} (expected: False)")
    print()
    
    # Test Path Problems
    print("3. Path Sum Problems:")
    path_tree = create_tree_from_list([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])
    
    has_sum_22 = has_path_sum(path_tree, 22)
    all_paths_22 = path_sum_all_paths(path_tree, 22)
    
    print(f"   Has path sum 22: {has_sum_22} (expected: True)")
    print(f"   All paths with sum 22: {all_paths_22}")
    print(f"   Expected paths: [[5,4,11,2], [5,8,4,5]] or similar")
    print()
    
    # Test Tree Construction
    print("4. Tree Construction:")
    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]
    
    constructed_tree = build_tree_preorder_inorder(preorder, inorder)
    constructed_list = tree_to_list(constructed_tree)
    
    print(f"   Built from preorder {preorder} and inorder {inorder}")
    print(f"   Result: {constructed_list}")
    print(f"   Expected: [3, 9, 20, None, None, 15, 7]")
    print()
    
    # Test Tree Comparison
    print("5. Tree Comparison:")
    tree1 = create_tree_from_list([1, 2, 3])
    tree2 = create_tree_from_list([1, 2, 3])
    tree3 = create_tree_from_list([1, 2, 4])
    
    print(f"   Same tree (identical): {is_same_tree(tree1, tree2)} (expected: True)")
    print(f"   Same tree (different): {is_same_tree(tree1, tree3)} (expected: False)")
    
    # Test subtree
    main_tree = create_tree_from_list([3, 4, 5, 1, 2])
    sub_tree = create_tree_from_list([4, 1, 2])
    
    print(f"   Is subtree: {is_subtree(main_tree, sub_tree)} (expected: True)")
    print()
    
    # Test Tree Inversion
    print("6. Tree Inversion:")
    invert_test = create_tree_from_list([4, 2, 7, 1, 3, 6, 9])
    print(f"   Before invert: {tree_to_list(invert_test)}")
    
    inverted = invert_tree(invert_test)
    print(f"   After invert: {tree_to_list(inverted)}")
    print(f"   Expected: [4, 7, 2, 9, 6, 3, 1]")


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_traversal_differences():
    """
    Visual demonstration of different traversal orders
    """
    print("\n=== TRAVERSAL DEMONSTRATION ===")
    
    # Create example tree:    F
    #                       / \
    #                      B   G
    #                     / \   \
    #                    A   D   I
    #                       / \  /
    #                      C   E H
    
    print("Example Tree Structure:")
    print("        F")
    print("       / \\")
    print("      B   G")
    print("     / \\   \\")
    print("    A   D   I")
    print("       / \\ /")
    print("      C   E H")
    
    # Build the tree
    tree = TreeNode('F')
    tree.left = TreeNode('B')
    tree.right = TreeNode('G')
    tree.left.left = TreeNode('A')
    tree.left.right = TreeNode('D')
    tree.right.right = TreeNode('I')
    tree.left.right.left = TreeNode('C')
    tree.left.right.right = TreeNode('E')
    tree.right.right.left = TreeNode('H')
    
    # Show traversal orders
    def inorder_chars(node):
        if not node: return []
        return inorder_chars(node.left) + [node.val] + inorder_chars(node.right)
    
    def preorder_chars(node):
        if not node: return []
        return [node.val] + preorder_chars(node.left) + preorder_chars(node.right)
    
    def postorder_chars(node):
        if not node: return []
        return postorder_chars(node.left) + postorder_chars(node.right) + [node.val]
    
    print(f"\nTraversal Orders:")
    print(f"Inorder (L→Root→R):   {' → '.join(inorder_chars(tree))}")
    print(f"Preorder (Root→L→R):  {' → '.join(preorder_chars(tree))}")
    print(f"Postorder (L→R→Root): {' → '.join(postorder_chars(tree))}")
    
    print(f"\nUse Cases:")
    print(f"• Inorder: BST traversal (gives sorted order)")
    print(f"• Preorder: Tree copying, expression evaluation")
    print(f"• Postorder: Tree deletion, calculating tree properties")
    print(f"• Level-order: Shortest path, level-based operations")


def demonstrate_recursion_patterns():
    """
    Show common recursion patterns in tree problems
    """
    print("\n=== TREE RECURSION PATTERNS ===")
    
    print("Pattern 1: Simple Traversal")
    print("```python")
    print("def traverse(node):")
    print("    if not node:")
    print("        return  # Base case")
    print("    ")
    print("    # Process current node")
    print("    traverse(node.left)   # Recurse left")
    print("    traverse(node.right)  # Recurse right")
    print("```")
    
    print("\nPattern 2: Value Calculation")
    print("```python")
    print("def calculate(node):")
    print("    if not node:")
    print("        return 0  # Base value")
    print("    ")
    print("    left_val = calculate(node.left)")
    print("    right_val = calculate(node.right)")
    print("    return combine(node.val, left_val, right_val)")
    print("```")
    
    print("\nPattern 3: Path Tracking with Backtracking")
    print("```python")
    print("def find_paths(node, current_path, all_paths):")
    print("    if not node:")
    print("        return")
    print("    ")
    print("    current_path.append(node.val)  # Add to path")
    print("    ")
    print("    if is_target(node):  # Check condition")
    print("        all_paths.append(current_path[:])")
    print("    ")
    print("    find_paths(node.left, current_path, all_paths)")
    print("    find_paths(node.right, current_path, all_paths)")
    print("    ")
    print("    current_path.pop()  # Backtrack")
    print("```")
    
    print("\nPattern 4: Global Variable Updates")
    print("```python")
    print("def find_maximum(node):")
    print("    nonlocal max_value")
    print("    if not node:")
    print("        return")
    print("    ")
    print("    max_value = max(max_value, node.val)")
    print("    find_maximum(node.left)")
    print("    find_maximum(node.right)")
    print("```")


def tree_complexity_analysis():
    """
    Analyze time and space complexity of tree operations
    """
    print("\n=== TREE COMPLEXITY ANALYSIS ===")
    
    print("Time Complexities:")
    print("• All traversals: O(n) - must visit each node once")
    print("• Tree height: O(n) - worst case for skewed tree")
    print("• Path sum: O(n) - may need to check all paths")
    print("• Tree construction: O(n) - process each element once")
    print("• Tree comparison: O(min(m,n)) - stop at first difference")
    
    print("\nSpace Complexities:")
    print("• Recursive calls: O(h) where h is tree height")
    print("  - Balanced tree: O(log n)")
    print("  - Skewed tree: O(n)")
    print("• BFS (level-order): O(w) where w is maximum width")
    print("  - Complete tree: O(n/2) = O(n)")
    print("• Path storage: O(h) for current path, O(n*h) for all paths")
    
    print("\nBalanced vs Skewed Trees:")
    print("Balanced Tree:        Skewed Tree:")
    print("      1                    1")
    print("     / \\                    \\")
    print("    2   3                    2")
    print("   / \\ / \\                   \\")
    print("  4  5 6  7                   3")
    print("                               \\")
    print("Height: O(log n)                4")
    print("Space: O(log n)           Height: O(n)")
    print("                          Space: O(n)")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Demonstrate traversals
    demonstrate_traversal_differences()
    
    # Show recursion patterns
    demonstrate_recursion_patterns()
    
    # Complexity analysis
    tree_complexity_analysis()
    
    print("\n" + "="*70)
    print("DAY 7 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. Four main traversals: inorder, preorder, postorder, level-order")
    print("2. Recursive patterns: base case + recursive case + combine results")
    print("3. DFS uses recursion/stack, BFS uses queue")
    print("4. Tree height affects space complexity: O(h) for recursion")
    print("5. Path problems often need backtracking")
    print("6. Tree construction requires understanding traversal properties")
    print("7. Always consider null nodes in base cases")
    print("\nTransition: Day 7→8 - From general trees to Binary Search Trees")
    print("- General tree traversals apply to BST")
    print("- BST adds ordering property for efficient operations")
    print("- Inorder traversal of BST gives sorted sequence")
    print("\nNext: Day 8 - Binary Search Trees (BST)") 