"""
=============================================================================
                    WEEK 2 - DAY 10: ADVANCED TREE PROBLEMS
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. ADVANCED TREE CONCEPTS
   - Tree modification: In-place transformations
   - Tree serialization: Convert to/from string representation
   - Path problems: Complex path finding and calculations
   - Tree views: Different perspectives of same tree
   - Parent-child relationships: Navigate upward in tree

2. COMPLEX TREE PATTERNS
   - Path sum variations: Any path, not just root-to-leaf
   - Tree flattening: Convert to linear structure
   - Next pointer problems: Add horizontal connections
   - Vertical/zigzag traversals: Non-standard orderings
   - Tree reconstruction: Build from various inputs

3. OPTIMIZATION TECHNIQUES
   - Memoization in tree recursion
   - Multi-pass algorithms: Gather info, then process
   - Bottom-up vs top-down approaches
   - Early termination conditions
   - Space optimization in tree operations

4. TREE MODIFICATION STRATEGIES
   - In-place modifications: Change structure without extra space
   - Preserving original structure vs creating new
   - Handling edge cases: Empty trees, single nodes
   - Maintaining invariants during modifications

5. ADVANCED TRAVERSAL PATTERNS
   - Spiral/zigzag order traversal
   - Vertical order traversal
   - Boundary traversal
   - Morris traversal (O(1) space)
   - Custom traversal orders

6. INTEGRATION WITH PREVIOUS CONCEPTS
   - Hash tables for parent tracking and path storage
   - Binary search in tree problems
   - Two pointers concept in tree navigation
   - Stack/queue for complex traversals

=============================================================================
"""

from collections import deque, defaultdict
from typing import List, Optional


# TreeNode definition from previous days
class TreeNode:
    """Standard binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"


# =============================================================================
# PROBLEM 1: SERIALIZE AND DESERIALIZE BINARY TREE (HARD) - 60 MIN
# =============================================================================

def serialize(root):
    """
    PROBLEM: Serialize and Deserialize Binary Tree
    
    Serialization is the process of converting a data structure or object into a sequence of bits 
    so that it can be stored in a file or memory buffer, or transmitted across a network connection 
    link to be reconstructed later in the same or another computer environment.
    
    Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how 
    your serialization/deserialization algorithm should work. You just need to ensure that a binary 
    tree can be serialized to a string and this string can be deserialized to the original tree structure.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4]
    - -1000 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,3,null,null,4,5]
        Output: [1,2,3,null,null,4,5]
        
    Example 2:
        Input: root = []
        Output: []
        
    Example 3:
        Input: root = [1]
        Output: [1]
        
    Example 4:
        Input: root = [1,2]
        Output: [1,2]
    
    APPROACH: Preorder Traversal with Null Markers
    
    Use preorder traversal (root, left, right) to serialize the tree.
    Include null markers to represent missing nodes, enabling unique reconstruction.
    During deserialization, use the same preorder pattern to rebuild the tree.
    
    The key insight is that preorder traversal with null markers provides enough
    information to uniquely reconstruct the tree structure.
    
    TIME: O(n) for both serialize and deserialize, SPACE: O(n) for recursion stack and output
    """
    def preorder(node):
        if not node:
            return "null,"
        
        return f"{node.val}," + preorder(node.left) + preorder(node.right)
    
    return preorder(root)


def deserialize(data):
    """
    Deserialize string back to tree
    
    Reconstruct using preorder traversal pattern
    
    TIME: O(n), SPACE: O(n)
    """
    def build():
        val = next(values)
        if val == "null":
            return None
        
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    
    values = iter(data.split(','))
    return build()


# =============================================================================
# PROBLEM 2: BINARY TREE MAXIMUM PATH SUM (HARD) - 60 MIN
# =============================================================================

def max_path_sum(root):
    """
    PROBLEM: Binary Tree Maximum Path Sum
    
    A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence 
    has an edge connecting them. A node can only appear in the sequence at most once. Note that the 
    path does not need to pass through the root.
    
    The path sum of a path is the sum of the node's values in the path.
    
    Given the root of a binary tree, return the maximum path sum of any non-empty path.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 3 * 10^4]
    - -1000 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,3]
        Output: 6
        Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
        
    Example 2:
        Input: root = [-10,9,20,null,null,15,7]
        Output: 42
        Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
        
    Example 3:
        Input: root = [-3]
        Output: -3
        
    Example 4:
        Input: root = [2,-1]
        Output: 2
    
    APPROACH: Recursive DFS with Global Maximum
    
    For each node, we calculate the maximum path sum that can be achieved by:
    1. A path ending at this node (going down to one subtree)
    2. A path passing through this node (connecting both subtrees)
    
    Key insights:
    - For each node, the maximum path through it is: node.val + max_left_path + max_right_path
    - But we can only return one path ending at the node: node.val + max(left_path, right_path)
    - Use max(0, path) to ignore negative paths
    - Keep track of global maximum across all nodes
    
    TIME: O(n) - visit each node once, SPACE: O(h) - recursion stack depth
    """
    max_sum = float('-inf')
    
    def max_path_ending_at(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Get max path sum ending at left and right children
        # Use max(0, ...) to ignore negative paths
        left_max = max(0, max_path_ending_at(node.left))
        right_max = max(0, max_path_ending_at(node.right))
        
        # Max path through current node (connects both subtrees)
        path_through_node = node.val + left_max + right_max
        max_sum = max(max_sum, path_through_node)
        
        # Return max path ending at current node (can only use one side)
        return node.val + max(left_max, right_max)
    
    max_path_ending_at(root)
    return max_sum


# =============================================================================
# PROBLEM 3: FLATTEN BINARY TREE TO LINKED LIST (MEDIUM) - 45 MIN
# =============================================================================

def flatten_tree_to_list(root):
    """
    PROBLEM: Flatten Binary Tree to Linked List
    
    Given the root of a binary tree, flatten the tree into a "linked list":
    - The "linked list" should use the same TreeNode class where the right child pointer points 
      to the next node in the list and the left child pointer is always null.
    - The "linked list" should be in the same order as a preorder traversal of the binary tree.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 2000]
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,5,3,4,null,6]
        Output: [1,null,2,null,3,null,4,null,5,null,6]
        Explanation: 
        Original tree:    Flattened:
            1                1
           / \                \
          2   5                2
         / \   \                \
        3   4   6                3
                                  \
                                   4
                                    \
                                     5
                                      \
                                       6
                                       
    Example 2:
        Input: root = []
        Output: []
        
    Example 3:
        Input: root = [0]
        Output: [0]
    
    APPROACH: Recursive In-Place Modification
    
    The key insight is to use preorder traversal pattern: process root, then left subtree, then right subtree.
    
    For each node:
    1. Recursively flatten left and right subtrees
    2. If left subtree exists, insert it between current node and right subtree
    3. Connect the tail of left subtree to the right subtree
    4. Set left pointer to null
    
    Return the tail of the flattened subtree for connection purposes.
    
    TIME: O(n) - visit each node once, SPACE: O(h) - recursion stack depth
    """
    def flatten(node):
        if not node:
            return None
        
        # Base case: leaf node
        if not node.left and not node.right:
            return node
        
        # Recursively flatten left and right subtrees
        left_tail = flatten(node.left)
        right_tail = flatten(node.right)
        
        # If left subtree exists, rewire connections
        if left_tail:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        
        # Return rightmost node
        return right_tail if right_tail else left_tail
    
    flatten(root)


def flatten_iterative(root):
    """
    APPROACH 2: Iterative with Stack
    
    Simulate preorder traversal with explicit stack.
    Process nodes in preorder and connect them linearly.
    
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return
    
    stack = [root]
    
    while stack:
        current = stack.pop()
        
        # Push children in reverse order (right first)
        if current.right:
            stack.append(current.right)
        if current.left:
            stack.append(current.left)
        
        # Connect to next node in preorder
        if stack:
            current.right = stack[-1]
        current.left = None


# =============================================================================
# PROBLEM 4: POPULATING NEXT RIGHT POINTERS (MEDIUM) - 45 MIN
# =============================================================================

class TreeNodeWithNext:
    """Tree node with next pointer for horizontal connections"""
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


def connect_next_pointers(root):
    """
    PROBLEM: Populating Next Right Pointers in Each Node
    
    You are given a perfect binary tree where all leaves are on the same level, and every parent 
    has two children. Populate each next pointer to point to its next right node. If there is no 
    next right node, the next pointer should be set to NULL.
    
    Initially, all next pointers are set to NULL.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 2^12 - 1]
    - -1000 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,3,4,5,6,7]
        Output: [1,#,2,3,#,4,5,6,7,#]
        Explanation: Given the above perfect binary tree, your function should populate each next pointer 
        to point to its next right node. The serialized output is in level order as connected by the next 
        pointers, with '#' signifying the end of each level.
        
    Example 2:
        Input: root = []
        Output: []
    
    APPROACH: Level-by-Level Connection using Existing Next Pointers
    
    Since it's a perfect binary tree, we can use the next pointers we've already established
    to navigate horizontally and connect the next level.
    
    For each level:
    1. Use the current level's next pointers to traverse horizontally
    2. Connect the children of adjacent nodes
    3. Move to the next level
    
    This achieves O(1) space complexity by using existing structure.
    
    TIME: O(n) - visit each node once, SPACE: O(1) - no extra space needed
    """
    if not root:
        return root
    
    # Start with the root level
    leftmost = root
    
    # Process each level
    while leftmost.left:  # While not at leaf level
        # Traverse current level using next pointers
        head = leftmost
        
        while head:
            # Connect left child to right child
            head.left.next = head.right
            
            # Connect right child to next node's left child
            if head.next:
                head.right.next = head.next.left
            
            # Move to next node in current level
            head = head.next
        
        # Move to next level
        leftmost = leftmost.left
    
    return root


def connect_next_pointers_general(root):
    """
    PROBLEM: Populating Next Right Pointers in Each Node II
    
    Given a binary tree (not necessarily perfect), populate each next pointer to point to its next right node.
    
    APPROACH: Level-by-Level with Dummy Node
    
    Use a dummy node to simplify the connection process for each level.
    
    TIME: O(n), SPACE: O(1)
    """
    def get_next_child(node):
        """Find the next child node in the current level"""
        while node:
            if node.left:
                return node.left
            if node.right:
                return node.right
            node = node.next
        return None
    
    if not root:
        return root
    
    # Start with root level
    curr_level_start = root
    
    while curr_level_start:
        # Dummy node to simplify connections
        dummy = TreeNodeWithNext(0)
        prev = dummy
        
        # Traverse current level
        curr = curr_level_start
        while curr:
            # Connect left child
            if curr.left:
                prev.next = curr.left
                prev = prev.next
            
            # Connect right child
            if curr.right:
                prev.next = curr.right
                prev = prev.next
            
            # Move to next node in current level
            curr = curr.next
        
        # Move to next level
        curr_level_start = dummy.next
    
    return root


# =============================================================================
# PROBLEM 5: VERTICAL ORDER TRAVERSAL (HARD) - 60 MIN
# =============================================================================

def vertical_order_traversal(root):
    """
    PROBLEM: Vertical Order Traversal of a Binary Tree
    
    Given the root of a binary tree, calculate the vertical order traversal of the binary tree.
    
    For each node at position (row, col), its left and right children will be at positions 
    (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of the tree is at (0, 0).
    
    The vertical order traversal of a binary tree is a list of top-to-bottom orderings for each 
    column index from left to right. If two nodes are in the same row and column, the order should 
    be from left to right.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 1000]
    - 0 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [3,9,20,null,null,15,7]
        Output: [[9],[3,15],[20],[7]]
        Explanation:
        Column -1: Only node 9 is in this column.
        Column 0: Nodes 3 and 15 are in this column in that order from top to bottom.
        Column 1: Only node 20 is in this column.
        Column 2: Only node 7 is in this column.
        
    Example 2:
        Input: root = [1,2,3,4,5,6,7]
        Output: [[4],[2],[1,5,6],[3],[7]]
        
    Example 3:
        Input: root = [1,2,3,4,6,5,7]
        Output: [[4],[2],[1,5,6],[3],[7]]
    
    APPROACH: DFS with Coordinate Tracking and Sorting
    
    1. Assign coordinates (row, col) to each node
    2. Use DFS to traverse and collect all node positions
    3. Group nodes by column
    4. Sort within each column by row, then by value
    5. Return columns in left-to-right order
    
    TIME: O(n log n) - sorting nodes, SPACE: O(n) - storing coordinates
    """
    if not root:
        return []
    
    # Store (col, row, val) for each node
    nodes = []
    
    def dfs(node, row, col):
        if not node:
            return
        
        nodes.append((col, row, node.val))
        dfs(node.left, row + 1, col - 1)
        dfs(node.right, row + 1, col + 1)
    
    dfs(root, 0, 0)
    
    # Sort by column, then row, then value
    nodes.sort()
    
    # Group by column
    result = []
    prev_col = float('-inf')
    
    for col, row, val in nodes:
        if col != prev_col:
            result.append([])
            prev_col = col
        result[-1].append(val)
    
    return result


# =============================================================================
# PROBLEM 6: BINARY TREE ZIGZAG LEVEL ORDER TRAVERSAL (MEDIUM) - 45 MIN
# =============================================================================

def zigzag_level_order(root):
    """
    PROBLEM: Binary Tree Zigzag Level Order Traversal
    
    Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. 
    (i.e., from left to right, then right to left for the next level and alternate between).
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 2000]
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [3,9,20,null,null,15,7]
        Output: [[3],[20,9],[15,7]]
        Explanation:
        Level 0: [3] (left to right)
        Level 1: [20,9] (right to left)
        Level 2: [15,7] (left to right)
        
    Example 2:
        Input: root = [1]
        Output: [[1]]
        
    Example 3:
        Input: root = []
        Output: []
    
    APPROACH: BFS with Direction Flag
    
    Use standard level-order traversal (BFS) but alternate the direction
    of adding nodes to each level's result.
    
    Use a flag to track whether current level should be left-to-right or right-to-left.
    
    TIME: O(n) - visit each node once, SPACE: O(w) - maximum width of tree for queue
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Reverse if right-to-left direction
        if not left_to_right:
            level_nodes.reverse()
        
        result.append(level_nodes)
        left_to_right = not left_to_right  # Flip direction
    
    return result


# =============================================================================
# PROBLEM 7: BOUNDARY TRAVERSAL OF BINARY TREE (MEDIUM) - 45 MIN
# =============================================================================

def boundary_traversal(root):
    """
    PROBLEM: Boundary Traversal of Binary Tree
    
    Given a binary tree, return the values of its boundary in anti-clockwise direction starting from root.
    The boundary includes left boundary, leaves, and right boundary in order without duplicate nodes.
    
    Left boundary: defined as the path from root to the left-most node.
    Right boundary: defined as the path from root to the right-most node.
    If the root doesn't have left/right subtree, then the root itself is left/right boundary.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 10^4]
    - -1000 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [1,null,2,3,4]
        Output: [1,3,4,2]
        Explanation: 
        Left boundary: [1] (root)
        Leaves: [3,4]
        Right boundary: [2] (excluding root)
        Result: [1,3,4,2]
        
    Example 2:
        Input: root = [1,2,3,4,5,6,null,null,null,7,8,9,10]
        Output: [1,2,4,7,8,9,10,6,3]
    
    APPROACH: Three Separate Traversals
    
    1. Left boundary: Go left preferentially, then right if no left child
    2. Leaves: Inorder traversal collecting only leaf nodes
    3. Right boundary: Go right preferentially, then left if no right child (in reverse)
    
    Combine results while avoiding duplicates (root, leftmost leaf, rightmost leaf).
    
    TIME: O(n) - three traversals, SPACE: O(h) - recursion stack
    """
    def is_leaf(node):
        return node and not node.left and not node.right
    
    def left_boundary(node, result):
        if not node or is_leaf(node):
            return
        
        result.append(node.val)
        
        if node.left:
            left_boundary(node.left, result)
        else:
            left_boundary(node.right, result)
    
    def right_boundary(node, result):
        if not node or is_leaf(node):
            return
        
        if node.right:
            right_boundary(node.right, result)
        else:
            right_boundary(node.left, result)
        
        result.append(node.val)  # Add after recursion for reverse order
    
    def leaves(node, result):
        if not node:
            return
        
        if is_leaf(node):
            result.append(node.val)
            return
        
        leaves(node.left, result)
        leaves(node.right, result)
    
    if not root:
        return []
    
    result = [root.val]
    
    # Add left boundary (excluding root and leaves)
    if root.left:
        left_boundary(root.left, result)
    
    # Add leaves (excluding root if it's a leaf)
    if not is_leaf(root):
        leaves(root, result)
    
    # Add right boundary (excluding root and leaves)
    if root.right:
        right_boundary(root.right, result)
    
    return result


# =============================================================================
# PROBLEM 8: RECOVER TREE FROM PREORDER TRAVERSAL (HARD) - 60 MIN
# =============================================================================

def recover_from_preorder(s):
    """
    PROBLEM: Recover a Tree From Preorder Traversal
    
    We run a preorder depth-first search (DFS) on the root of a binary tree.
    At each node in this traversal, we output D dashes (where D is the depth of this node), 
    then we output the value of this node.
    
    If the depth of a node is D, the depth of its immediate child is D + 1. The depth of the root node is 0.
    
    Given the output of this traversal as a string, recover the tree and return its root.
    
    CONSTRAINTS:
    - The number of nodes in the original tree is in the range [1, 1000]
    - 1 <= Node.val <= 10^9
    
    EXAMPLES:
    Example 1:
        Input: s = "1-2--3--4-5--6--7"
        Output: [1,2,5,3,4,6,7]
        Explanation: The preorder traversal is:
        1 (depth 0)
        -2 (depth 1)
        --3 (depth 2)
        --4 (depth 2)
        -5 (depth 1)
        --6 (depth 2)
        --7 (depth 2)
        
    Example 2:
        Input: s = "1-2--3---4-5--6---7"
        Output: [1,2,5,3,null,6,null,4,null,7]
        
    Example 3:
        Input: s = "1-401--349---90--88"
        Output: [1,401,null,349,88,90]
    
    APPROACH: Recursive Parsing with Depth Tracking
    
    Parse the string character by character:
    1. Count dashes to determine depth
    2. Parse the number value
    3. Recursively build left and right subtrees at the correct depth
    4. Use a stack to keep track of nodes at different depths
    
    TIME: O(n) - parse string once, SPACE: O(h) - recursion stack and node stack
    """
    def build():
        nonlocal index
        
        if index >= len(s):
            return None
        
        # Count dashes to get depth
        depth = 0
        while index < len(s) and s[index] == '-':
            depth += 1
            index += 1
        
        # If we've gone too deep, backtrack
        if depth != expected_depth:
            index -= depth  # Backtrack
            return None
        
        # Parse number
        start = index
        while index < len(s) and s[index].isdigit():
            index += 1
        
        if start == index:  # No number found
            return None
        
        val = int(s[start:index])
        node = TreeNode(val)
        
        # Build left subtree at depth + 1
        expected_depth = depth + 1
        node.left = build()
        
        # Build right subtree at depth + 1
        expected_depth = depth + 1
        node.right = build()
        
        return node
    
    index = 0
    expected_depth = 0
    return build()


# =============================================================================
# PROBLEM 9: LONGEST UNIVALUE PATH (MEDIUM) - 45 MIN
# =============================================================================

def longest_univalue_path(root):
    """
    PROBLEM: Longest Univalue Path
    
    Given the root of a binary tree, return the length of the longest path, where each node in the path 
    has the same value. This path may or may not pass through the root.
    
    The length of the path between two nodes is represented by the number of edges between them.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4]
    - -1000 <= Node.val <= 1000
    - The depth of the tree will not exceed 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [5,4,5,1,1,null,5]
        Output: 2
        Explanation: The longest path with same values is [5,5,5]. The length is 2.
        
    Example 2:
        Input: root = [1,4,5,4,4,null,5]
        Output: 2
        Explanation: The longest path with same values is [4,4,4]. The length is 2.
        
    Example 3:
        Input: root = [1]
        Output: 0
        
    Example 4:
        Input: root = [1,null,1,1,1,1,1,1]
        Output: 4
    
    APPROACH: DFS with Path Length Tracking
    
    For each node, calculate:
    1. Longest univalue path ending at this node (going down one side)
    2. Longest univalue path passing through this node (connecting both sides)
    
    Similar to maximum path sum problem, but with value equality constraint.
    
    TIME: O(n) - visit each node once, SPACE: O(h) - recursion stack
    """
    max_length = 0
    
    def longest_path_ending_at(node):
        nonlocal max_length
        
        if not node:
            return 0
        
        # Get longest univalue paths from children
        left_length = longest_path_ending_at(node.left)
        right_length = longest_path_ending_at(node.right)
        
        # Reset lengths if values don't match
        if node.left and node.left.val != node.val:
            left_length = 0
        if node.right and node.right.val != node.val:
            right_length = 0
        
        # Path through current node
        path_through_node = left_length + right_length
        max_length = max(max_length, path_through_node)
        
        # Return longest path ending at current node
        return max(left_length, right_length) + 1
    
    longest_path_ending_at(root)
    return max_length


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all advanced tree problems with comprehensive test cases
    """
    print("=== TESTING DAY 10 PROBLEMS ===\n")
    
    # Create test tree:      1
    #                      / \
    #                     2   3
    #                    / \
    #                   4   5
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    # Test Serialization
    print("1. Tree Serialization:")
    serialized = serialize(root)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    
    print(f"   Original serialization: {serialized}")
    print(f"   After deserialize->serialize: {reserialized}")
    print(f"   ✓ Correct: {serialized == reserialized}")
    print()
    
    # Test Maximum Path Sum
    print("2. Maximum Path Sum:")
    # Tree: [-10, 9, 20, null, null, 15, 7]
    path_tree = TreeNode(-10)
    path_tree.left = TreeNode(9)
    path_tree.right = TreeNode(20)
    path_tree.right.left = TreeNode(15)
    path_tree.right.right = TreeNode(7)
    
    max_sum = max_path_sum(path_tree)
    print(f"   Tree: [-10, 9, 20, null, null, 15, 7]")
    print(f"   Maximum path sum: {max_sum} (expected: 42)")
    print(f"   Path: 15 -> 20 -> 7")
    print()
    
    # Test Tree Flattening
    print("3. Flatten Binary Tree:")
    flatten_tree = TreeNode(1)
    flatten_tree.left = TreeNode(2)
    flatten_tree.right = TreeNode(5)
    flatten_tree.left.left = TreeNode(3)
    flatten_tree.left.right = TreeNode(4)
    flatten_tree.right.right = TreeNode(6)
    
    print("   Before flattening: [1, 2, 5, 3, 4, null, 6]")
    flatten_tree_to_list(flatten_tree)
    
    # Check if flattened correctly
    flattened_vals = []
    current = flatten_tree
    while current:
        flattened_vals.append(current.val)
        current = current.right
    
    print(f"   After flattening: {flattened_vals}")
    print(f"   Expected: [1, 2, 3, 4, 5, 6]")
    print()
    
    # Test Zigzag Traversal
    print("4. Zigzag Level Order Traversal:")
    zigzag_tree = TreeNode(3)
    zigzag_tree.left = TreeNode(9)
    zigzag_tree.right = TreeNode(20)
    zigzag_tree.right.left = TreeNode(15)
    zigzag_tree.right.right = TreeNode(7)
    
    zigzag_result = zigzag_level_order(zigzag_tree)
    print(f"   Tree: [3, 9, 20, null, null, 15, 7]")
    print(f"   Zigzag traversal: {zigzag_result}")
    print(f"   Expected: [[3], [20, 9], [15, 7]]")
    print()


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_tree_modification_patterns():
    """
    Show common patterns for tree modification
    """
    print("\n=== TREE MODIFICATION PATTERNS ===")
    
    print("Pattern 1: In-place modification")
    print("  • Flatten tree to linked list")
    print("  • Mirror/invert binary tree")
    print("  • Connect next right pointers")
    
    print("\nPattern 2: Path-based calculations")
    print("  • Maximum path sum")
    print("  • Longest univalue path")
    print("  • Path sum variations")
    
    print("\nPattern 3: Tree reconstruction")
    print("  • Build from traversal strings")
    print("  • Serialize/deserialize")
    print("  • Recover from preorder")
    
    print("\nPattern 4: Complex traversals")
    print("  • Vertical order traversal")
    print("  • Zigzag level order")
    print("  • Boundary traversal")
    
    print("\nKey Insights:")
    print("  • Many problems require multi-pass approach")
    print("  • Consider both top-down and bottom-up solutions")
    print("  • Use helper functions to simplify complex logic")
    print("  • Hash maps often helpful for tracking relationships")


def advanced_tree_complexity_analysis():
    """
    Analyze complexity of advanced tree algorithms
    """
    print("\n=== ADVANCED TREE COMPLEXITY ANALYSIS ===")
    
    problems = [
        ("Serialize/Deserialize", "O(n)", "O(n)"),
        ("Maximum Path Sum", "O(n)", "O(h)"),
        ("Flatten Tree", "O(n)", "O(h)"),
        ("Connect Next Pointers", "O(n)", "O(1)"),
        ("Vertical Order", "O(n log n)", "O(n)"),
        ("Zigzag Traversal", "O(n)", "O(n)"),
        ("Boundary Traversal", "O(n)", "O(h)")
    ]
    
    print("Problem                | Time      | Space")
    print("----------------------|-----------|-------")
    for problem, time, space in problems:
        print(f"{problem:21} | {time:9} | {space}")
    
    print("\nSpace Complexity Notes:")
    print("  • h = height of tree (O(log n) balanced, O(n) skewed)")
    print("  • Connect next pointers achieves O(1) with level-by-level processing")
    print("  • Serialization requires O(n) to store all node values")
    print("  • Recursive algorithms typically use O(h) stack space")


def tree_problem_solving_strategy():
    """
    Guide for approaching complex tree problems
    """
    print("\n=== TREE PROBLEM SOLVING STRATEGY ===")
    
    print("Step 1: Understand the Requirements")
    print("  • What type of path/traversal is needed?")
    print("  • Is modification in-place or create new structure?")
    print("  • What information needs to be tracked?")
    
    print("\nStep 2: Choose the Right Approach")
    print("  • DFS vs BFS based on problem nature")
    print("  • Recursive vs iterative based on constraints")
    print("  • Top-down vs bottom-up based on information flow")
    
    print("\nStep 3: Handle Edge Cases")
    print("  • Empty tree (null root)")
    print("  • Single node tree")
    print("  • Skewed tree (height = n)")
    
    print("\nStep 4: Optimize")
    print("  • Can multiple passes be combined?")
    print("  • Can space be reduced with iterative approach?")
    print("  • Are there early termination conditions?")
    
    print("\nCommon Techniques:")
    print("  • Global variables for tracking max/min values")
    print("  • Helper functions for cleaner code")
    print("  • Hash maps for parent/position tracking")
    print("  • Two-pass algorithms for complex calculations")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Show modification patterns
    demonstrate_tree_modification_patterns()
    
    # Complexity analysis
    advanced_tree_complexity_analysis()
    
    # Problem solving strategy
    tree_problem_solving_strategy()
    
    print("\n" + "="*70)
    print("DAY 10 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. Complex tree problems often require multi-step approaches")
    print("2. In-place modifications save space but increase complexity")
    print("3. Path problems: distinguish between paths ending at vs through nodes")
    print("4. Serialization enables tree storage and transmission")
    print("5. Advanced traversals combine BFS/DFS with custom ordering")
    print("6. Global variables useful for tracking across recursive calls")
    print("7. Consider both recursive and iterative solutions")
    print("\nTransition: Day 10→11 - From trees to heaps")
    print("- Trees provide foundation for heap data structure")
    print("- Heaps are specialized trees with ordering properties")
    print("- Priority queues built on heap foundations")
    print("\nNext: Day 11 - Heaps & Priority Queues") 