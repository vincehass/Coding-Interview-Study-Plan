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


# Problem 1: Serialize and Deserialize Binary Tree - Tree representation
def serialize(root):
    """
    Serialize tree to string using preorder traversal
    
    Use preorder with null markers for complete representation
    
    Time: O(n), Space: O(n)
    """
    def preorder(node):
        if not node:
            return "null,"
        
        return f"{node.val}," + preorder(node.left) + preorder(node.right)
    
    return preorder(root)


def deserialize(data):
    """
    Deserialize string back to tree
    
    Reconstruct using preorder traversal
    
    Time: O(n), Space: O(n)
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


# Problem 2: Binary Tree Maximum Path Sum - Complex path problem
def max_path_sum(root):
    """
    Find maximum path sum between any two nodes
    
    Path can start and end at any nodes (not necessarily root-to-leaf)
    
    Key insight: For each node, max path through it is
    node.val + max_left_path + max_right_path
    
    Time: O(n), Space: O(h)
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
        
        # Max path through current node
        path_through_node = node.val + left_max + right_max
        max_sum = max(max_sum, path_through_node)
        
        # Return max path ending at current node (can only use one side)
        return node.val + max(left_max, right_max)
    
    max_path_ending_at(root)
    return max_sum


# Problem 3: Flatten Binary Tree to Linked List - Tree modification
def flatten_tree_to_list(root):
    """
    Flatten binary tree to linked list in-place
    
    Use preorder traversal: root -> left subtree -> right subtree
    
    Time: O(n), Space: O(h)
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
    Iterative approach using stack
    
    Simulate preorder traversal with explicit stack
    
    Time: O(n), Space: O(h)
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


# Problem 4: Populating Next Right Pointers - Tree augmentation
class TreeNodeWithNext:
    """Tree node with additional next pointer"""
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


def connect_next_pointers(root):
    """
    Connect each node to its next right node in same level
    
    Perfect binary tree version (every level fully filled)
    
    Time: O(n), Space: O(1)
    """
    if not root:
        return root
    
    # Start with leftmost node of each level
    leftmost = root
    
    while leftmost.left:  # While not at leaf level
        # Iterate through current level
        head = leftmost
        
        while head:
            # Connect children
            head.left.next = head.right
            
            # Connect to next node's left child
            if head.next:
                head.right.next = head.next.left
            
            head = head.next
        
        # Move to next level
        leftmost = leftmost.left
    
    return root


def connect_next_pointers_general(root):
    """
    Connect next pointers for any binary tree (not necessarily perfect)
    
    Time: O(n), Space: O(1)
    """
    def get_next_child(node):
        """Get next child node in the level"""
        while node:
            if node.left:
                return node.left
            if node.right:
                return node.right
            node = node.next
        return None
    
    if not root:
        return root
    
    level_start = root
    
    while level_start:
        current = level_start
        level_start = None
        prev_child = None
        
        while current:
            for child in [current.left, current.right]:
                if child:
                    if prev_child:
                        prev_child.next = child
                    else:
                        level_start = child
                    prev_child = child
            
            current = current.next
    
    return root


# Problem 5: Binary Tree Vertical Order Traversal - Complex traversal
def vertical_order_traversal(root):
    """
    Return vertical order traversal of binary tree
    
    Nodes at same column and row are sorted by value
    
    Time: O(n log n), Space: O(n)
    """
    if not root:
        return []
    
    # Map: column -> list of (row, value)
    column_map = defaultdict(list)
    
    def dfs(node, row, col):
        if not node:
            return
        
        column_map[col].append((row, node.val))
        dfs(node.left, row + 1, col - 1)
        dfs(node.right, row + 1, col + 1)
    
    dfs(root, 0, 0)
    
    # Sort columns and within each column sort by row then value
    result = []
    for col in sorted(column_map.keys()):
        # Sort by row first, then by value
        column_map[col].sort(key=lambda x: (x[0], x[1]))
        result.append([val for row, val in column_map[col]])
    
    return result


# Problem 6: Binary Tree Zigzag Level Order Traversal - Modified BFS
def zigzag_level_order(root):
    """
    Return zigzag level order traversal
    
    Alternate between left-to-right and right-to-left for each level
    
    Time: O(n), Space: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level_values = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level_values.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Reverse if right-to-left
        if not left_to_right:
            level_values.reverse()
        
        result.append(level_values)
        left_to_right = not left_to_right
    
    return result


# ADVANCED PROBLEMS FOR EXTRA PRACTICE

def boundary_traversal(root):
    """
    Return boundary traversal: left boundary + leaves + right boundary
    
    Time: O(n), Space: O(h)
    """
    if not root:
        return []
    
    def is_leaf(node):
        return not node.left and not node.right
    
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
        result.append(node.val)
    
    def leaves(node, result):
        if not node:
            return
        
        if is_leaf(node):
            result.append(node.val)
        else:
            leaves(node.left, result)
            leaves(node.right, result)
    
    if is_leaf(root):
        return [root.val]
    
    result = [root.val]
    left_boundary(root.left, result)
    leaves(root, result)
    right_boundary(root.right, result)
    
    return result


def recover_from_preorder(s):
    """
    Recover binary tree from preorder traversal string
    
    String format: "1-2--3--4-5--6--7"
    
    Time: O(n), Space: O(h)
    """
    def build():
        nonlocal index
        if index >= len(nodes):
            return None
        
        depth, val = nodes[index]
        if depth != current_depth:
            return None
        
        index += 1
        node = TreeNode(val)
        current_depth += 1
        node.left = build()
        node.right = build()
        current_depth -= 1
        
        return node
    
    if not s:
        return None
    
    # Parse string to get (depth, value) pairs
    nodes = []
    i = 0
    while i < len(s):
        depth = 0
        while i < len(s) and s[i] == '-':
            depth += 1
            i += 1
        
        val = 0
        negative = False
        if i < len(s) and s[i] == '-':
            negative = True
            i += 1
        
        while i < len(s) and s[i].isdigit():
            val = val * 10 + int(s[i])
            i += 1
        
        if negative:
            val = -val
        
        nodes.append((depth, val))
    
    index = 0
    current_depth = 0
    return build()


def longest_univalue_path(root):
    """
    Find longest path where all nodes have same value
    
    Path can be between any two nodes
    
    Time: O(n), Space: O(h)
    """
    max_length = 0
    
    def longest_path_ending_at(node):
        nonlocal max_length
        
        if not node:
            return 0
        
        # Get longest univalue path from left and right
        left_length = longest_path_ending_at(node.left)
        right_length = longest_path_ending_at(node.right)
        
        # Reset if value differs
        if node.left and node.left.val != node.val:
            left_length = 0
        if node.right and node.right.val != node.val:
            right_length = 0
        
        # Update global maximum (path through current node)
        max_length = max(max_length, left_length + right_length)
        
        # Return longest path ending at current node
        return 1 + max(left_length, right_length)
    
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