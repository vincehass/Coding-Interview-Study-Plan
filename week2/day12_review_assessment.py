"""
=============================================================================
                    WEEK 2 - DAY 12: REVIEW & ASSESSMENT  
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. WEEK 2 CONCEPT INTEGRATION
   - Binary Trees: Foundation for all tree-based algorithms
   - BST: Ordered trees enabling O(log n) operations
   - Binary Search: Divide-and-conquer on monotonic spaces
   - Advanced Trees: Complex manipulations and path problems
   - Heaps: Priority-based operations and streaming data

2. ALGORITHM COMPLEXITY MASTERY
   - Tree operations: Generally O(h) where h is height
   - Balanced trees: O(log n) guaranteed performance
   - Binary search: O(log n) on sorted/monotonic data
   - Heap operations: O(log n) insert/delete, O(1) peek
   - Advanced tree algorithms: Often O(n) with careful design

3. PATTERN RECOGNITION SKILLS
   - Traversal patterns: DFS vs BFS selection criteria
   - Path problems: Ending at vs passing through nodes
   - Search space: Array indices vs value ranges
   - Two heaps: Maintaining balanced partitions
   - Tree modification: In-place vs creating new structures

4. META INTERVIEW FOCUS AREAS
   - Tree problems: 40% of tree/graph questions
   - Binary search: Common in array/optimization problems
   - Heap applications: Priority queues, streaming data
   - System design: Tree structures in databases, caches
   - Optimization: Space-time tradeoffs in tree algorithms

5. PROBLEM-SOLVING STRATEGIES
   - Start simple: Binary tree basics before complex modifications
   - Visualize: Draw trees and trace algorithm execution
   - Edge cases: Empty trees, single nodes, skewed trees
   - Multiple approaches: Recursive vs iterative solutions
   - Optimization: Consider space-time tradeoffs

6. INTEGRATION WITH WEEK 1 CONCEPTS
   - Arrays + Trees: Tree representation in arrays (heaps)
   - Strings + Trees: Trie structures, parsing problems
   - Hash Tables + Trees: Parent tracking, memoization
   - Linked Lists + Trees: Tree flattening, reconstruction
   - Stacks/Queues + Trees: Iterative traversals, level processing

=============================================================================
"""

from collections import deque, defaultdict, Counter
import heapq
from typing import List, Optional
import math


# TreeNode class for consistency
class TreeNode:
    """Standard binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# =============================================================================
# INTEGRATION PROBLEM 1: BINARY TREE TO BST CONVERSION (MEDIUM) - 45 MIN
# =============================================================================

def binary_tree_to_bst(root):
    """
    PROBLEM: Convert Binary Tree to Binary Search Tree
    
    Given the root of a binary tree, convert it to a Binary Search Tree while preserving the structure.
    The conversion should be done such that the resulting BST has the same structure as the original 
    binary tree, but the values are rearranged to satisfy the BST property.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 10^4]
    - -10^4 <= Node.val <= 10^4
    - The tree structure must be preserved (same shape)
    
    EXAMPLES:
    Example 1:
        Input: root = [10,2,7,8,4]
        Original tree:     BST result:
            10                 8
           /  \               / \
          2    7             4   10
         / \                /   /
        8   4              2   7
        Output: [8,4,10,2,null,7,null]
        
    Example 2:
        Input: root = [5,3,8,2,4,7,9]
        Output: [5,3,8,2,4,7,9] (already a BST)
        
    Example 3:
        Input: root = [1]
        Output: [1]
    
    APPROACH: Inorder Extraction and Filling
    
    The key insight is that an inorder traversal of a BST gives values in sorted order.
    
    1. Extract all values using inorder traversal
    2. Sort the extracted values
    3. Fill the tree nodes with sorted values using inorder traversal
    
    This preserves the tree structure while ensuring BST property.
    
    TIME: O(n log n) - sorting dominates, SPACE: O(n) - storing values and recursion stack
    """
    if not root:
        return None
    
    # Step 1: Extract all values using inorder traversal
    def inorder_extract(node, values):
        if not node:
            return
        inorder_extract(node.left, values)
        values.append(node.val)
        inorder_extract(node.right, values)
    
    values = []
    inorder_extract(root, values)
    values.sort()  # Create sorted sequence for BST
    
    # Step 2: Fill tree nodes with sorted values using inorder
    def inorder_fill(node):
        nonlocal index
        if not node:
            return
        
        inorder_fill(node.left)
        node.val = values[index]
        index += 1
        inorder_fill(node.right)
    
    index = 0
    inorder_fill(root)
    return root


# =============================================================================
# INTEGRATION PROBLEM 2: PATH SUM WITH TARGET RANGE (MEDIUM) - 45 MIN
# =============================================================================

def path_sum_with_target_range(root, target_min, target_max):
    """
    PROBLEM: Path Sum with Target Range
    
    Given the root of a binary tree and two integers target_min and target_max, find all root-to-leaf 
    paths where the sum of the node values in the path is within the range [target_min, target_max] inclusive.
    
    A leaf is a node with no children. A root-to-leaf path is a path starting from the root and ending at any leaf.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 5000]
    - -1000 <= Node.val <= 1000
    - -1000 <= target_min <= target_max <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], target_min = 22, target_max = 22
        Output: [[5,4,11,2],[5,8,4,5]]
        Explanation: Both paths sum to 22, which is within [22,22]
        
    Example 2:
        Input: root = [1,2,3], target_min = 4, target_max = 6
        Output: [[1,2],[1,3]]
        Explanation: Path [1,2] sums to 3, path [1,3] sums to 4, both outside range
        Output: []
        
    Example 3:
        Input: root = [1,2], target_min = 1, target_max = 3
        Output: [[1,2]]
        Explanation: Path [1,2] sums to 3, which is within [1,3]
    
    APPROACH: DFS with Path Tracking and Range Checking
    
    Use depth-first search to explore all root-to-leaf paths.
    Maintain current path and running sum, check range when reaching leaf nodes.
    
    Combines concepts:
    - Tree path traversal (DFS)
    - Range checking (binary search thinking)
    - Backtracking for path exploration
    
    TIME: O(n * h) where h is height for path copying, SPACE: O(h) for recursion stack
    """
    def dfs(node, current_path, current_sum, all_paths):
        if not node:
            return
        
        current_path.append(node.val)
        current_sum += node.val
        
        # Check if leaf node with valid sum
        if not node.left and not node.right:
            if target_min <= current_sum <= target_max:
                all_paths.append(current_path[:])
        
        # Continue DFS
        dfs(node.left, current_path, current_sum, all_paths)
        dfs(node.right, current_path, current_sum, all_paths)
        
        # Backtrack
        current_path.pop()
    
    result = []
    dfs(root, [], 0, result)
    return result


# =============================================================================
# INTEGRATION PROBLEM 3: TREE SERIALIZATION WITH PRIORITY (HARD) - 60 MIN
# =============================================================================

def serialize_tree_by_levels_priority(root):
    """
    PROBLEM: Tree Serialization with Priority
    
    Serialize a binary tree using level-order traversal, but within each level, process nodes 
    in order of their values (smallest first). This creates a unique serialization that combines 
    level-order structure with value-based priority.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4]
    - -1000 <= Node.val <= 1000
    - All node values are unique
    
    EXAMPLES:
    Example 1:
        Input: root = [3,9,20,null,null,15,7]
        Regular level-order: [3,9,20,15,7]
        Priority level-order: [3,9,20,7,15] (within level {15,7}, process 7 first)
        Output: "3,9,20,7,15"
        
    Example 2:
        Input: root = [1,3,2,5,3,null,9]
        Output: "1,2,3,3,5,9" (process nodes by value within each level)
        
    Example 3:
        Input: root = []
        Output: ""
    
    APPROACH: BFS with Priority Queue per Level
    
    Combines multiple concepts:
    - Level-order traversal (BFS)
    - Priority queue (heap) for ordering within levels
    - Tree serialization techniques
    
    For each level:
    1. Use heap to process nodes by value priority
    2. Collect children for next level
    3. Continue until all levels processed
    
    TIME: O(n log w) where w is maximum width, SPACE: O(w) for level storage
    """
    if not root:
        return ""
    
    result = []
    
    # Use heap to process nodes by value priority within each level
    current_level = [(root.val, 0, root)]  # (value, id, node)
    heapq.heapify(current_level)
    node_id = 1
    
    while current_level:
        next_level = []
        level_values = []
        
        while current_level:
            val, nid, node = heapq.heappop(current_level)
            level_values.append(val)
            
            # Add children for next level
            if node.left:
                heapq.heappush(next_level, (node.left.val, node_id, node.left))
                node_id += 1
            if node.right:
                heapq.heappush(next_level, (node.right.val, node_id, node.right))
                node_id += 1
        
        result.extend(level_values)
        current_level = next_level
    
    return ','.join(map(str, result))


# =============================================================================
# INTEGRATION PROBLEM 4: BALANCED BST FROM STREAM (HARD) - 60 MIN
# =============================================================================

class BalancedBSTFromStream:
    """
    PROBLEM: Build Balanced BST from Streaming Sorted Data
    
    Design a data structure that can efficiently build and maintain a balanced BST from streaming 
    sorted data. The stream provides sorted integers, and you need to maintain a balanced BST 
    that supports search operations.
    
    Implement the BalancedBSTFromStream class:
    - BalancedBSTFromStream(buffer_size): Initialize with given buffer size
    - add_value(val): Add a value from the stream
    - search(val): Return true if val exists in the BST
    
    CONSTRAINTS:
    - 1 <= buffer_size <= 1000
    - -10^6 <= val <= 10^6
    - Values in stream are in non-decreasing order
    - At most 10^4 calls to add_value and search
    
    EXAMPLES:
    Example 1:
        Input: ["BalancedBSTFromStream", "add_value", "add_value", "search", "add_value", "search"]
               [[3], [1], [2], [1], [3], [2]]
        Output: [null, null, null, true, null, true]
        
    Example 2:
        Operations: add values 1,2,3,4,5 with buffer_size=2
        BST rebuilds: after [1,2], after [3,4], after [5] (final)
        All values searchable in balanced BST
    
    APPROACH: Buffered Reconstruction with Heap
    
    Combines multiple advanced concepts:
    - Streaming data processing
    - BST construction and balancing
    - Heap-based buffering
    - Amortized complexity analysis
    
    Strategy:
    1. Buffer incoming values in a heap
    2. When buffer is full, rebuild entire BST
    3. Extract existing values + buffer values, sort, build balanced BST
    4. This gives amortized good performance
    
    TIME: O(log n) for search, O(n log n) amortized for add_value, SPACE: O(n)
    """
    
    def __init__(self, buffer_size=100):
        self.buffer = []
        self.buffer_size = buffer_size
        self.root = None
    
    def add_value(self, val):
        """
        Add value to stream and maintain balanced BST
        
        APPROACH: Buffered Rebuilding
        
        Buffer values and periodically rebuild BST for optimal balance.
        """
        heapq.heappush(self.buffer, val)
        
        # Rebuild BST when buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._rebuild_bst()
    
    def _rebuild_bst(self):
        """Rebuild balanced BST from existing tree + buffer"""
        # Extract all values from current BST
        all_values = []
        self._inorder_extract(self.root, all_values)
        
        # Add buffered values
        while self.buffer:
            all_values.append(heapq.heappop(self.buffer))
        
        # Remove duplicates and sort
        all_values = sorted(set(all_values))
        
        # Build balanced BST
        self.root = self._build_balanced_bst(all_values, 0, len(all_values) - 1)
    
    def _inorder_extract(self, node, values):
        """Extract values from BST using inorder traversal"""
        if not node:
            return
        self._inorder_extract(node.left, values)
        values.append(node.val)
        self._inorder_extract(node.right, values)
    
    def _build_balanced_bst(self, values, left, right):
        """Build balanced BST from sorted array"""
        if left > right:
            return None
        
        mid = (left + right) // 2
        node = TreeNode(values[mid])
        node.left = self._build_balanced_bst(values, left, mid - 1)
        node.right = self._build_balanced_bst(values, mid + 1, right)
        return node
    
    def search(self, val):
        """Search for value in BST"""
        return self._search_bst(self.root, val) or val in self.buffer
    
    def _search_bst(self, node, val):
        """Standard BST search"""
        if not node:
            return False
        if node.val == val:
            return True
        elif val < node.val:
            return self._search_bst(node.left, val)
        else:
            return self._search_bst(node.right, val)


# =============================================================================
# INTEGRATION PROBLEM 5: EXPRESSION TREE EVALUATOR (HARD) - 60 MIN
# =============================================================================

class ExpressionTree:
    """
    PROBLEM: Expression Tree Construction and Evaluation
    
    Given a mathematical expression as a string, build an expression tree and evaluate it.
    The expression contains integers, +, -, *, /, and parentheses. Support the standard 
    order of operations.
    
    Implement the ExpressionTree class:
    - ExpressionTree(expression): Build tree from infix expression
    - evaluate(): Return the result of evaluating the expression
    - get_infix(): Return the infix representation with minimal parentheses
    
    CONSTRAINTS:
    - 1 <= expression.length <= 1000
    - expression consists of digits, '+', '-', '*', '/', '(', ')', and spaces
    - The expression is guaranteed to be valid
    - No division by zero will occur
    
    EXAMPLES:
    Example 1:
        Input: expression = "3 + 2 * 2"
        Tree structure:
            +
           / \
          3   *
             / \
            2   2
        evaluate(): 7
        get_infix(): "3 + 2 * 2"
        
    Example 2:
        Input: expression = "(1 + 2) * 3"
        evaluate(): 9
        get_infix(): "(1 + 2) * 3"
    
    APPROACH: Shunting Yard Algorithm + Tree Construction
    
    Combines multiple concepts:
    - Expression parsing and tokenization
    - Stack-based algorithms (Shunting Yard)
    - Binary tree construction
    - Tree traversal for evaluation
    - Operator precedence handling
    
    Steps:
    1. Convert infix to postfix using Shunting Yard algorithm
    2. Build expression tree from postfix notation
    3. Evaluate tree using post-order traversal
    
    TIME: O(n) for construction and evaluation, SPACE: O(n) for tree storage
    """
    
    def __init__(self, expression):
        self.expression = expression.replace(' ', '')
        postfix = self._to_postfix(self.expression)
        self.root = self._build_from_postfix(postfix)
    
    def _to_postfix(self, expression):
        """Convert infix expression to postfix using Shunting Yard algorithm"""
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        stack = []
        postfix = []
        i = 0
        
        while i < len(expression):
            char = expression[i]
            
            if char.isdigit():
                # Handle multi-digit numbers
                num = ''
                while i < len(expression) and expression[i].isdigit():
                    num += expression[i]
                    i += 1
                postfix.append(int(num))
                continue
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()  # Remove '('
            elif char in precedence:
                while (stack and stack[-1] != '(' and 
                       stack[-1] in precedence and
                       precedence[stack[-1]] >= precedence[char]):
                    postfix.append(stack.pop())
                stack.append(char)
            
            i += 1
        
        while stack:
            postfix.append(stack.pop())
        
        return postfix
    
    def _build_from_postfix(self, postfix):
        """Build expression tree from postfix notation"""
        stack = []
        
        for token in postfix:
            if isinstance(token, int):
                stack.append(TreeNode(token))
            else:
                # Operator: pop two operands
                right = stack.pop()
                left = stack.pop()
                
                node = TreeNode(token)
                node.left = left
                node.right = right
                stack.append(node)
        
        return stack[0] if stack else None
    
    def evaluate(self):
        """Evaluate the expression tree"""
        def eval_node(node):
            if not node:
                return 0
            
            # Leaf node (operand)
            if isinstance(node.val, int):
                return node.val
            
            # Internal node (operator)
            left_val = eval_node(node.left)
            right_val = eval_node(node.right)
            
            if node.val == '+':
                return left_val + right_val
            elif node.val == '-':
                return left_val - right_val
            elif node.val == '*':
                return left_val * right_val
            elif node.val == '/':
                return left_val // right_val  # Integer division
        
        return eval_node(self.root)
    
    def get_infix(self):
        """Get infix representation with minimal parentheses"""
        def infix_helper(node):
            if not node:
                return ""
            
            if isinstance(node.val, int):
                return str(node.val)
            
            left_expr = infix_helper(node.left)
            right_expr = infix_helper(node.right)
            
            # Add parentheses based on precedence
            precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
            
            if (node.left and isinstance(node.left.val, str) and 
                precedence.get(node.left.val, 0) < precedence.get(node.val, 0)):
                left_expr = f"({left_expr})"
            
            if (node.right and isinstance(node.right.val, str) and 
                precedence.get(node.right.val, 0) < precedence.get(node.val, 0)):
                right_expr = f"({right_expr})"
            
            return f"{left_expr} {node.val} {right_expr}"
        
        return infix_helper(self.root)


# =============================================================================
# INTEGRATION PROBLEM 6: MULTI-STRUCTURE ANALYZER (HARD) - 60 MIN
# =============================================================================

class MultiStructureAnalyzer:
    """
    PROBLEM: Multi-Structure Data Analyzer
    
    Given an array of integers, build multiple data structures (BST, heap, sorted array) and 
    provide comprehensive analysis including search performance, heap properties, and optimization suggestions.
    
    Implement the MultiStructureAnalyzer class:
    - MultiStructureAnalyzer(data): Initialize with integer array
    - binary_search_analysis(target): Analyze binary search performance
    - bst_analysis(target): Analyze BST search with path tracking
    - heap_analysis(): Analyze heap properties and operations
    - comprehensive_report(target): Generate complete analysis report
    
    CONSTRAINTS:
    - 1 <= data.length <= 1000
    - -1000 <= data[i] <= 1000
    - Data may contain duplicates
    
    EXAMPLES:
    Example 1:
        Input: data = [4, 2, 6, 1, 3, 5, 7], target = 5
        Output: Comprehensive analysis including:
        - Binary search: 2 comparisons on sorted array
        - BST search: Path [4, 6, 5], 3 comparisons
        - Heap: Valid min-heap property, min element 1
        - Recommendations: BST optimal for this target
    
    APPROACH: Multi-Structure Construction and Analysis
    
    Integrates all major data structures and algorithms:
    - Array sorting and binary search
    - BST construction and search analysis
    - Heap construction and property verification
    - Performance comparison and optimization analysis
    
    This problem demonstrates real-world data structure selection decisions.
    
    TIME: O(n log n) for construction, various for operations, SPACE: O(n) for each structure
    """
    
    def __init__(self, data):
        self.original_data = data[:]
        self.sorted_data = sorted(data)
        self.bst_root = self._build_bst()
        self.heap_data = data[:]
        heapq.heapify(self.heap_data)
    
    def _build_bst(self):
        """Build BST from original data order"""
        root = None
        
        def build(left, right):
            if left > right:
                return None
            
            mid = (left + right) // 2
            node = TreeNode(self.sorted_data[mid])
            node.left = build(left, mid - 1)
            node.right = build(mid + 1, right)
            return node
        
        return build(0, len(self.sorted_data) - 1)
    
    def _build_frequency_tree(self):
        """Build tree based on element frequencies (bonus integration)"""
        from collections import Counter
        freq_count = Counter(self.original_data)
        
        # Build tree where more frequent elements are closer to root
        freq_items = sorted(freq_count.items(), key=lambda x: -x[1])
        
        def build_freq_tree(items, start, end):
            if start > end:
                return None
            
            mid = (start + end) // 2
            val, freq = items[mid]
            node = TreeNode(val)
            node.left = build_freq_tree(items, start, mid - 1)
            node.right = build_freq_tree(items, mid + 1, end)
            return node
        
        return build_freq_tree(freq_items, 0, len(freq_items) - 1)
    
    def binary_search_analysis(self, target):
        """Analyze binary search performance on sorted array"""
        comparisons = 0
        left, right = 0, len(self.sorted_data) - 1
        path = []
        
        while left <= right:
            mid = (left + right) // 2
            comparisons += 1
            path.append(self.sorted_data[mid])
            
            if self.sorted_data[mid] == target:
                return {
                    'found': True,
                    'comparisons': comparisons,
                    'path': path,
                    'index': mid,
                    'complexity': 'O(log n)'
                }
            elif self.sorted_data[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return {
            'found': False,
            'comparisons': comparisons,
            'path': path,
            'complexity': 'O(log n)'
        }
    
    def bst_analysis(self, target):
        """Analyze BST search with detailed path tracking"""
        def search_with_path(node, target, path):
            if not node:
                return False, path, len(path)
            
            path.append(node.val)
            
            if node.val == target:
                return True, path, len(path)
            elif target < node.val:
                return search_with_path(node.left, target, path)
            else:
                return search_with_path(node.right, target, path)
        
        found, path, comparisons = search_with_path(self.bst_root, target, [])
        
        # Calculate tree height and balance factor
        def get_height(node):
            if not node:
                return 0
            return 1 + max(get_height(node.left), get_height(node.right))
        
        height = get_height(self.bst_root)
        optimal_height = math.ceil(math.log2(len(self.original_data) + 1))
        
        return {
            'found': found,
            'comparisons': comparisons,
            'path': path,
            'tree_height': height,
            'optimal_height': optimal_height,
            'balance_factor': height - optimal_height,
            'complexity': 'O(h) where h is height'
        }
    
    def heap_analysis(self):
        """Analyze heap properties and performance characteristics"""
        # Verify heap property
        is_valid_heap = self._verify_heap_property()
        
        # Get heap statistics
        heap_size = len(self.heap_data)
        min_element = self.heap_data[0] if self.heap_data else None
        
        # Simulate heap operations
        insert_complexity = math.ceil(math.log2(heap_size + 1)) if heap_size > 0 else 1
        extract_complexity = math.ceil(math.log2(heap_size)) if heap_size > 0 else 0
        
        return {
            'valid_heap': is_valid_heap,
            'size': heap_size,
            'min_element': min_element,
            'insert_complexity': f"O(log n) ≈ {insert_complexity} comparisons",
            'extract_min_complexity': f"O(log n) ≈ {extract_complexity} comparisons",
            'peek_complexity': 'O(1)',
            'heap_data': self.heap_data[:10]  # Show first 10 elements
        }
    
    def _verify_heap_property(self):
        """Verify min-heap property"""
        for i in range(len(self.heap_data)):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            if left_child < len(self.heap_data):
                if self.heap_data[i] > self.heap_data[left_child]:
                    return False
            
            if right_child < len(self.heap_data):
                if self.heap_data[i] > self.heap_data[right_child]:
                    return False
        
        return True
    
    def comprehensive_report(self, target):
        """Generate comprehensive analysis report"""
        binary_search_result = self.binary_search_analysis(target)
        bst_result = self.bst_analysis(target)
        heap_result = self.heap_analysis()
        
        report = {
            'target': target,
            'data_size': len(self.original_data),
            'binary_search': binary_search_result,
            'bst_search': bst_result,
            'heap_analysis': heap_result,
            'recommendations': []
        }
        
        # Generate recommendations
        if binary_search_result['found'] and bst_result['found']:
            if binary_search_result['comparisons'] < bst_result['comparisons']:
                report['recommendations'].append("Binary search on sorted array is more efficient for this target")
            elif bst_result['comparisons'] < binary_search_result['comparisons']:
                report['recommendations'].append("BST search is more efficient for this target")
            else:
                report['recommendations'].append("Both binary search and BST have similar performance")
        
        if bst_result['balance_factor'] > 2:
            report['recommendations'].append("BST is unbalanced; consider using AVL or Red-Black tree")
        
        if target == heap_result['min_element']:
            report['recommendations'].append("For minimum element access, heap provides O(1) performance")
        
        return report


# COMPREHENSIVE TESTING SUITE
def test_integration_problems():
    """
    Test all integration problems to verify Week 2 mastery
    """
    print("=== WEEK 2 INTEGRATION TESTING ===\n")
    
    # Test 1: Binary Tree to BST
    print("1. Binary Tree to BST Conversion:")
    
    # Create unordered binary tree
    tree = TreeNode(1)
    tree.left = TreeNode(3)
    tree.right = TreeNode(2)
    tree.left.left = TreeNode(5)
    tree.left.right = TreeNode(4)
    
    print("   Original tree (level order): [1, 3, 2, 5, 4]")
    
    # Convert to BST
    bst_tree = binary_tree_to_bst(tree)
    
    # Verify BST property by inorder traversal
    def inorder_traversal(node):
        if not node:
            return []
        return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)
    
    inorder_result = inorder_traversal(bst_tree)
    print(f"   BST inorder traversal: {inorder_result}")
    print(f"   ✓ Is sorted: {inorder_result == sorted(inorder_result)}")
    print()
    
    # Test 2: Expression Tree
    print("2. Expression Tree Evaluator:")
    expressions = ["3 + 4 * 2", "(1 + 2) * (3 + 4)", "2 ^ 3 + 1"]
    
    for expr in expressions:
        expr_tree = ExpressionTree(expr)
        result = expr_tree.evaluate()
        infix = expr_tree.get_infix()
        
        print(f"   Expression: {expr}")
        print(f"   Tree form: {infix}")
        print(f"   Result: {result}")
        print()
    
    # Test 3: Multi-structure Analyzer
    print("3. Multi-structure Data Analysis:")
    test_data = [4, 2, 7, 1, 3, 6, 8, 5]
    analyzer = MultiStructureAnalyzer(test_data)
    
    target = 6
    report = analyzer.comprehensive_report(target)
    
    print(f"   Data: {test_data}")
    print(f"   Analysis for target {target}:")
    print(f"   Data Summary: {report['data_summary']}")
    print(f"   Binary Search: {report['binary_search']}")
    print(f"   BST Search: {report['bst_search']}")
    print(f"   Heap Analysis: {report['heap_analysis']}")
    print(f"   Top Frequencies: {report['frequency_analysis']}")


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_algorithm_selection():
    """
    Show when to use different algorithms based on problem characteristics
    """
    print("\n=== ALGORITHM SELECTION GUIDE ===")
    
    problem_types = {
        "Search in sorted array": {
            "best": "Binary Search O(log n)",
            "alternatives": "Linear search O(n), Hash table O(1) with preprocessing",
            "when_to_use": "Data is sorted, need repeated searches"
        },
        "Find Kth largest/smallest": {
            "best": "Min/Max heap O(n log k) or Quickselect O(n) average",
            "alternatives": "Sort O(n log n), Min heap all elements O(n log n)",
            "when_to_use": "K is small relative to n, streaming data"
        },
        "Range sum queries": {
            "best": "BST O(log n) per query after O(n log n) build",
            "alternatives": "Prefix sums O(1) query, O(n) build",
            "when_to_use": "Many range queries, data updates needed"
        },
        "Merge sorted sequences": {
            "best": "Min heap O(n log k) for k sequences",
            "alternatives": "Merge pairs O(n k), Sort all O(n log n)",
            "when_to_use": "Many sequences, streaming merge"
        },
        "Tree path problems": {
            "best": "DFS with backtracking O(n)",
            "alternatives": "BFS with path storage O(n * h)",
            "when_to_use": "Need all paths, complex path conditions"
        }
    }
    
    for problem, info in problem_types.items():
        print(f"\n{problem}:")
        print(f"  Best approach: {info['best']}")
        print(f"  Alternatives: {info['alternatives']}")
        print(f"  When to use: {info['when_to_use']}")


def week2_complexity_summary():
    """
    Comprehensive complexity analysis for Week 2 algorithms
    """
    print("\n=== WEEK 2 COMPLEXITY SUMMARY ===")
    
    algorithms = [
        # Tree algorithms
        ("Tree Traversal (DFS/BFS)", "O(n)", "O(h) recursive, O(w) BFS"),
        ("BST Search/Insert/Delete", "O(h)", "O(h) recursion"),
        ("Tree Serialization", "O(n)", "O(n) storage"),
        ("Path Sum Problems", "O(n)", "O(h) recursion"),
        ("Tree Reconstruction", "O(n)", "O(n) storage"),
        
        # Binary search algorithms
        ("Binary Search", "O(log n)", "O(1) iterative"),
        ("Search Range", "O(log n)", "O(1)"),
        ("Rotated Array Search", "O(log n)", "O(1)"),
        ("2D Matrix Search", "O(log(mn))", "O(1)"),
        ("Value Binary Search", "O(log(range))", "O(1)"),
        
        # Heap algorithms  
        ("Heap Insert/Delete", "O(log n)", "O(1)"),
        ("Build Heap", "O(n)", "O(1)"),
        ("Kth Largest/Smallest", "O(n log k)", "O(k)"),
        ("Merge K Sorted", "O(n log k)", "O(k)"),
        ("Median from Stream", "O(log n) insert", "O(n) total")
    ]
    
    print("Algorithm                 | Time           | Space")
    print("--------------------------|----------------|------------------")
    for alg, time, space in algorithms:
        print(f"{alg:25} | {time:14} | {space}")
    
    print("\nKey Variables:")
    print("  n = number of elements")
    print("  h = height of tree (log n balanced, n skewed)")
    print("  k = parameter (k largest, k sequences, etc.)")
    print("  w = width of tree (maximum nodes at any level)")


def meta_interview_tips():
    """
    Specific tips for Meta interviews based on Week 2 content
    """
    print("\n=== META INTERVIEW TIPS - WEEK 2 ===")
    
    print("Common Meta Tree Problems:")
    print("  • Binary tree path sum variations")
    print("  • BST validation and modification")
    print("  • Tree serialization/deserialization")
    print("  • Lowest common ancestor variants")
    print("  • Tree view problems (vertical, boundary)")
    
    print("\nFrequent Binary Search Applications:")
    print("  • Search in rotated sorted arrays")
    print("  • Find peak element variations")
    print("  • Capacity/allocation optimization problems")
    print("  • Matrix search problems")
    print("  • Time-based problems (events, logs)")
    
    print("\nHeap/Priority Queue Scenarios:")
    print("  • Top K problems in various contexts")
    print("  • Streaming data analysis")
    print("  • Task scheduling simulations")
    print("  • Multi-way merge operations")
    print("  • Real-time median tracking")
    
    print("\nOptimization Focus Areas:")
    print("  • Space optimization: Iterative vs recursive")
    print("  • Early termination conditions")
    print("  • Multiple solution approaches comparison")
    print("  • Scalability considerations")
    print("  • Edge case handling")
    
    print("\nCommunication Tips:")
    print("  • Always clarify tree structure assumptions")
    print("  • Explain choice between DFS vs BFS")
    print("  • Discuss balanced vs unbalanced tree performance")
    print("  • Mention space-time tradeoffs explicitly")
    print("  • Consider follow-up optimizations")


if __name__ == "__main__":
    # Run integration tests
    test_integration_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Algorithm selection guide
    demonstrate_algorithm_selection()
    
    # Complexity summary
    week2_complexity_summary()
    
    # Meta interview tips
    meta_interview_tips()
    
    print("\n" + "="*70)
    print("WEEK 2 COMPLETE - COMPREHENSIVE REVIEW:")
    print("="*70)
    print("✓ Binary Trees: Foundation traversals and basic operations")
    print("✓ BST: Ordered tree operations and validation")
    print("✓ Binary Search: Efficient search in sorted/monotonic spaces")
    print("✓ Advanced Trees: Complex modifications and path problems")
    print("✓ Heaps: Priority queues and streaming data handling")
    print("✓ Integration: Combining multiple concepts effectively")
    
    print("\nKey Mastery Areas:")
    print("• Tree traversal selection (DFS vs BFS)")
    print("• BST property maintenance and validation")
    print("• Binary search template and variations")
    print("• Path problem solving strategies")
    print("• Heap operations and two-heap techniques")
    print("• Algorithm complexity analysis")
    print("• Problem pattern recognition")
    
    print("\nTransition to Week 3:")
    print("• Graph algorithms build on tree foundations")
    print("• BFS/DFS extend to graph traversals")
    print("• Advanced data structures for complex problems")
    print("• System design considerations for scalability")
    
    print("\n🎯 Week 2 Assessment: Successfully completed 48+ problems")
    print("🎯 Ready for Week 3: Graphs & Advanced Structures")
    print("🎯 Meta Interview Confidence: Trees & Search Algorithms ✓") 