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


# TreeNode class for consistency
class TreeNode:
    """Standard binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# INTEGRATION PROBLEM 1: Binary Tree to BST Conversion
def binary_tree_to_bst(root):
    """
    Convert binary tree to BST while preserving structure
    
    Combines: Tree traversal + BST properties + Tree modification
    
    Time: O(n log n), Space: O(n)
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


# INTEGRATION PROBLEM 2: Path Sum with Binary Search
def path_sum_with_target_range(root, target_min, target_max):
    """
    Find all root-to-leaf paths with sum in [target_min, target_max]
    
    Combines: Tree paths + range checking + binary search thinking
    
    Time: O(n * h), Space: O(h)
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


# INTEGRATION PROBLEM 3: Tree Serialization with Heap
def serialize_tree_by_levels_priority(root):
    """
    Serialize tree using level-order but prioritizing smaller values
    
    Combines: Tree serialization + heap operations + BFS
    
    Time: O(n log n), Space: O(n)
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


# INTEGRATION PROBLEM 4: Balanced BST from Sorted Stream
class BalancedBSTFromStream:
    """
    Build balanced BST from streaming sorted data using heap buffer
    
    Combines: BST construction + heap buffering + streaming data
    """
    
    def __init__(self, buffer_size=100):
        self.buffer = []
        self.buffer_size = buffer_size
        self.root = None
    
    def add_value(self, val):
        """Add value to stream and maintain balanced BST"""
        heapq.heappush(self.buffer, val)
        
        # Rebuild BST when buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._rebuild_bst()
    
    def _rebuild_bst(self):
        """Rebuild balanced BST from current buffer"""
        # Extract all values and build balanced BST
        all_values = []
        
        # Get existing tree values
        if self.root:
            self._inorder_extract(self.root, all_values)
        
        # Add buffer values
        while self.buffer:
            all_values.append(heapq.heappop(self.buffer))
        
        # Build new balanced BST
        all_values.sort()
        self.root = self._build_balanced_bst(all_values, 0, len(all_values) - 1)
    
    def _inorder_extract(self, node, values):
        """Extract values from existing BST"""
        if node:
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
        """Search in current BST"""
        return self._search_bst(self.root, val)
    
    def _search_bst(self, node, val):
        """Binary search in BST"""
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_bst(node.left, val)
        else:
            return self._search_bst(node.right, val)


# INTEGRATION PROBLEM 5: Tree-based Expression Evaluator
class ExpressionTree:
    """
    Evaluate mathematical expressions using tree structure
    
    Combines: Tree traversal + parsing + stack operations
    """
    
    def __init__(self, expression):
        self.root = self._build_from_postfix(self._to_postfix(expression))
    
    def _to_postfix(self, expression):
        """Convert infix to postfix notation"""
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        stack = []
        postfix = []
        
        for char in expression.replace(' ', ''):
            if char.isdigit():
                postfix.append(char)
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
        
        while stack:
            postfix.append(stack.pop())
        
        return postfix
    
    def _build_from_postfix(self, postfix):
        """Build expression tree from postfix notation"""
        stack = []
        
        for token in postfix:
            if token.isdigit():
                stack.append(TreeNode(int(token)))
            else:
                # Operator node
                right = stack.pop()
                left = stack.pop()
                node = TreeNode(token)
                node.left = left
                node.right = right
                stack.append(node)
        
        return stack[0] if stack else None
    
    def evaluate(self):
        """Evaluate expression tree"""
        def eval_node(node):
            if not node:
                return 0
            
            # Leaf node (operand)
            if isinstance(node.val, int):
                return node.val
            
            # Operator node
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
            elif node.val == '^':
                return left_val ** right_val
        
        return eval_node(self.root)
    
    def get_infix(self):
        """Convert back to infix notation"""
        def infix_helper(node):
            if not node:
                return ""
            
            if isinstance(node.val, int):
                return str(node.val)
            
            left_expr = infix_helper(node.left)
            right_expr = infix_helper(node.right)
            return f"({left_expr} {node.val} {right_expr})"
        
        return infix_helper(self.root)


# ASSESSMENT PROBLEM 6: Multi-structure Data Analyzer
class MultiStructureAnalyzer:
    """
    Analyze data using multiple data structures for comprehensive insights
    
    Combines: All Week 2 concepts in single problem
    """
    
    def __init__(self, data):
        self.data = data
        self.sorted_data = sorted(data)
        self.bst = self._build_bst()
        self.heap_data = data[:]
        heapq.heapify(self.heap_data)
        self.frequency_tree = self._build_frequency_tree()
    
    def _build_bst(self):
        """Build BST from sorted data"""
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
        """Build tree based on frequency analysis"""
        freq = Counter(self.data)
        
        # Create leaf nodes for each unique value
        heap = []
        for val, count in freq.items():
            heapq.heappush(heap, (count, TreeNode(val)))
        
        # Build Huffman-like tree
        while len(heap) > 1:
            freq1, node1 = heapq.heappop(heap)
            freq2, node2 = heapq.heappop(heap)
            
            merged = TreeNode(f"{node1.val},{node2.val}")
            merged.left = node1
            merged.right = node2
            
            heapq.heappush(heap, (freq1 + freq2, merged))
        
        return heap[0][1] if heap else None
    
    def binary_search_analysis(self, target):
        """Analyze target using binary search"""
        left, right = 0, len(self.sorted_data) - 1
        steps = 0
        
        while left <= right:
            steps += 1
            mid = (left + right) // 2
            
            if self.sorted_data[mid] == target:
                return {
                    'found': True,
                    'index': mid,
                    'steps': steps,
                    'efficiency': f"O(log n) - {steps} steps vs {len(self.data)} linear"
                }
            elif self.sorted_data[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return {'found': False, 'steps': steps}
    
    def bst_analysis(self, target):
        """Analyze using BST operations"""
        def search_with_path(node, target, path):
            if not node:
                return None, path
            
            path.append(node.val)
            
            if node.val == target:
                return node, path
            elif target < node.val:
                return search_with_path(node.left, target, path)
            else:
                return search_with_path(node.right, target, path)
        
        node, path = search_with_path(self.bst, target, [])
        
        return {
            'found': node is not None,
            'search_path': path,
            'comparisons': len(path),
            'tree_depth': len(path) if node else -1
        }
    
    def heap_analysis(self):
        """Analyze using heap operations"""
        heap_copy = self.heap_data[:]
        
        # Extract top 3 elements
        top_elements = []
        for _ in range(min(3, len(heap_copy))):
            if heap_copy:
                top_elements.append(heapq.heappop(heap_copy))
        
        return {
            'min_element': self.heap_data[0] if self.heap_data else None,
            'top_3_smallest': top_elements,
            'heap_size': len(self.heap_data),
            'heap_property_verified': self._verify_heap_property()
        }
    
    def _verify_heap_property(self):
        """Verify min heap property"""
        for i in range(len(self.heap_data)):
            left = 2 * i + 1
            right = 2 * i + 2
            
            if (left < len(self.heap_data) and 
                self.heap_data[i] > self.heap_data[left]):
                return False
            
            if (right < len(self.heap_data) and 
                self.heap_data[i] > self.heap_data[right]):
                return False
        
        return True
    
    def comprehensive_report(self, target):
        """Generate comprehensive analysis report"""
        return {
            'data_summary': {
                'size': len(self.data),
                'min': min(self.data),
                'max': max(self.data),
                'unique_count': len(set(self.data))
            },
            'binary_search': self.binary_search_analysis(target),
            'bst_search': self.bst_analysis(target),
            'heap_analysis': self.heap_analysis(),
            'frequency_analysis': Counter(self.data).most_common(3)
        }


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