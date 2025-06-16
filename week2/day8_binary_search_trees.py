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


# Problem 1: Validate Binary Search Tree - Core BST concept
def is_valid_bst_wrong(root):
    """
    INCORRECT approach: Only checks immediate children
    
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
    CORRECT approach: Maintain valid range for each node
    
    Each node must be within (min_val, max_val) range
    
    Time: O(n), Space: O(h)
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
    Alternative approach: In-order traversal should be sorted
    
    Time: O(n), Space: O(h)
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
    Optimized in-order: Track previous value without storing all values
    
    Time: O(n), Space: O(h)
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


# Problem 2: Search in BST - Basic BST operation
def search_bst_recursive(root, val):
    """
    Search for value in BST using recursion
    
    Navigate left/right based on comparison
    
    Time: O(h), Space: O(h)
    """
    if not root or root.val == val:
        return root
    
    if val < root.val:
        return search_bst_recursive(root.left, val)
    else:
        return search_bst_recursive(root.right, val)


def search_bst_iterative(root, val):
    """
    Search for value in BST iteratively
    
    More memory efficient than recursive
    
    Time: O(h), Space: O(1)
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


# Problem 3: Insert into BST - BST modification
def insert_into_bst_recursive(root, val):
    """
    Insert value into BST maintaining BST property
    
    Find correct position and insert as leaf
    
    Time: O(h), Space: O(h)
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
    Insert value into BST iteratively
    
    Time: O(h), Space: O(1)
    """
    if not root:
        return TreeNode(val)
    
    current = root
    
    while True:
        if val < current.val:
            if not current.left:
                current.left = TreeNode(val)
                break
            current = current.left
        else:
            if not current.right:
                current.right = TreeNode(val)
                break
            current = current.right
    
    return root


# Problem 4: Delete Node in BST - Complex BST operation
def delete_node_bst(root, key):
    """
    Delete node from BST maintaining BST property
    
    Three cases:
    1. Leaf node: Simply remove
    2. One child: Replace with child
    3. Two children: Replace with successor (or predecessor)
    
    Time: O(h), Space: O(h)
    """
    if not root:
        return root
    
    if key < root.val:
        root.left = delete_node_bst(root.left, key)
    elif key > root.val:
        root.right = delete_node_bst(root.right, key)
    else:
        # Found node to delete
        
        # Case 1: Leaf node or one child
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        
        # Case 2: Two children
        # Find successor (minimum in right subtree)
        successor = find_min(root.right)
        root.val = successor.val
        root.right = delete_node_bst(root.right, successor.val)
    
    return root


def find_min(root):
    """Find minimum value node in BST (leftmost)"""
    while root.left:
        root = root.left
    return root


def find_max(root):
    """Find maximum value node in BST (rightmost)"""
    while root.right:
        root = root.right
    return root


# Problem 5: Kth Smallest Element in BST - In-order traversal application
def kth_smallest_recursive(root, k):
    """
    Find kth smallest element using in-order traversal
    
    In-order gives sorted sequence, so kth element is answer
    
    Time: O(n), Space: O(h)
    """
    def inorder(node):
        if not node:
            return []
        
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    values = inorder(root)
    return values[k-1] if k <= len(values) else None


def kth_smallest_optimized(root, k):
    """
    Optimized: Stop as soon as we find kth element
    
    Time: O(h + k), Space: O(h)
    """
    def inorder(node):
        nonlocal count, result
        if not node or count >= k:
            return
        
        # Traverse left
        inorder(node.left)
        
        # Process current node
        count += 1
        if count == k:
            result = node.val
            return
        
        # Traverse right
        inorder(node.right)
    
    count = 0
    result = None
    inorder(root)
    return result


def kth_smallest_iterative(root, k):
    """
    Iterative in-order traversal with early termination
    
    Time: O(h + k), Space: O(h)
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


# Problem 6: Lowest Common Ancestor in BST - BST-specific optimization
def lowest_common_ancestor_bst(root, p, q):
    """
    Find LCA in BST using BST property
    
    More efficient than general tree LCA due to ordering
    
    Time: O(h), Space: O(1) iterative, O(h) recursive
    """
    # Ensure p.val <= q.val for simplicity
    if p.val > q.val:
        p, q = q, p
    
    while root:
        if q.val < root.val:
            # Both in left subtree
            root = root.left
        elif p.val > root.val:
            # Both in right subtree
            root = root.right
        else:
            # Found LCA (split point)
            return root
    
    return None


def lowest_common_ancestor_bst_recursive(root, p, q):
    """
    Recursive version of BST LCA
    
    Time: O(h), Space: O(h)
    """
    if not root:
        return None
    
    if p.val < root.val and q.val < root.val:
        return lowest_common_ancestor_bst_recursive(root.left, p, q)
    elif p.val > root.val and q.val > root.val:
        return lowest_common_ancestor_bst_recursive(root.right, p, q)
    else:
        return root


# Problem 7: Convert Sorted Array to BST - BST construction
def sorted_array_to_bst(nums):
    """
    Convert sorted array to height-balanced BST
    
    Use middle element as root to maintain balance
    
    Time: O(n), Space: O(log n)
    """
    def build_bst(left, right):
        if left > right:
            return None
        
        # Choose middle as root
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        
        # Recursively build left and right subtrees
        root.left = build_bst(left, mid - 1)
        root.right = build_bst(mid + 1, right)
        
        return root
    
    return build_bst(0, len(nums) - 1)


# ADVANCED PROBLEMS FOR EXTRA PRACTICE

def recover_bst(root):
    """
    Recover BST where exactly two nodes are swapped
    
    Find the two swapped nodes and fix them
    
    Time: O(n), Space: O(h)
    """
    def inorder(node):
        nonlocal first, second, prev
        if not node:
            return
        
        inorder(node.left)
        
        # Find anomaly
        if prev and prev.val > node.val:
            if not first:
                first = prev
            second = node
        
        prev = node
        inorder(node.right)
    
    first = second = prev = None
    inorder(root)
    
    # Swap the values
    if first and second:
        first.val, second.val = second.val, first.val


def bst_to_greater_tree(root):
    """
    Convert BST to Greater Sum Tree
    
    Each node's value = original value + sum of all greater values
    
    Time: O(n), Space: O(h)
    """
    def reverse_inorder(node):
        nonlocal total
        if not node:
            return
        
        # Traverse right first (larger values)
        reverse_inorder(node.right)
        
        # Update current node
        total += node.val
        node.val = total
        
        # Traverse left
        reverse_inorder(node.left)
    
    total = 0
    reverse_inorder(root)
    return root


def range_sum_bst(root, low, high):
    """
    Calculate sum of all node values in given range [low, high]
    
    Use BST property to prune unnecessary branches
    
    Time: O(n), Space: O(h)
    """
    if not root:
        return 0
    
    if root.val < low:
        # Current and left subtree too small
        return range_sum_bst(root.right, low, high)
    elif root.val > high:
        # Current and right subtree too large
        return range_sum_bst(root.left, low, high)
    else:
        # Current node in range
        return (root.val + 
                range_sum_bst(root.left, low, high) +
                range_sum_bst(root.right, low, high))


def trim_bst(root, low, high):
    """
    Trim BST to contain only nodes in range [low, high]
    
    Time: O(n), Space: O(h)
    """
    if not root:
        return None
    
    if root.val < low:
        # Root too small, trim left subtree
        return trim_bst(root.right, low, high)
    elif root.val > high:
        # Root too large, trim right subtree
        return trim_bst(root.left, low, high)
    else:
        # Root in range, trim both subtrees
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