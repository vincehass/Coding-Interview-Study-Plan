"""
=============================================================================
                        WEEK 2 SINGLE PROBLEM PRACTICE
                        BINARY TREE INORDER TRAVERSAL
                           Meta Interview Preparation
=============================================================================

Focus on mastering one core tree problem with comprehensive testing.
This represents the most fundamental tree traversal pattern in interviews.

=============================================================================
"""

from typing import List, Optional


class TreeNode:
    """Definition for a binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    PROBLEM: Binary Tree Inorder Traversal
    
    DESCRIPTION:
    Given the root of a binary tree, return the inorder traversal of its nodes' values.
    
    Inorder traversal visits nodes in this order: Left -> Root -> Right
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 100].
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [1,null,2,3]
              1
               \
                2
               /
              3
        Output: [1,3,2]
    
    Example 2:
        Input: root = []
        Output: []
    
    Example 3:
        Input: root = [1]
        Output: [1]
    
    FOLLOW-UP: 
    Recursive solution is trivial, could you do it iteratively?
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h) where h is height of tree
    
    Args:
        root (Optional[TreeNode]): Root of the binary tree
        
    Returns:
        List[int]: Inorder traversal of the tree
    """
    # Recursive solution
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result


def create_tree_from_list(values: List[Optional[int]]) -> Optional[TreeNode]:
    """Helper function to create binary tree from level-order list"""
    if not values or values[0] is None:
        return None
    
    from collections import deque
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


def tree_to_string(root: Optional[TreeNode]) -> str:
    """Helper function to visualize tree structure"""
    if not root:
        return "Empty Tree"
    
    def build_string(node, level=0, prefix="Root: "):
        if not node:
            return ""
        
        result = "  " * level + prefix + str(node.val) + "\n"
        
        if node.left or node.right:
            if node.left:
                result += build_string(node.left, level + 1, "L--- ")
            else:
                result += "  " * (level + 1) + "L--- None\n"
                
            if node.right:
                result += build_string(node.right, level + 1, "R--- ")
            else:
                result += "  " * (level + 1) + "R--- None\n"
        
        return result
    
    return build_string(root)


def main():
    """Test the inorder_traversal function with various test cases"""
    
    print("=" * 70)
    print("           WEEK 2 SINGLE PROBLEM PRACTICE")
    print("            BINARY TREE INORDER TRAVERSAL")
    print("=" * 70)
    
    # Test Case 1: Example from problem description
    print("\n🧪 Test Case 1: Basic Example [1,null,2,3]")
    tree1_list = [1, None, 2, 3]
    tree1 = create_tree_from_list(tree1_list)
    expected1 = [1, 3, 2]
    result1 = inorder_traversal(tree1)
    
    print(f"Tree structure:")
    print(tree_to_string(tree1))
    print(f"Input: {tree1_list}")
    print(f"Expected: {expected1}")
    print(f"Got: {result1}")
    print(f"✅ PASS" if result1 == expected1 else f"❌ FAIL")
    
    # Test Case 2: Empty tree
    print("\n🧪 Test Case 2: Empty Tree")
    tree2_list = []
    tree2 = create_tree_from_list(tree2_list)
    expected2 = []
    result2 = inorder_traversal(tree2)
    
    print(f"Tree structure: Empty")
    print(f"Input: {tree2_list}")
    print(f"Expected: {expected2}")
    print(f"Got: {result2}")
    print(f"✅ PASS" if result2 == expected2 else f"❌ FAIL")
    
    # Test Case 3: Single node
    print("\n🧪 Test Case 3: Single Node")
    tree3_list = [1]
    tree3 = create_tree_from_list(tree3_list)
    expected3 = [1]
    result3 = inorder_traversal(tree3)
    
    print(f"Tree structure:")
    print(tree_to_string(tree3))
    print(f"Input: {tree3_list}")
    print(f"Expected: {expected3}")
    print(f"Got: {result3}")
    print(f"✅ PASS" if result3 == expected3 else f"❌ FAIL")
    
    # Test Case 4: Complete binary tree
    print("\n🧪 Test Case 4: Complete Binary Tree [1,2,3,4,5,6,7]")
    tree4_list = [1, 2, 3, 4, 5, 6, 7]
    tree4 = create_tree_from_list(tree4_list)
    expected4 = [4, 2, 5, 1, 6, 3, 7]  # Left->Root->Right
    result4 = inorder_traversal(tree4)
    
    print(f"Tree structure:")
    print(tree_to_string(tree4))
    print(f"Input: {tree4_list}")
    print(f"Expected: {expected4}")
    print(f"Got: {result4}")
    print(f"✅ PASS" if result4 == expected4 else f"❌ FAIL")
    
    # Test Case 5: Left skewed tree
    print("\n🧪 Test Case 5: Left Skewed Tree")
    tree5_list = [3, 2, None, 1]
    tree5 = create_tree_from_list(tree5_list)
    expected5 = [1, 2, 3]
    result5 = inorder_traversal(tree5)
    
    print(f"Tree structure:")
    print(tree_to_string(tree5))
    print(f"Input: {tree5_list}")
    print(f"Expected: {expected5}")
    print(f"Got: {result5}")
    print(f"✅ PASS" if result5 == expected5 else f"❌ FAIL")
    
    # Test Case 6: Right skewed tree
    print("\n🧪 Test Case 6: Right Skewed Tree")
    tree6_list = [1, None, 2, None, 3]
    tree6 = create_tree_from_list(tree6_list)
    expected6 = [1, 2, 3]
    result6 = inorder_traversal(tree6)
    
    print(f"Tree structure:")
    print(tree_to_string(tree6))
    print(f"Input: {tree6_list}")
    print(f"Expected: {expected6}")
    print(f"Got: {result6}")
    print(f"✅ PASS" if result6 == expected6 else f"❌ FAIL")
    
    # Test Case 7: Negative values
    print("\n🧪 Test Case 7: Negative Values")
    tree7_list = [0, -3, 9, -10, None, 5]
    tree7 = create_tree_from_list(tree7_list)
    expected7 = [-10, -3, 0, 5, 9]
    result7 = inorder_traversal(tree7)
    
    print(f"Tree structure:")
    print(tree_to_string(tree7))
    print(f"Input: {tree7_list}")
    print(f"Expected: {expected7}")
    print(f"Got: {result7}")
    print(f"✅ PASS" if result7 == expected7 else f"❌ FAIL")
    
    print("\n" + "=" * 70)
    print("                  TEST SUMMARY")
    print("=" * 70)
    
    # Count passes
    test_results = [
        result1 == expected1,
        result2 == expected2,
        result3 == expected3,
        result4 == expected4,
        result5 == expected5,
        result6 == expected6,
        result7 == expected7
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Great job!")
    else:
        print("❌ Some tests failed. Review your solution.")
    
    print("\n💡 SOLUTION APPROACHES:")
    print("1. RECURSIVE: Use helper function for left->root->right traversal")
    print("2. ITERATIVE: Use stack to simulate recursion")
    print("3. MORRIS: Use threading for O(1) space complexity")
    
    print("\n📚 LEARNING OBJECTIVES:")
    print("- Master recursive tree traversal patterns")
    print("- Understand the relationship between recursion and stack")
    print("- Practice with different tree structures and edge cases")
    print("- Learn iterative alternatives to recursive solutions")


# Reference solutions (uncomment to check your work)
def inorder_traversal_recursive(root: Optional[TreeNode]) -> List[int]:
    """
    Reference recursive solution
    Time: O(n), Space: O(h)
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
    Reference iterative solution using stack
    Time: O(n), Space: O(h)
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


def inorder_traversal_morris(root: Optional[TreeNode]) -> List[int]:
    """
    Reference Morris traversal solution
    Time: O(n), Space: O(1)
    """
    result = []
    current = root
    
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # Create thread
                predecessor.right = current
                current = current.left
            else:
                # Remove thread
                predecessor.right = None
                result.append(current.val)
                current = current.right
    
    return result


if __name__ == "__main__":
    main() 