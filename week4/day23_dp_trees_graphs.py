"""
=============================================================================
                        DAY 23: DP ON TREES AND GRAPHS
                           Meta Interview Preparation
                              Week 4 - Day 23
=============================================================================

FOCUS: Tree DP, graph DP patterns
TIME ALLOCATION: 4 hours
- Theory (1 hour): Tree DP patterns, graph state management
- Problems (3 hours): Complex tree and graph DP problems

TOPICS COVERED:
- Tree DP with subtree states
- Path problems on trees
- Graph DP with memoization
- State space reduction techniques

=============================================================================
"""

from typing import List, Dict, Optional, Tuple
from functools import lru_cache
from collections import defaultdict, deque


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
TREE AND GRAPH DP PATTERNS:

1. TREE DP:
   - Bottom-up: Process children before parent
   - Top-down: Process parent before children
   - Common states: max/min path, subtree properties
   - Often involves choosing to include/exclude nodes

2. GRAPH DP:
   - Use memoization due to cycles
   - State includes current node + additional info
   - Path-dependent vs path-independent problems
   - Bitmask for visited states in small graphs

3. COMMON PATTERNS:
   - Diameter problems: max path through any node
   - Path sum problems: include/exclude decisions
   - Subtree optimization: optimal choices in subtrees
   - State compression: reduce dimensions when possible
"""


# =============================================================================
# TREE NODE DEFINITION
# =============================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# =============================================================================
# PROBLEM 1: BINARY TREE MAXIMUM PATH SUM (HARD) - 60 MIN
# =============================================================================

def max_path_sum(root: Optional[TreeNode]) -> int:
    """
    PROBLEM: Binary Tree Maximum Path Sum
    
    A path in a binary tree is a sequence of nodes where each pair of adjacent 
    nodes in the sequence has an edge connecting them. A node can only appear 
    in the sequence at most once. The path does not need to pass through the root.
    
    The path sum of a path is the sum of the node's values in the path.
    Given the root of a binary tree, return the maximum path sum of any path.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 3 * 10^4]
    - -1000 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,3]
        Output: 6
        Explanation: The optimal path is 2 -> 1 -> 3 with path sum 2 + 1 + 3 = 6
    
    Example 2:
        Input: root = [-10,9,20,null,null,15,7]
        Output: 42
        Explanation: The optimal path is 15 -> 20 -> 7 with path sum 15 + 20 + 7 = 42
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h) where h is height
    
    GOAL: Master tree DP with global optimization
    """
    max_sum = float('-inf')
    
    def max_gain(node):
        """Return max path sum starting from node going down"""
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Max gain from left and right subtrees (0 if negative)
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # Max path sum through current node
        current_max = node.val + left_gain + right_gain
        max_sum = max(max_sum, current_max)
        
        # Return max gain if we start from current node
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum


# =============================================================================
# PROBLEM 2: HOUSE ROBBER III (MEDIUM) - 45 MIN
# =============================================================================

def rob_tree(root: Optional[TreeNode]) -> int:
    """
    PROBLEM: House Robber III
    
    The thief has found himself a new place for his thievery. There is only 
    one entrance to this area, called root. Besides the root, each house has 
    one and only one parent house. After a tour, the smart thief realized that 
    all houses form a binary tree. If two directly-linked houses were broken 
    into on the same night, it will automatically contact the police.
    
    Given the root of the binary tree, return the maximum amount of money the 
    thief can rob without alerting the police.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 10^4]
    - 0 <= Node.val <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: root = [3,2,3,null,3,null,1]
        Output: 7
        Explanation: Maximum amount = 3 + 3 + 1 = 7
    
    Example 2:
        Input: root = [3,4,5,1,3,null,1]
        Output: 9
        Explanation: Maximum amount = 4 + 5 = 9
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h)
    
    GOAL: Apply state machine DP to tree structures
    """
    def rob_helper(node):
        """Return (rob_current, not_rob_current)"""
        if not node:
            return (0, 0)
        
        left_rob, left_not_rob = rob_helper(node.left)
        right_rob, right_not_rob = rob_helper(node.right)
        
        # If we rob current node, we can't rob children
        rob_current = node.val + left_not_rob + right_not_rob
        
        # If we don't rob current, we can choose optimally for children
        not_rob_current = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
        
        return (rob_current, not_rob_current)
    
    rob_current, not_rob_current = rob_helper(root)
    return max(rob_current, not_rob_current)


# =============================================================================
# PROBLEM 3: DIAMETER OF BINARY TREE (EASY) - 30 MIN
# =============================================================================

def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    """
    PROBLEM: Diameter of Binary Tree
    
    Given the root of a binary tree, return the length of the diameter of the tree.
    The diameter of a binary tree is the length of the longest path between any 
    two nodes in a tree. This path may or may not pass through the root.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [1, 10^4]
    - -100 <= Node.val <= 100
    
    EXAMPLES:
    Example 1:
        Input: root = [1,2,3,4,5]
        Output: 3
        Explanation: The diameter is the path [4,2,1,3] or [5,2,1,3] with length 3
    
    Example 2:
        Input: root = [1,2]
        Output: 1
        Explanation: The diameter is the path [1,2] with length 1
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h)
    
    GOAL: Understand diameter pattern in tree DP
    """
    max_diameter = 0
    
    def depth(node):
        """Return depth of subtree rooted at node"""
        nonlocal max_diameter
        
        if not node:
            return 0
        
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        
        # Diameter through current node
        current_diameter = left_depth + right_depth
        max_diameter = max(max_diameter, current_diameter)
        
        # Return depth of current subtree
        return max(left_depth, right_depth) + 1
    
    depth(root)
    return max_diameter


# =============================================================================
# PROBLEM 4: LONGEST UNIVALUE PATH (MEDIUM) - 60 MIN
# =============================================================================

def longest_univalue_path(root: Optional[TreeNode]) -> int:
    """
    PROBLEM: Longest Univalue Path
    
    Given the root of a binary tree, return the length of the longest path, 
    where each node in the path has the same value. This path may or may not 
    pass through the root. The length of the path between two nodes is 
    represented by the number of edges between them.
    
    CONSTRAINTS:
    - The number of nodes in the tree is in the range [0, 10^4]
    - -1000 <= Node.val <= 1000
    - The depth of the tree will not exceed 1000
    
    EXAMPLES:
    Example 1:
        Input: root = [5,4,5,1,1,null,5]
        Output: 2
        Explanation: The longest path is [5,5,5] with length 2
    
    Example 2:
        Input: root = [1,4,5,4,4,null,5]
        Output: 2
        Explanation: The longest path is [4,4,4] with length 2
    
    EXPECTED TIME COMPLEXITY: O(n)
    EXPECTED SPACE COMPLEXITY: O(h)
    
    GOAL: Handle conditional path extension in tree DP
    """
    max_length = 0
    
    def longest_path(node):
        """Return longest univalue path starting from node going down"""
        nonlocal max_length
        
        if not node:
            return 0
        
        left_length = longest_path(node.left)
        right_length = longest_path(node.right)
        
        # Reset lengths if values don't match
        if node.left and node.left.val != node.val:
            left_length = 0
        if node.right and node.right.val != node.val:
            right_length = 0
        
        # Longest path through current node
        current_max = left_length + right_length
        max_length = max(max_length, current_max)
        
        # Return longest path starting from current node
        return max(left_length, right_length) + 1
    
    longest_path(root)
    return max_length


# =============================================================================
# PROBLEM 5: MINIMUM COST TO MAKE ARRAY NON-DECREASING (HARD) - 90 MIN
# =============================================================================

def min_cost_to_make_array_non_decreasing(nums: List[int], cost: List[int]) -> int:
    """
    PROBLEM: Minimum Cost to Make Array Non-decreasing
    
    You are given two 0-indexed integer arrays nums and cost consisting each of n 
    positive integers. You can do the following operation any number of times:
    - Increase or decrease nums[i] by 1 with cost of cost[i]
    
    Return the minimum cost to make nums non-decreasing.
    
    CONSTRAINTS:
    - n == nums.length == cost.length
    - 1 <= n <= 10^5
    - 1 <= nums[i], cost[i] <= 10^6
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,3,5,2], cost = [2,3,1,14]
        Output: 8
        Explanation: Change nums[3] from 2 to 3, cost = 14. Change nums[3] from 3 to 4, cost = 14. 
        Change nums[3] from 4 to 5, cost = 14. Total cost = 14 + 14 + 14 = 42.
        Actually optimal: Change nums[3] from 2 to 3 (cost 14), then nums[2] from 5 to 3 (cost 2). 
        Total = 16. Even better: keep nums[1]=3, change nums[2] to 3 (cost 2), change nums[3] to 3 (cost 14). 
        Total = 16. Optimal: nums[0] stays 1, nums[1] stays 3, nums[2] becomes 3 (cost 2), nums[3] becomes 3 (cost 14). 
        But 1 <= 3 <= 3 <= 3, so cost is 2 + 14 = 16.
        Wait, let me recalculate: If final array is [1,3,3,3], cost = 0 + 0 + 2*2 + 14*1 = 18.
        If final is [3,3,3,3], cost = 2*2 + 0 + 1*2 + 14*1 = 20.
        If final is [1,1,1,2], cost = 0 + 3*2 + 1*4 + 0 = 10. But this violates non-decreasing.
        Let me try [1,3,3,3]: cost = 0 + 0 + 1*2 + 14*1 = 16.
        Actually [1,1,2,2]: cost = 0 + 3*2 + 1*3 + 0 = 9. But 1,1,2,2 is non-decreasing.
        Wait, example says output is 8. Let me try [1,1,1,1]: cost = 0 + 3*2 + 1*4 + 14*1 = 24.
        [2,2,2,2]: cost = 2*1 + 3*1 + 1*3 + 0 = 8. Yes!
    
    Example 2:
        Input: nums = [1,2,3,4,5], cost = [1,1,1,1,1]
        Output: 0
        Explanation: Array is already non-decreasing
    
    EXPECTED TIME COMPLEXITY: O(n²) naive DP, O(n log n) optimized
    EXPECTED SPACE COMPLEXITY: O(n)
    
    GOAL: Apply DP with optimization constraints
    """
    n = len(nums)
    if n <= 1:
        return 0
    
    # Get all possible target values
    targets = sorted(set(nums))
    
    # dp[i][j] = min cost to make nums[0:i+1] non-decreasing with nums[i] = targets[j]
    prev_dp = [float('inf')] * len(targets)
    
    # Initialize for first element
    for j, target in enumerate(targets):
        prev_dp[j] = cost[0] * abs(nums[0] - target)
    
    for i in range(1, n):
        curr_dp = [float('inf')] * len(targets)
        
        # For each possible value for current position
        for j, target in enumerate(targets):
            change_cost = cost[i] * abs(nums[i] - target)
            
            # Try all valid previous values (non-decreasing constraint)
            for k in range(j + 1):  # targets[k] <= targets[j]
                curr_dp[j] = min(curr_dp[j], prev_dp[k] + change_cost)
        
        prev_dp = curr_dp
    
    return min(prev_dp)


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day23_problems():
    """Test all Day 23 problems"""
    
    print("=" * 60)
    print("         DAY 23: DP ON TREES AND GRAPHS")
    print("=" * 60)
    
    # Test Binary Tree Maximum Path Sum
    print("\n🧪 Testing Binary Tree Maximum Path Sum")
    # Create tree: [1,2,3]
    root1 = TreeNode(1)
    root1.left = TreeNode(2)
    root1.right = TreeNode(3)
    
    max_sum1 = max_path_sum(root1)
    print(f"Max Path Sum [1,2,3]: {max_sum1} (Expected: 6)")
    
    # Create tree: [-10,9,20,null,null,15,7]
    root2 = TreeNode(-10)
    root2.left = TreeNode(9)
    root2.right = TreeNode(20)
    root2.right.left = TreeNode(15)
    root2.right.right = TreeNode(7)
    
    max_sum2 = max_path_sum(root2)
    print(f"Max Path Sum [-10,9,20,null,null,15,7]: {max_sum2} (Expected: 42)")
    
    # Test House Robber III
    print("\n🧪 Testing House Robber III")
    # Create tree: [3,2,3,null,3,null,1]
    rob_root = TreeNode(3)
    rob_root.left = TreeNode(2)
    rob_root.right = TreeNode(3)
    rob_root.left.right = TreeNode(3)
    rob_root.right.right = TreeNode(1)
    
    rob_amount = rob_tree(rob_root)
    print(f"House Robber III: {rob_amount} (Expected: 7)")
    
    # Test Diameter of Binary Tree
    print("\n🧪 Testing Diameter of Binary Tree")
    # Create tree: [1,2,3,4,5]
    diameter_root = TreeNode(1)
    diameter_root.left = TreeNode(2)
    diameter_root.right = TreeNode(3)
    diameter_root.left.left = TreeNode(4)
    diameter_root.left.right = TreeNode(5)
    
    diameter = diameter_of_binary_tree(diameter_root)
    print(f"Diameter: {diameter} (Expected: 3)")
    
    # Test Longest Univalue Path
    print("\n🧪 Testing Longest Univalue Path")
    # Create tree: [5,4,5,1,1,null,5]
    uni_root = TreeNode(5)
    uni_root.left = TreeNode(4)
    uni_root.right = TreeNode(5)
    uni_root.left.left = TreeNode(1)
    uni_root.left.right = TreeNode(1)
    uni_root.right.right = TreeNode(5)
    
    uni_length = longest_univalue_path(uni_root)
    print(f"Longest Univalue Path: {uni_length} (Expected: 2)")
    
    # Test Minimum Cost to Make Array Non-decreasing
    print("\n🧪 Testing Min Cost Non-decreasing Array")
    min_cost1 = min_cost_to_make_array_non_decreasing([1,3,5,2], [2,3,1,14])
    print(f"Min Cost [1,3,5,2], [2,3,1,14]: {min_cost1} (Expected: 8)")
    
    min_cost2 = min_cost_to_make_array_non_decreasing([1,2,3,4,5], [1,1,1,1,1])
    print(f"Min Cost [1,2,3,4,5], [1,1,1,1,1]: {min_cost2} (Expected: 0)")
    
    print("\n" + "=" * 60)
    print("           DAY 23 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# TREE DP PATTERNS SUMMARY
# =============================================================================

def tree_dp_patterns_summary():
    """Summary of key tree DP patterns"""
    
    print("\n" + "=" * 70)
    print("                    TREE DP PATTERNS SUMMARY")
    print("=" * 70)
    
    print("\n🌳 COMMON TREE DP PATTERNS:")
    print("• Path Through Node: max/min path passing through current node")
    print("• Include/Exclude Node: optimal choice considering current node")
    print("• Subtree Properties: aggregate information from subtrees")
    print("• Diameter Problems: longest path between any two nodes")
    
    print("\n📊 STATE DEFINITIONS:")
    print("• Single Value: return one optimal value from subtree")
    print("• Multiple States: return tuple of different choices")
    print("• Global Variable: maintain global optimum across calls")
    print("• Memoization: cache results for overlapping subproblems")
    
    print("\n🎯 PROBLEM TYPES:")
    print("• Path Sum: Maximum/minimum path sums")
    print("• Diameter: Longest paths in trees")
    print("• Robber: Include/exclude with constraints")
    print("• Univalue: Conditional path extensions")
    
    print("\n💡 OPTIMIZATION TECHNIQUES:")
    print("• Bottom-up Processing: children before parent")
    print("• State Compression: reduce number of states")
    print("• Early Termination: prune impossible branches")
    print("• Global Tracking: maintain optimal across recursion")
    
    print("=" * 70)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 23 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review tree DP patterns
    """
    
    print("🚀 Starting Day 23: DP on Trees and Graphs")
    print("\n📚 Theory Topics:")
    print("- Tree DP with subtree states")
    print("- Path problems on trees")
    print("- Graph DP with memoization")
    print("- State space reduction techniques")
    
    print("\n💻 Practice Problems:")
    print("1. Binary Tree Maximum Path Sum (Hard) - 60 min")
    print("2. House Robber III (Medium) - 45 min")
    print("3. Diameter of Binary Tree (Easy) - 30 min")
    print("4. Longest Univalue Path (Medium) - 60 min")
    print("5. Min Cost Non-decreasing Array (Hard) - 90 min")
    
    print("\n🧪 Running Tests...")
    test_day23_problems()
    
    print("\n📊 Tree DP Patterns Summary...")
    tree_dp_patterns_summary()
    
    print("\n✅ Day 23 Complete!")
    print("📈 Next: Day 24 - Final Review & Mock Interviews")


if __name__ == "__main__":
    main() 