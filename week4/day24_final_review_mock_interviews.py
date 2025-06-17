"""
=============================================================================
                        DAY 24: FINAL REVIEW & MOCK INTERVIEWS
                           Meta Interview Preparation
                              Week 4 - Day 24
=============================================================================

FOCUS: Comprehensive review, interview simulation
TIME ALLOCATION: 4 hours
- Review (1 hour): All concepts recap
- Mock Interviews (3 hours): Simulated interview problems

TOPICS COVERED:
- Complete algorithm mastery review
- Interview simulation and timing
- Problem-solving strategies
- Communication and optimization

=============================================================================
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque, Counter
import heapq
from functools import lru_cache


# =============================================================================
# 4-WEEK MASTERY REVIEW (1 HOUR)
# =============================================================================

"""
COMPLETE ALGORITHM MASTERY CHECKLIST:

✅ WEEK 1 - FOUNDATIONS:
  • Arrays & Hash Tables: Two Sum patterns, sliding window
  • Strings: Pattern matching, manipulation
  • Linked Lists: Traversal, modification, cycle detection
  • Stacks & Queues: Monotonic stacks, BFS applications

✅ WEEK 2 - TREES & SEARCH:
  • Binary Trees: Traversals, properties, construction
  • BST: Search, insertion, validation
  • Binary Search: Templates, rotated arrays
  • Heaps: Priority queues, top-K problems

✅ WEEK 3 - GRAPHS & ADVANCED:
  • Graph Traversal: BFS/DFS mastery
  • Graph Algorithms: Topological sort, cycle detection
  • Union-Find: Connectivity problems
  • Tries: Prefix operations, string algorithms

✅ WEEK 4 - DYNAMIC PROGRAMMING:
  • 1D DP: Linear patterns, optimization
  • 2D DP: Grid problems, string matching
  • Advanced DP: Complex state spaces
  • DP Optimization: Space/time improvements

INTERVIEW READINESS SCORE: 95/100 🎯
"""


# =============================================================================
# MOCK INTERVIEW PROBLEM 1: DESIGN DATA STRUCTURE (HARD) - 45 MIN
# =============================================================================

class LRUCache:
    """
    PROBLEM: LRU Cache
    
    Design a data structure that follows the constraints of a Least Recently 
    Used (LRU) cache. Implement get and put operations in O(1) time.
    
    This combines multiple concepts: Hash tables + Doubly linked lists
    """
    
    class Node:
        def __init__(self, key=0, value=0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Dummy head and tail for easier manipulation
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove an existing node"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node):
        """Move node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove last node before tail"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if not node:
            return -1
        
        # Move to head (recently used)
        self._move_to_head(node)
        return node.value
    
    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        
        if not node:
            # New key
            new_node = self.Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove LRU
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            # Update existing
            node.value = value
            self._move_to_head(node)


# =============================================================================
# MOCK INTERVIEW PROBLEM 2: SYSTEM DESIGN COMPONENT (HARD) - 60 MIN
# =============================================================================

class TimeBasedKeyValueStore:
    """
    PROBLEM: Time Based Key-Value Store
    
    Design a time-based key-value data structure that can store multiple 
    values for the same key at different time stamps and retrieve the key's 
    value at a certain timestamp.
    
    Combines: Hash tables + Binary search + System design
    """
    
    def __init__(self):
        self.store = defaultdict(list)  # key -> [(timestamp, value)]
    
    def set(self, key: str, value: str, timestamp: int) -> None:
        """Store key-value pair at given timestamp"""
        self.store[key].append((timestamp, value))
    
    def get(self, key: str, timestamp: int) -> str:
        """Get value at timestamp (or latest before timestamp)"""
        if key not in self.store:
            return ""
        
        values = self.store[key]
        
        # Binary search for largest timestamp <= given timestamp
        left, right = 0, len(values) - 1
        result = ""
        
        while left <= right:
            mid = (left + right) // 2
            if values[mid][0] <= timestamp:
                result = values[mid][1]
                left = mid + 1
            else:
                right = mid - 1
        
        return result


# =============================================================================
# MOCK INTERVIEW PROBLEM 3: ALGORITHM OPTIMIZATION (HARD) - 60 MIN
# =============================================================================

def max_sliding_window(nums: List[int], k: int) -> List[int]:
    """
    PROBLEM: Sliding Window Maximum
    
    You are given an array of integers nums, there is a sliding window of 
    size k which is moving from the very left of the array to the very right. 
    Return the max sliding window.
    
    Tests: Deque usage, sliding window optimization
    TIME: O(n), SPACE: O(k)
    """
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result when window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


def trap_rain_water(height: List[int]) -> int:
    """
    PROBLEM: Trapping Rain Water
    
    Given n non-negative integers representing an elevation map where the 
    width of each bar is 1, compute how much water it can trap after raining.
    
    Tests: Two pointers, optimization thinking
    TIME: O(n), SPACE: O(1)
    """
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water


# =============================================================================
# MOCK INTERVIEW PROBLEM 4: COMPLEX INTEGRATION (HARD) - 75 MIN
# =============================================================================

def word_break_ii(s: str, wordDict: List[str]) -> List[str]:
    """
    PROBLEM: Word Break II
    
    Given a string s and a dictionary of strings wordDict, add spaces in s 
    to construct a sentence where each word is a valid dictionary word. 
    Return all such possible sentences.
    
    Combines: DP + Backtracking + Trie optimization
    TIME: O(2^n) worst case, SPACE: O(2^n)
    """
    word_set = set(wordDict)
    memo = {}
    
    def backtrack(start):
        if start in memo:
            return memo[start]
        
        if start == len(s):
            return [""]
        
        result = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                rest_sentences = backtrack(end)
                for sentence in rest_sentences:
                    if sentence:
                        result.append(word + " " + sentence)
                    else:
                        result.append(word)
        
        memo[start] = result
        return result
    
    return backtrack(0)


def serialize_deserialize_tree(root):
    """
    PROBLEM: Serialize and Deserialize Binary Tree
    
    Design an algorithm to serialize and deserialize a binary tree.
    
    Tests: Tree traversal, string manipulation, design skills
    """
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    def serialize(root):
        """Serialize tree to string"""
        def preorder(node):
            if not node:
                return "null"
            return str(node.val) + "," + preorder(node.left) + "," + preorder(node.right)
        
        return preorder(root)
    
    def deserialize(data):
        """Deserialize string to tree"""
        def build_tree():
            val = next(values)
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.left = build_tree()
            node.right = build_tree()
            return node
        
        values = iter(data.split(","))
        return build_tree()
    
    return serialize, deserialize


# =============================================================================
# INTERVIEW SIMULATION FRAMEWORK
# =============================================================================

class MockInterview:
    """
    Simulates a real technical interview environment
    """
    
    def __init__(self):
        self.problems = [
            {
                "name": "LRU Cache",
                "difficulty": "Hard",
                "time_limit": 45,
                "concepts": ["Hash Table", "Doubly Linked List", "Design"],
                "function": "LRUCache implementation"
            },
            {
                "name": "Time Based Key-Value Store", 
                "difficulty": "Hard",
                "time_limit": 60,
                "concepts": ["Hash Table", "Binary Search", "System Design"],
                "function": "TimeBasedKeyValueStore implementation"
            },
            {
                "name": "Sliding Window Maximum",
                "difficulty": "Hard", 
                "time_limit": 60,
                "concepts": ["Deque", "Sliding Window", "Optimization"],
                "function": max_sliding_window
            },
            {
                "name": "Word Break II",
                "difficulty": "Hard",
                "time_limit": 75,
                "concepts": ["DP", "Backtracking", "Memoization"],
                "function": word_break_ii
            }
        ]
    
    def conduct_interview(self, problem_index=0):
        """Simulate interview for given problem"""
        problem = self.problems[problem_index]
        
        print(f"\n{'='*60}")
        print(f"         MOCK INTERVIEW SESSION")
        print(f"{'='*60}")
        print(f"\n🎯 Problem: {problem['name']}")
        print(f"📊 Difficulty: {problem['difficulty']}")
        print(f"⏰ Time Limit: {problem['time_limit']} minutes")
        print(f"🧠 Concepts: {', '.join(problem['concepts'])}")
        
        print(f"\n💭 Interview Tips:")
        print("1. Think out loud - explain your approach")
        print("2. Start with brute force, then optimize")
        print("3. Consider edge cases")
        print("4. Analyze time/space complexity")
        print("5. Test with examples")
        
        return problem


# =============================================================================
# COMPREHENSIVE TESTING
# =============================================================================

def test_mock_interviews():
    """Test all mock interview problems"""
    
    print("=" * 60)
    print("         DAY 24: FINAL REVIEW & MOCK INTERVIEWS")
    print("=" * 60)
    
    # Test LRU Cache
    print("\n🧪 Testing LRU Cache")
    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    print(f"Get 1: {lru.get(1)} (Expected: 1)")
    lru.put(3, 3)  # Evicts key 2
    print(f"Get 2: {lru.get(2)} (Expected: -1)")
    
    # Test Time Based KV Store
    print("\n🧪 Testing Time Based Key-Value Store")
    kv = TimeBasedKeyValueStore()
    kv.set("foo", "bar", 1)
    print(f"Get foo at 1: '{kv.get('foo', 1)}' (Expected: 'bar')")
    print(f"Get foo at 3: '{kv.get('foo', 3)}' (Expected: 'bar')")
    kv.set("foo", "bar2", 4)
    print(f"Get foo at 4: '{kv.get('foo', 4)}' (Expected: 'bar2')")
    print(f"Get foo at 5: '{kv.get('foo', 5)}' (Expected: 'bar2')")
    
    # Test Sliding Window Maximum
    print("\n🧪 Testing Sliding Window Maximum")
    window_max = max_sliding_window([1,3,-1,-3,5,3,6,7], 3)
    print(f"Sliding Window Max: {window_max} (Expected: [3,3,5,5,6,7])")
    
    # Test Trapping Rain Water
    print("\n🧪 Testing Trapping Rain Water")
    trapped = trap_rain_water([0,1,0,2,1,0,1,3,2,1,2,1])
    print(f"Trapped Water: {trapped} (Expected: 6)")
    
    # Test Word Break II
    print("\n🧪 Testing Word Break II")
    sentences = word_break_ii("catsanddog", ["cat","cats","and","sand","dog"])
    print(f"Word Break II: {len(sentences)} sentences")
    if sentences:
        print(f"First sentence: '{sentences[0]}'")
    
    print("\n" + "=" * 60)
    print("           MOCK INTERVIEW TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# FINAL PREPARATION SUMMARY
# =============================================================================

def final_preparation_summary():
    """Complete preparation summary and next steps"""
    
    print("\n" + "=" * 70)
    print("                    🎉 INTERVIEW PREPARATION COMPLETE! 🎉")
    print("=" * 70)
    
    print("\n🏆 4-WEEK ACHIEVEMENT SUMMARY:")
    print("✅ Week 1: Mastered foundational data structures")
    print("✅ Week 2: Conquered trees, search, and heaps")
    print("✅ Week 3: Dominated graphs and advanced algorithms")
    print("✅ Week 4: Perfected dynamic programming")
    
    print("\n📊 SKILLS MASTERED (96 hours of focused practice):")
    print("• Arrays & Hash Tables: EXPERT")
    print("• Strings & Linked Lists: EXPERT")
    print("• Stacks & Queues: EXPERT")
    print("• Binary Trees & BST: EXPERT")
    print("• Binary Search: EXPERT")
    print("• Heaps & Priority Queues: EXPERT")
    print("• Graph Algorithms: EXPERT")
    print("• Union-Find: EXPERT")
    print("• Tries: EXPERT")
    print("• Dynamic Programming: EXPERT")
    
    print("\n🎯 INTERVIEW READINESS:")
    print("• Problem Recognition: INSTANT")
    print("• Algorithm Selection: OPTIMAL")
    print("• Implementation Speed: FAST")
    print("• Optimization Skills: ADVANCED")
    print("• Communication: CLEAR")
    
    print("\n🚀 META INTERVIEW CONFIDENCE: 95%")
    
    print("\n💡 FINAL INTERVIEW TIPS:")
    print("1. Stay calm and think out loud")
    print("2. Start with brute force, then optimize")
    print("3. Handle edge cases explicitly")
    print("4. Always analyze complexity")
    print("5. Practice mock interviews regularly")
    
    print("\n🎊 YOU'RE READY FOR META! GO CRUSH THOSE INTERVIEWS! 🎊")
    print("=" * 70)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 24 Final Review Routine:
    1. Complete concept review (1 hour)
    2. Mock interview sessions (3 hours)
    3. Final preparation summary
    4. Interview confidence building
    """
    
    print("🚀 Starting Day 24: Final Review & Mock Interviews")
    print("\n📚 Final Review Topics:")
    print("- Complete algorithm mastery verification")
    print("- Interview simulation and timing")
    print("- Problem-solving strategy refinement")
    print("- Confidence building exercises")
    
    print("\n🎭 Mock Interview Problems:")
    print("1. LRU Cache (Hard) - 45 min")
    print("2. Time Based Key-Value Store (Hard) - 60 min")
    print("3. Sliding Window Maximum (Hard) - 60 min")
    print("4. Word Break II (Hard) - 75 min")
    
    print("\n🧪 Running Final Tests...")
    test_mock_interviews()
    
    print("\n✅ Day 24 Complete!")
    final_preparation_summary()


if __name__ == "__main__":
    main() 