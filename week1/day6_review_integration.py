"""
=============================================================================
                      WEEK 1 - DAY 6: REVIEW & INTEGRATION
                           Meta Interview Preparation
=============================================================================

WEEK 1 COMPREHENSIVE REVIEW
===========================

This day consolidates all Week 1 learning with:
1. Concept integration and pattern recognition
2. Mixed practice problems combining multiple topics
3. Mock interview simulation
4. Preparation for Week 2 (Trees & Graphs)

TOPICS COVERED THIS WEEK:
- Day 1: Arrays & Two Pointers
- Day 2: Strings & Pattern Matching  
- Day 3: Hash Tables & Sets
- Day 4: Linked Lists
- Day 5: Stacks & Queues

KEY PATTERNS MASTERED:
1. Two Pointers: Opposite ends, same direction, fast/slow
2. Sliding Window: Fixed and variable size windows
3. Hash Maps: Frequency counting, complement lookup, grouping
4. Linked List: Pointer manipulation, cycle detection, merging
5. Stack: LIFO operations, monotonic patterns, nested structures
6. Queue: FIFO operations, BFS preparation, sliding window max

INTEGRATION THEMES:
- Hash tables accelerate array/string problems
- Two pointers optimize nested loop patterns
- Stacks/queues bridge to tree/graph traversals
- All patterns combine in complex problems

=============================================================================
"""

from collections import defaultdict, deque, Counter
import heapq


# INTEGRATION PROBLEM 1: LRU Cache (Hash Table + Doubly Linked List)
class LRUCache:
    """
    Least Recently Used Cache
    
    Combines hash table (O(1) access) with doubly linked list (O(1) insertion/deletion)
    Demonstrates integration of multiple data structures
    
    All operations: O(1) time, O(capacity) space
    """
    
    class Node:
        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # Hash table: key -> node
        
        # Dummy head and tail for easier list manipulation
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
        """Remove existing node"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node):
        """Move node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove least recently used node"""
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        return lru_node
    
    def get(self, key):
        """Get value and mark as recently used"""
        node = self.cache.get(key)
        
        if not node:
            return -1
        
        # Move to head (mark as recently used)
        self._move_to_head(node)
        return node.val
    
    def put(self, key, value):
        """Put key-value pair"""
        node = self.cache.get(key)
        
        if not node:
            new_node = self.Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove LRU node
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            # Add new node
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            # Update existing node
            node.val = value
            self._move_to_head(node)


# INTEGRATION PROBLEM 2: Design Browser History (Stack + Array)
class BrowserHistory:
    """
    Browser History with back/forward functionality
    
    Combines array-like structure with stack operations
    Demonstrates practical application of data structures
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self, homepage):
        self.history = [homepage]
        self.current = 0
    
    def visit(self, url):
        """Visit new URL (clears forward history)"""
        # Remove forward history
        self.history = self.history[:self.current + 1]
        self.history.append(url)
        self.current += 1
    
    def back(self, steps):
        """Go back in history"""
        self.current = max(0, self.current - steps)
        return self.history[self.current]
    
    def forward(self, steps):
        """Go forward in history"""
        self.current = min(len(self.history) - 1, self.current + steps)
        return self.history[self.current]


# INTEGRATION PROBLEM 3: Design Phone Directory (Trie + Set)
class PhoneDirectory:
    """
    Phone directory with prefix search
    
    Combines Trie (prefix matching) with hash sets (fast lookup)
    Shows integration of string processing with tree structures
    """
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.phone_numbers = set()
    
    def __init__(self):
        self.root = self.TrieNode()
        self.all_numbers = set()
    
    def add_contact(self, name, phone):
        """Add contact with name and phone number"""
        self.all_numbers.add(phone)
        
        # Add to trie for prefix search
        node = self.root
        for char in name.lower():
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
            node.phone_numbers.add(phone)
    
    def search_by_prefix(self, prefix):
        """Find all phone numbers for names starting with prefix"""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        return list(node.phone_numbers)
    
    def has_phone(self, phone):
        """Check if phone number exists"""
        return phone in self.all_numbers


# INTEGRATION PROBLEM 4: Evaluate Mathematical Expression
def evaluate_expression(expression):
    """
    Evaluate mathematical expression with +, -, *, /, (, )
    
    Combines:
    - Stack for operator precedence
    - String processing for parsing
    - Two stacks algorithm
    
    Time: O(n), Space: O(n)
    """
    def apply_operator(operators, values):
        """Apply top operator to top two values"""
        operator = operators.pop()
        right = values.pop()
        left = values.pop()
        
        if operator == '+':
            values.append(left + right)
        elif operator == '-':
            values.append(left - right)
        elif operator == '*':
            values.append(left * right)
        elif operator == '/':
            values.append(left // right)
    
    def precedence(op):
        """Return operator precedence"""
        if op in ['+', '-']:
            return 1
        if op in ['*', '/']:
            return 2
        return 0
    
    values = []  # Stack for numbers
    operators = []  # Stack for operators
    i = 0
    
    while i < len(expression):
        char = expression[i]
        
        if char == ' ':
            i += 1
            continue
        
        if char.isdigit():
            # Parse number
            num = 0
            while i < len(expression) and expression[i].isdigit():
                num = num * 10 + int(expression[i])
                i += 1
            values.append(num)
            continue
        
        if char == '(':
            operators.append(char)
        elif char == ')':
            # Apply all operators until '('
            while operators and operators[-1] != '(':
                apply_operator(operators, values)
            operators.pop()  # Remove '('
        else:
            # Operator
            while (operators and operators[-1] != '(' and
                   precedence(operators[-1]) >= precedence(char)):
                apply_operator(operators, values)
            operators.append(char)
        
        i += 1
    
    # Apply remaining operators
    while operators:
        apply_operator(operators, values)
    
    return values[0]


# INTEGRATION PROBLEM 5: Design Hit Counter (Queue + Sliding Window)
class HitCounter:
    """
    Design hit counter that counts hits in past 5 minutes
    
    Combines queue (time-based ordering) with sliding window concept
    Efficient for time-based problems
    
    Time: O(1) amortized, Space: O(k) where k is number of hits in window
    """
    
    def __init__(self):
        self.hits = deque()  # Store (timestamp, count) pairs
        self.total = 0
    
    def hit(self, timestamp):
        """Record a hit at given timestamp"""
        # Group hits by timestamp for efficiency
        if self.hits and self.hits[-1][0] == timestamp:
            # Same timestamp, increment count
            old_count = self.hits[-1][1]
            self.hits[-1] = (timestamp, old_count + 1)
            self.total += 1
        else:
            # New timestamp
            self.hits.append((timestamp, 1))
            self.total += 1
        
        # Remove old hits (older than 300 seconds)
        self._clean_old_hits(timestamp)
    
    def get_hits(self, timestamp):
        """Get number of hits in past 300 seconds"""
        self._clean_old_hits(timestamp)
        return self.total
    
    def _clean_old_hits(self, timestamp):
        """Remove hits older than 300 seconds"""
        while self.hits and self.hits[0][0] <= timestamp - 300:
            _, count = self.hits.popleft()
            self.total -= count


# COMPREHENSIVE MIXED PRACTICE PROBLEMS

def find_anagram_groups_with_indices(words):
    """
    Group anagrams and return their original indices
    
    Combines: String processing + Hash tables + Array indexing
    
    Time: O(n * m log m), Space: O(n * m)
    """
    anagram_map = defaultdict(list)
    
    for i, word in enumerate(words):
        # Use sorted word as key
        key = ''.join(sorted(word))
        anagram_map[key].append(i)
    
    return [indices for indices in anagram_map.values() if len(indices) > 1]


def merge_k_sorted_arrays(arrays):
    """
    Merge k sorted arrays into one sorted array
    
    Combines: Heap operations + Array merging + Multiple pointers
    
    Time: O(n log k), Space: O(k)
    """
    heap = []
    result = []
    
    # Initialize heap with first element from each array
    for i, array in enumerate(arrays):
        if array:
            heapq.heappush(heap, (array[0], i, 0))
    
    while heap:
        val, array_idx, element_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same array
        if element_idx + 1 < len(arrays[array_idx]):
            next_val = arrays[array_idx][element_idx + 1]
            heapq.heappush(heap, (next_val, array_idx, element_idx + 1))
    
    return result


def longest_valid_parentheses(s):
    """
    Find length of longest valid parentheses substring
    
    Combines: Stack operations + String processing + Two pointers
    
    Time: O(n), Space: O(n)
    """
    stack = [-1]  # Base for valid substring calculation
    max_length = 0
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:  # char == ')'
            stack.pop()
            
            if not stack:
                # No matching '(', push current index as new base
                stack.append(i)
            else:
                # Calculate length of valid substring
                length = i - stack[-1]
                max_length = max(max_length, length)
    
    return max_length


def find_all_duplicates(nums):
    """
    Find all duplicates in array where 1 ≤ nums[i] ≤ n
    
    Combines: Array as hash table + Negative marking technique
    
    Time: O(n), Space: O(1)
    """
    result = []
    
    for num in nums:
        index = abs(num) - 1
        
        if nums[index] < 0:
            # Already seen this number
            result.append(abs(num))
        else:
            # Mark as seen by making negative
            nums[index] = -nums[index]
    
    return result


def three_sum_closest(nums, target):
    """
    Find three numbers whose sum is closest to target
    
    Combines: Two pointers + Sorting + Optimization
    
    Time: O(n²), Space: O(1)
    """
    nums.sort()
    closest_sum = float('inf')
    min_diff = float('inf')
    
    for i in range(len(nums) - 2):
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            diff = abs(current_sum - target)
            
            if diff < min_diff:
                min_diff = diff
                closest_sum = current_sum
            
            if current_sum < target:
                left += 1
            else:
                right -= 1
    
    return closest_sum


# MOCK INTERVIEW PROBLEMS
def mock_interview_problem_1():
    """
    Problem: Design a data structure that supports:
    1. Insert a number
    2. Remove a number  
    3. Get random number from current numbers
    
    All operations should be O(1) average time
    """
    
    class RandomizedSet:
        def __init__(self):
            self.nums = []  # Array for O(1) random access
            self.indices = {}  # Hash map: num -> index in array
        
        def insert(self, val):
            if val in self.indices:
                return False
            
            # Add to end of array
            self.indices[val] = len(self.nums)
            self.nums.append(val)
            return True
        
        def remove(self, val):
            if val not in self.indices:
                return False
            
            # Swap with last element and remove
            last_element = self.nums[-1]
            idx_to_remove = self.indices[val]
            
            self.nums[idx_to_remove] = last_element
            self.indices[last_element] = idx_to_remove
            
            # Remove last element
            self.nums.pop()
            del self.indices[val]
            return True
        
        def get_random(self):
            import random
            return random.choice(self.nums)
    
    return RandomizedSet


def mock_interview_problem_2(matrix, target):
    """
    Search in row-wise and column-wise sorted matrix
    
    Combines: Two pointers + Matrix traversal + Binary search concepts
    
    Time: O(m + n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    # Start from top-right corner
    row, col = 0, len(matrix[0]) - 1
    
    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1  # Move left
        else:
            row += 1  # Move down
    
    return False


# COMPREHENSIVE TESTING AND REVIEW
def week1_comprehensive_test():
    """
    Test integration problems and review all concepts
    """
    print("=== WEEK 1 COMPREHENSIVE REVIEW TEST ===\n")
    
    # Test LRU Cache
    print("1. LRU Cache Test:")
    lru = LRUCache(2)
    operations = [
        ("put", 1, 1), ("put", 2, 2), ("get", 1), 
        ("put", 3, 3), ("get", 2), ("get", 3), ("get", 1)
    ]
    expected = [None, None, 1, None, -1, 3, 1]
    
    results = []
    for op_data in operations:
        if op_data[0] == "put":
            lru.put(op_data[1], op_data[2])
            results.append(None)
        else:  # get
            results.append(lru.get(op_data[1]))
    
    print(f"   Operations: {operations}")
    print(f"   Results: {results}")
    print(f"   Expected: {expected}")
    print(f"   ✓ Correct" if results == expected else f"   ✗ Wrong")
    print()
    
    # Test Expression Evaluation
    print("2. Expression Evaluation:")
    expr_tests = [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("10 + 2 * 6", 22)
    ]
    
    for expr, expected in expr_tests:
        result = evaluate_expression(expr)
        print(f"   Expression: '{expr}'")
        print(f"   Result: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Three Sum Closest
    print("3. Three Sum Closest:")
    closest_tests = [
        ([-1, 2, 1, -4], 1, 2),
        ([0, 0, 0], 1, 0)
    ]
    
    for nums, target, expected in closest_tests:
        result = three_sum_closest(nums.copy(), target)
        print(f"   Input: {nums}, Target: {target}")
        print(f"   Result: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()


def pattern_recognition_guide():
    """
    Guide for recognizing which patterns to use
    """
    print("\n=== PATTERN RECOGNITION GUIDE ===")
    
    patterns = {
        "Two Pointers": [
            "Pair sum problems (sorted array)",
            "Palindrome checking",
            "Removing duplicates from sorted array",
            "Container with most water"
        ],
        "Sliding Window": [
            "Longest/shortest substring with condition",
            "Maximum sum subarray of size k",
            "All anagrams in string",
            "Minimum window substring"
        ],
        "Hash Table": [
            "Frequency counting",
            "Complement lookup (two sum)",
            "Grouping by computed key",
            "Fast lookups needed"
        ],
        "Stack": [
            "Nested structures (parentheses)",
            "Next/previous greater element",
            "Expression evaluation",
            "DFS simulation"
        ],
        "Queue": [
            "Level-by-level processing",
            "First come first serve",
            "BFS traversal",
            "Sliding window maximum"
        ],
        "Linked List": [
            "Dynamic size needed",
            "Frequent insertion/deletion",
            "No random access required",
            "Cycle detection problems"
        ]
    }
    
    for pattern, use_cases in patterns.items():
        print(f"\n{pattern}:")
        for use_case in use_cases:
            print(f"  • {use_case}")


def week2_preview():
    """
    Preview of Week 2 topics and how they build on Week 1
    """
    print("\n=== WEEK 2 PREVIEW ===")
    
    print("Building on Week 1 foundations:")
    print("\nDay 7 - Binary Trees:")
    print("  • Uses stack/queue for traversals")
    print("  • Recursive patterns similar to linked lists")
    print("  • Hash tables for efficient tree problems")
    
    print("\nDay 8 - Binary Search Trees:")
    print("  • Two pointers concept in tree navigation")
    print("  • In-order traversal gives sorted sequence")
    print("  • Hash tables for fast lookups in tree problems")
    
    print("\nDay 9 - Binary Search:")
    print("  • Two pointers pattern on search space")
    print("  • Array problems with sorted property")
    print("  • String search problems")
    
    print("\nDay 10 - Tree Advanced:")
    print("  • Combines multiple Week 1 patterns")
    print("  • Stack for iterative traversals")
    print("  • Hash tables for parent-child relationships")
    
    print("\nKey Transitions:")
    print("  • Linear structures → Tree structures")
    print("  • Single pointer → Multiple pointers")
    print("  • Iterative → Recursive thinking")
    print("  • 1D problems → 2D problems")


if __name__ == "__main__":
    # Run comprehensive review
    week1_comprehensive_test()
    
    # Pattern recognition guide
    pattern_recognition_guide()
    
    # Week 2 preview
    week2_preview()
    
    print("\n" + "="*70)
    print("WEEK 1 COMPLETION - MASTER CHECKLIST:")
    print("="*70)
    print("✓ Arrays & Two Pointers: Pair problems, palindromes, containers")
    print("✓ Strings & Sliding Window: Substrings, pattern matching, anagrams")
    print("✓ Hash Tables: Frequency counting, fast lookups, grouping")
    print("✓ Linked Lists: Pointer manipulation, cycles, merging")
    print("✓ Stacks & Queues: LIFO/FIFO, monotonic patterns, nested structures")
    print("✓ Integration: Multiple patterns in complex problems")
    print("✓ Problem Recognition: When to use which pattern")
    
    print("\n" + "="*70)
    print("META INTERVIEW READINESS - WEEK 1:")
    print("="*70)
    print("🎯 Foundation Level: ACHIEVED")
    print("• Can solve fundamental data structure problems")
    print("• Understands time/space complexity trade-offs")
    print("• Recognizes common patterns and when to apply them")
    print("• Ready for intermediate tree and graph problems")
    
    print("\nNext Week Focus:")
    print("• Tree traversals and manipulation")
    print("• Binary search variations")
    print("• Graph fundamentals")
    print("• Advanced data structures")
    
    print("\nRecommended Practice:")
    print("• Review any weak areas from this week")
    print("• Practice pattern recognition on new problems")
    print("• Time yourself on Week 1 problems")
    print("• Start thinking about recursive problem-solving")
    
    print("\n🚀 READY FOR WEEK 2: TREES & BINARY SEARCH 🚀") 