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


# =============================================================================
# INTEGRATION PROBLEM 1: LRU CACHE (MEDIUM) - 60 MIN
# =============================================================================

class LRUCache:
    """
    PROBLEM: LRU Cache
    
    Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
    
    Implement the LRUCache class:
    - LRUCache(int capacity) Initialize the LRU cache with positive size capacity
    - int get(int key) Return the value of the key if the key exists, otherwise return -1
    - void put(int key, int value) Update the value of the key if the key exists. 
      Otherwise, add the key-value pair to the cache. If the number of keys exceeds 
      the capacity from this operation, evict the least recently used key.
    
    The functions get and put must each run in O(1) average time complexity.
    
    CONSTRAINTS:
    - 1 <= capacity <= 3000
    - 0 <= key <= 10^4
    - 0 <= value <= 10^5
    - At most 2 * 10^5 calls will be made to get and put
    
    EXAMPLES:
    Example 1:
        Input: ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
               [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
        Output: [null, null, null, 1, null, -1, null, -1, 3, 4]
        
        Explanation:
        LRUCache lRUCache = new LRUCache(2);
        lRUCache.put(1, 1); // cache is {1=1}
        lRUCache.put(2, 2); // cache is {1=1, 2=2}
        lRUCache.get(1);    // return 1
        lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
        lRUCache.get(2);    // returns -1 (not found)
        lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
        lRUCache.get(1);    // return -1 (not found)
        lRUCache.get(3);    // return 3
        lRUCache.get(4);    // return 4
    
    APPROACH: Hash Table + Doubly Linked List
    
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


# =============================================================================
# INTEGRATION PROBLEM 2: DESIGN BROWSER HISTORY (MEDIUM) - 45 MIN
# =============================================================================

class BrowserHistory:
    """
    PROBLEM: Design Browser History
    
    You have a browser of one tab where you start on the homepage and you can visit 
    another url, get back in the history number of steps or move forward in the history 
    number of steps.
    
    Implement the BrowserHistory class:
    - BrowserHistory(string homepage) Initializes the object with the homepage of the browser
    - void visit(string url) Visits url from the current page. It clears up all the forward history
    - string back(int steps) Move steps back in history. Return the current url after moving back
    - string forward(int steps) Move steps forward in history. Return the current url after moving forward
    
    CONSTRAINTS:
    - 1 <= homepage.length <= 20
    - 1 <= url.length <= 20
    - 1 <= steps <= 100
    - homepage and url consist of '.' or lower case English letters
    - At most 5000 calls will be made to visit, back, and forward
    
    EXAMPLES:
    Example 1:
        Input:
        ["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
        [["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
        Output:
        [null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]
        
        Explanation:
        BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
        browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
        browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
        browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
        browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
        browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
        browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
        browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
        browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.
        browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com" return "google.com"
        browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com" return "leetcode.com"
    
    APPROACH: Array with Current Pointer
    
    Combines array-like structure with stack operations
    Demonstrates practical application of data structures
    
    TIME: O(1) for all operations, SPACE: O(n)
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


# =============================================================================
# INTEGRATION PROBLEM 3: DESIGN PHONE DIRECTORY (MEDIUM) - 45 MIN
# =============================================================================

class PhoneDirectory:
    """
    PROBLEM: Design Phone Directory
    
    Design a phone directory which supports the following operations:
    - get(): Provide a number which is not assigned to anyone
    - check(number): Check if a number is available or not
    - release(number): Recycle or release a number
    
    CONSTRAINTS:
    - 1 <= maxNumbers <= 10^4
    - 0 <= number < maxNumbers
    - At most 2 * 10^4 calls will be made to get, check, and release
    
    EXAMPLES:
    Example 1:
        Input: ["PhoneDirectory", "get", "get", "check", "get", "check", "release", "check"]
               [[3], [], [], [2], [], [2], [2], [2]]
        Output: [null, 0, 1, true, 2, false, null, true]
        
        Explanation:
        PhoneDirectory phoneDirectory = new PhoneDirectory(3);
        phoneDirectory.get();      // It can return any available phone number. Here we assume it returns 0.
        phoneDirectory.get();      // Assume it returns 1.
        phoneDirectory.check(2);   // The number 2 is available, so return true.
        phoneDirectory.get();      // It returns 2, the only number that is left.
        phoneDirectory.check(2);   // The number 2 is no longer available, so return false.
        phoneDirectory.release(2); // Release number 2 back to the pool.
        phoneDirectory.check(2);   // Number 2 is available again, return true.
    
    APPROACH: Set + Queue
    
    Combines set (fast lookup) with queue (FIFO order for available numbers)
    Shows integration of multiple data structures for efficiency
    
    TIME: O(1) for all operations, SPACE: O(n)
    """
    
    def __init__(self, maxNumbers):
        """Initialize with maxNumbers phone numbers"""
        self.available = set(range(maxNumbers))
        self.queue = deque(range(maxNumbers))
    
    def get(self):
        """Get an available phone number"""
        if not self.queue:
            return -1
        
        number = self.queue.popleft()
        self.available.remove(number)
        return number
    
    def check(self, number):
        """Check if number is available"""
        return number in self.available
    
    def release(self, number):
        """Release number back to available pool"""
        if number not in self.available:
            self.available.add(number)
            self.queue.append(number)


# =============================================================================
# INTEGRATION PROBLEM 4: EVALUATE EXPRESSION (HARD) - 60 MIN
# =============================================================================

def evaluate_expression(expression):
    """
    PROBLEM: Basic Calculator
    
    Given a string s representing a valid expression, implement a basic calculator 
    to evaluate it, and return the result of the evaluation.
    
    Note: You are not allowed to use any built-in function which evaluates strings 
    as mathematical expressions, such as eval().
    
    CONSTRAINTS:
    - 1 <= s.length <= 3 * 10^5
    - s consists of integers and operators ('+', '-', '*', '/') and spaces ' '
    - s represents a valid expression
    - All the integers in the expression are non-negative integers in the range [0, 2^31 - 1]
    - The answer is guaranteed to fit in a 32-bit integer
    
    EXAMPLES:
    Example 1:
        Input: s = "3+2*2"
        Output: 7
    
    Example 2:
        Input: s = " 3/2 "
        Output: 1
    
    Example 3:
        Input: s = " 3+5 / 2 "
        Output: 5
    
    APPROACH: Stack with Operator Precedence
    
    Use stack to handle operator precedence (* and / before + and -)
    Process expression left to right, handling high precedence immediately
    
    TIME: O(n), SPACE: O(n)
    """
    if not expression:
        return 0
    
    stack = []
    num = 0
    sign = '+'
    
    for i, char in enumerate(expression):
        if char.isdigit():
            num = num * 10 + int(char)
        
        if char in '+-*/' or i == len(expression) - 1:
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                stack.append(stack.pop() * num)
            elif sign == '/':
                # Handle negative division (truncate toward zero)
                prev = stack.pop()
                stack.append(int(prev / num))
            
            sign = char
            num = 0
    
    return sum(stack)


# =============================================================================
# INTEGRATION PROBLEM 5: HIT COUNTER (MEDIUM) - 45 MIN
# =============================================================================

class HitCounter:
    """
    PROBLEM: Design Hit Counter
    
    Design a hit counter which counts the number of hits received in the past 5 minutes (i.e., the past 300 seconds).
    
    Your system should accept a timestamp parameter (in seconds granularity), and you may assume that 
    calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing). 
    Several hits may arrive roughly at the same time.
    
    Implement the HitCounter class:
    - HitCounter() Initializes the object of the hit counter system
    - void hit(int timestamp) Records a hit that happened at timestamp (in seconds)
    - int getHits(int timestamp) Returns the number of hits in the past 5 minutes from timestamp
    
    CONSTRAINTS:
    - 1 <= timestamp <= 2 * 10^9
    - All the calls are being made to the system in chronological order
    - At most 300 calls will be made to hit and getHits
    
    EXAMPLES:
    Example 1:
        Input: ["HitCounter", "hit", "hit", "hit", "getHits", "hit", "getHits", "getHits"]
               [[], [1], [2], [3], [4], [300], [300], [301]]
        Output: [null, null, null, null, 3, null, 4, 3]
        
        Explanation:
        HitCounter hitCounter = new HitCounter();
        hitCounter.hit(1);       // hit at timestamp 1.
        hitCounter.hit(2);       // hit at timestamp 2.
        hitCounter.hit(3);       // hit at timestamp 3.
        hitCounter.getHits(4);   // get hits at timestamp 4, return 3.
        hitCounter.hit(300);     // hit at timestamp 300.
        hitCounter.getHits(300); // get hits at timestamp 300, return 4.
        hitCounter.getHits(301); // get hits at timestamp 301, return 3.
    
    APPROACH: Circular Array with Timestamps
    
    Use circular array to store hit counts for each second in 300-second window
    Efficiently handles the sliding window requirement
    
    TIME: O(1) for both operations, SPACE: O(300) = O(1)
    """
    
    def __init__(self):
        """Initialize hit counter with 300-second window"""
        self.times = [0] * 300  # timestamps
        self.hits = [0] * 300   # hit counts
    
    def hit(self, timestamp):
        """Record a hit at given timestamp"""
        index = timestamp % 300
        
        if self.times[index] != timestamp:
            # New timestamp, reset count
            self.times[index] = timestamp
            self.hits[index] = 1
        else:
            # Same timestamp, increment count
            self.hits[index] += 1
    
    def get_hits(self, timestamp):
        """Get hits in past 300 seconds"""
        total = 0
        
        for i in range(300):
            # Only count hits within 300 seconds
            if timestamp - self.times[i] < 300:
                total += self.hits[i]
        
        return total


# =============================================================================
# INTEGRATION PROBLEM 6: FIND ANAGRAM GROUPS WITH INDICES (MEDIUM) - 45 MIN
# =============================================================================

def find_anagram_groups_with_indices(words):
    """
    PROBLEM: Group Anagrams with Indices
    
    Given an array of strings strs, group the anagrams together and return both 
    the grouped anagrams and their original indices.
    
    CONSTRAINTS:
    - 1 <= strs.length <= 10^4
    - 0 <= strs[i].length <= 100
    - strs[i] consists of lowercase English letters
    
    EXAMPLES:
    Example 1:
        Input: strs = ["eat","tea","tan","ate","nat","bat"]
        Output: {
            "aet": [(0, "eat"), (1, "tea"), (3, "ate")],
            "ant": [(2, "tan"), (4, "nat")],
            "abt": [(5, "bat")]
        }
    
    APPROACH: Hash Map with Sorted Keys + Index Tracking
    
    Combines anagram grouping with index preservation
    Demonstrates data structure integration for complex requirements
    
    TIME: O(n * m log m), SPACE: O(n * m)
    """
    anagram_groups = defaultdict(list)
    
    for i, word in enumerate(words):
        # Sort characters to create anagram key
        key = ''.join(sorted(word))
        anagram_groups[key].append((i, word))
    
    return dict(anagram_groups)


# =============================================================================
# INTEGRATION PROBLEM 7: MERGE K SORTED ARRAYS (HARD) - 60 MIN
# =============================================================================

def merge_k_sorted_arrays(arrays):
    """
    PROBLEM: Merge k Sorted Arrays
    
    Given k sorted arrays, merge them into one sorted array.
    
    CONSTRAINTS:
    - k == arrays.length
    - 0 <= k <= 10^4
    - 0 <= arrays[i].length <= 500
    - -10^4 <= arrays[i][j] <= 10^4
    - arrays[i] is sorted in ascending order
    
    EXAMPLES:
    Example 1:
        Input: arrays = [[1,4,5],[1,3,4],[2,6]]
        Output: [1,1,2,3,4,4,5,6]
    
    Example 2:
        Input: arrays = []
        Output: []
    
    Example 3:
        Input: arrays = [[]]
        Output: []
    
    APPROACH: Min Heap
    
    Use min heap to efficiently get the smallest element among all arrays
    Demonstrates heap usage for merging multiple sorted sequences
    
    TIME: O(n log k), SPACE: O(k) where n = total elements, k = number of arrays
    """
    if not arrays:
        return []
    
    result = []
    heap = []
    
    # Initialize heap with first element from each non-empty array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))  # (value, array_index, element_index)
    
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same array if exists
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))
    
    return result


# =============================================================================
# INTEGRATION PROBLEM 8: LONGEST VALID PARENTHESES (HARD) - 60 MIN
# =============================================================================

def longest_valid_parentheses(s):
    """
    PROBLEM: Longest Valid Parentheses
    
    Given a string containing just the characters '(' and ')', find the length 
    of the longest valid (well-formed) parentheses substring.
    
    CONSTRAINTS:
    - 0 <= s.length <= 3 * 10^4
    - s[i] is '(' or ')'
    
    EXAMPLES:
    Example 1:
        Input: s = "(()"
        Output: 2
        Explanation: The longest valid parentheses substring is "()"
    
    Example 2:
        Input: s = ")()())"
        Output: 4
        Explanation: The longest valid parentheses substring is "()()"
    
    Example 3:
        Input: s = ""
        Output: 0
    
    APPROACH: Stack with Index Tracking
    
    Use stack to track indices of unmatched parentheses
    Calculate lengths of valid substrings between unmatched positions
    
    TIME: O(n), SPACE: O(n)
    """
    stack = [-1]  # Initialize with -1 to handle edge cases
    max_length = 0
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:  # char == ')'
            stack.pop()
            
            if not stack:
                # No matching '(', push current index as base
                stack.append(i)
            else:
                # Calculate length of current valid substring
                current_length = i - stack[-1]
                max_length = max(max_length, current_length)
    
    return max_length


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