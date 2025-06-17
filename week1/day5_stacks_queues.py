"""
=============================================================================
                        WEEK 1 - DAY 5: STACKS & QUEUES
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. STACK FUNDAMENTALS (LIFO - Last In, First Out)
   - Push: Add element to top - O(1)
   - Pop: Remove element from top - O(1)
   - Peek/Top: View top element without removing - O(1)
   - IsEmpty: Check if stack is empty - O(1)

2. QUEUE FUNDAMENTALS (FIFO - First In, First Out)
   - Enqueue: Add element to rear - O(1)
   - Dequeue: Remove element from front - O(1)
   - Front: View front element - O(1)
   - IsEmpty: Check if queue is empty - O(1)

3. IMPLEMENTATION OPTIONS
   - Array-based: Fixed size, simple implementation
   - Linked list-based: Dynamic size, more memory overhead
   - Python: list for stack, collections.deque for queue

4. STACK PATTERNS
   - Parentheses matching: use stack to track opening brackets
   - Function call stack: recursion simulation
   - Expression evaluation: infix to postfix conversion
   - Monotonic stack: maintain increasing/decreasing order
   - Histogram problems: largest rectangle using stack

5. QUEUE PATTERNS
   - BFS traversal: level-by-level exploration
   - Task scheduling: process in order of arrival
   - Sliding window maximum: deque with monotonic property
   - Implementing stack using queues

6. ADVANCED STRUCTURES
   - Deque (double-ended queue): operations at both ends
   - Priority queue: elements have priority (heap-based)
   - Circular queue: efficient use of array space

7. TRANSITION FROM LINKED LISTS
   - Stacks/queues can be implemented using linked lists
   - Stack: insert/delete at head
   - Queue: insert at tail, delete at head
   - Understanding helps with tree traversals next week

=============================================================================
"""

from collections import deque
import heapq


# STACK IMPLEMENTATIONS

class ArrayStack:
    """Stack implementation using Python list"""
    
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items in stack"""
        return len(self.items)


class LinkedListStack:
    """Stack implementation using linked list"""
    
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
    
    def __init__(self):
        self.head = None
        self._size = 0
    
    def push(self, item):
        """Add item to top of stack"""
        new_node = self.Node(item)
        new_node.next = self.head
        self.head = new_node
        self._size += 1
    
    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        data = self.head.data
        self.head = self.head.next
        self._size -= 1
        return data
    
    def peek(self):
        """Return top item without removing"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.head.data
    
    def is_empty(self):
        """Check if stack is empty"""
        return self.head is None
    
    def size(self):
        """Return number of items in stack"""
        return self._size


# QUEUE IMPLEMENTATIONS

class ArrayQueue:
    """Queue implementation using circular array"""
    
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = 0
        self.rear = 0
        self.count = 0
    
    def enqueue(self, item):
        """Add item to rear of queue"""
        if self.count == self.capacity:
            raise OverflowError("Queue is full")
        
        self.items[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
        self.count += 1
    
    def dequeue(self):
        """Remove and return front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.items[self.front]
        self.items[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.count -= 1
        return item
    
    def peek(self):
        """Return front item without removing"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[self.front]
    
    def is_empty(self):
        """Check if queue is empty"""
        return self.count == 0
    
    def size(self):
        """Return number of items in queue"""
        return self.count


# =============================================================================
# PROBLEM 1: VALID PARENTHESES (EASY) - 30 MIN
# =============================================================================

def is_valid_parentheses(s):
    """
    PROBLEM: Valid Parentheses
    
    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', 
    determine if the input string is valid.
    
    An input string is valid if:
    1. Open brackets must be closed by the same type of brackets
    2. Open brackets must be closed in the correct order
    3. Every close bracket has a corresponding open bracket of the same type
    
    CONSTRAINTS:
    - 1 <= s.length <= 10^4
    - s consists of parentheses only '()[]{}'
    
    EXAMPLES:
    Example 1:
        Input: s = "()"
        Output: true
    
    Example 2:
        Input: s = "()[]{}"
        Output: true
    
    Example 3:
        Input: s = "(]"
        Output: false
    
    APPROACH: Stack
    
    Use stack to track opening brackets. When closing bracket is found,
    check if it matches the most recent opening bracket.
    
    TIME: O(n), SPACE: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    return not stack


# =============================================================================
# PROBLEM 2: DAILY TEMPERATURES (MEDIUM) - 45 MIN
# =============================================================================

def daily_temperatures(temperatures):
    """
    PROBLEM: Daily Temperatures
    
    Given an array of integers temperatures represents the daily temperatures, return 
    an array answer such that answer[i] is the number of days you have to wait after 
    the ith day to get a warmer temperature. If there is no future day for which this 
    is possible, keep answer[i] == 0 instead.
    
    CONSTRAINTS:
    - 1 <= temperatures.length <= 10^5
    - 30 <= temperatures[i] <= 100
    
    EXAMPLES:
    Example 1:
        Input: temperatures = [73,74,75,71,69,72,76,73]
        Output: [1,1,4,2,1,1,0,0]
        Explanation: 
        - Day 0: Next warmer day is day 1 (74 > 73), so wait 1 day
        - Day 1: Next warmer day is day 2 (75 > 74), so wait 1 day
        - Day 2: Next warmer day is day 6 (76 > 75), so wait 4 days
    
    Example 2:
        Input: temperatures = [30,40,50,60]
        Output: [1,1,1,0]
    
    Example 3:
        Input: temperatures = [30,60,90]
        Output: [1,1,0]
    
    APPROACH: Monotonic Stack
    
    Use stack to maintain indices of temperatures in decreasing order.
    When we find a warmer temperature, pop from stack and calculate days.
    
    TIME: O(n), SPACE: O(n)
    """
    result = [0] * len(temperatures)
    stack = []  # Stack stores indices
    
    for i, temp in enumerate(temperatures):
        # Pop indices with cooler temperatures
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        
        # Push current index
        stack.append(i)
    
    return result


# =============================================================================
# PROBLEM 3: LARGEST RECTANGLE IN HISTOGRAM (HARD) - 60 MIN
# =============================================================================

def largest_rectangle_area(heights):
    """
    PROBLEM: Largest Rectangle in Histogram
    
    Given an array of integers heights representing the histogram's bar height where 
    the width of each bar is 1, return the area of the largest rectangle in the histogram.
    
    CONSTRAINTS:
    - 1 <= heights.length <= 10^5
    - 0 <= heights[i] <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: heights = [2,1,5,6,2,3]
        Output: 10
        Explanation: The largest rectangle has area = 10 units (width=2, height=5)
    
    Example 2:
        Input: heights = [2,4]
        Output: 4
    
    APPROACH: Monotonic Stack
    
    Use stack to maintain indices of bars in increasing height order.
    When we find a shorter bar, calculate area using previous bars.
    
    TIME: O(n), SPACE: O(n)
    """
    stack = []
    max_area = 0
    
    for i, h in enumerate(heights):
        # Process bars that are taller than current
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            # Width is distance between current position and previous element in stack
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    # Process remaining bars in stack
    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, height * width)
    
    return max_area


# =============================================================================
# PROBLEM 4: IMPLEMENT QUEUE USING STACKS (EASY) - 30 MIN
# =============================================================================

class MyQueue:
    """
    PROBLEM: Implement Queue using Stacks
    
    Implement a first in first out (FIFO) queue using only two stacks. The implemented 
    queue should support all the functions of a normal queue (push, peek, pop, and empty).
    
    Implement the MyQueue class:
    - MyQueue() Initializes the queue object
    - void push(int x) Pushes element x to the back of the queue
    - int pop() Removes the element from the front of the queue and returns it
    - int peek() Returns the element at the front of the queue
    - boolean empty() Returns true if the queue is empty, false otherwise
    
    CONSTRAINTS:
    - 1 <= x <= 9
    - At most 100 calls will be made to push, pop, peek, and empty
    - All the calls to pop and peek are valid
    
    EXAMPLES:
    Example 1:
        Input: ["MyQueue", "push", "push", "peek", "pop", "empty"]
               [[], [1], [2], [], [], []]
        Output: [null, null, null, 1, 1, false]
        
        Explanation:
        MyQueue myQueue = new MyQueue();
        myQueue.push(1); // queue is: [1]
        myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
        myQueue.peek(); // return 1
        myQueue.pop(); // return 1, queue is [2]
        myQueue.empty(); // return false
    
    APPROACH: Two Stacks (Input and Output)
    
    Use two stacks: input for push operations, output for pop/peek operations.
    Transfer elements from input to output when output is empty.
    
    TIME: O(1) amortized for all operations, SPACE: O(n)
    """
    
    def __init__(self):
        """Initialize the queue with two stacks"""
        self.input_stack = []
        self.output_stack = []

    def push(self, x):
        """Push element x to the back of queue"""
        self.input_stack.append(x)

    def pop(self):
        """Remove element from front of queue and return it"""
        self._move_to_output()
        return self.output_stack.pop()

    def peek(self):
        """Get the front element"""
        self._move_to_output()
        return self.output_stack[-1]

    def empty(self):
        """Return whether the queue is empty"""
        return not self.input_stack and not self.output_stack

    def _move_to_output(self):
        """Move elements from input to output stack if output is empty"""
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())


# =============================================================================
# PROBLEM 5: NEXT GREATER ELEMENT I (EASY) - 30 MIN
# =============================================================================

def next_greater_element(nums1, nums2):
    """
    PROBLEM: Next Greater Element I
    
    The next greater element of some element x in an array is the first greater element 
    that is to the right of x in the same array.
    
    You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is 
    a subset of nums2.
    
    For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and 
    determine the next greater element of nums2[j] in nums2. If there is no next greater 
    element, then the answer for this query is -1.
    
    Return an array ans of length nums1.length such that ans[i] is the next greater 
    element as described above.
    
    CONSTRAINTS:
    - 1 <= nums1.length <= nums2.length <= 1000
    - 0 <= nums1[i], nums2[i] <= 10^4
    - All integers in nums1 and nums2 are unique
    - All the integers of nums1 also appear in nums2
    
    EXAMPLES:
    Example 1:
        Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
        Output: [-1,3,-1]
        Explanation: 
        - For number 4 in the first array, there is no next greater number in the second array, so output -1
        - For number 1 in the first array, the next greater number in the second array is 3
        - For number 2 in the first array, there is no next greater number in the second array, so output -1
    
    Example 2:
        Input: nums1 = [2,4], nums2 = [1,2,3,4]
        Output: [3,-1]
    
    APPROACH: Monotonic Stack + Hash Map
    
    1. Use monotonic stack to find next greater element for each element in nums2
    2. Use hash map to store the results
    3. Build result array for nums1 using the hash map
    
    TIME: O(n + m), SPACE: O(n)
    """
    # Build next greater element mapping for nums2
    next_greater = {}
    stack = []
    
    for num in nums2:
        # Pop elements that are smaller than current
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # Elements remaining in stack have no next greater element
    while stack:
        next_greater[stack.pop()] = -1
    
    # Build result for nums1
    return [next_greater[num] for num in nums1]


# =============================================================================
# PROBLEM 6: SLIDING WINDOW MAXIMUM (HARD) - 60 MIN
# =============================================================================

def sliding_window_maximum(nums, k):
    """
    PROBLEM: Sliding Window Maximum
    
    You are given an array of integers nums, there is a sliding window of size k which 
    is moving from the very left of the array to the very right. You can only see the k 
    numbers in the window. Each time the sliding window moves right by one position.
    
    Return the max sliding window.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    - 1 <= k <= nums.length
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
        Output: [3,3,5,5,6,7]
        Explanation: 
        Window position                Max
        ---------------               -----
        [1  3  -1] -3  5  3  6  7       3
         1 [3  -1  -3] 5  3  6  7       3
         1  3 [-1  -3  5] 3  6  7       5
         1  3  -1 [-3  5  3] 6  7       5
         1  3  -1  -3 [5  3  6] 7       6
         1  3  -1  -3  5 [3  6  7]      7
    
    Example 2:
        Input: nums = [1], k = 1
        Output: [1]
    
    APPROACH: Monotonic Deque
    
    Use deque to maintain indices of array elements in decreasing order of their values.
    The front of deque always contains the index of maximum element in current window.
    
    TIME: O(n), SPACE: O(k)
    """
    if not nums or k == 0:
        return []
    
    result = []
    window = deque()  # Store indices
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while window and window[0] <= i - k:
            window.popleft()
        
        # Remove indices with smaller values (maintain decreasing order)
        while window and nums[window[-1]] < nums[i]:
            window.pop()
        
        # Add current index
        window.append(i)
        
        # Add maximum to result (front of deque)
        if i >= k - 1:
            result.append(nums[window[0]])
    
    return result


# =============================================================================
# PROBLEM 7: EVALUATE REVERSE POLISH NOTATION (MEDIUM) - 45 MIN
# =============================================================================

def evaluate_rpn(tokens):
    """
    PROBLEM: Evaluate Reverse Polish Notation
    
    Evaluate the value of an arithmetic expression in Reverse Polish Notation.
    
    Valid operators are +, -, *, and /. Each operand may be an integer or another expression.
    
    Note that division between two integers should truncate toward zero.
    
    It is guaranteed that the given RPN expression is always valid.
    
    CONSTRAINTS:
    - 1 <= tokens.length <= 10^4
    - tokens[i] is either an operator: "+", "-", "*", or "/", or an integer in the range [-200, 200]
    
    EXAMPLES:
    Example 1:
        Input: tokens = ["2","1","+","3","*"]
        Output: 9
        Explanation: ((2 + 1) * 3) = 9
    
    Example 2:
        Input: tokens = ["4","13","5","/","+"]
        Output: 6
        Explanation: (4 + (13 / 5)) = 6
    
    Example 3:
        Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
        Output: 22
        Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5 = 22
    
    APPROACH: Stack
    
    Use stack to store operands. When operator is encountered,
    pop two operands, apply operation, and push result back.
    
    TIME: O(n), SPACE: O(n)
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            # Pop two operands (order matters for - and /)
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                # Truncate toward zero
                result = int(a / b)
            
            stack.append(result)
        else:
            # Operand
            stack.append(int(token))
    
    return stack[0]


# =============================================================================
# PROBLEM 8: DECODE STRING (MEDIUM) - 45 MIN
# =============================================================================

def decode_string(s):
    """
    PROBLEM: Decode String
    
    Given an encoded string, return its decoded string.
    
    The encoding rule is: k[encoded_string], where the encoded_string inside the square 
    brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.
    
    You may assume that the input string is always valid; there are no extra white spaces, 
    square brackets are well-formed, etc. Furthermore, you may assume that the original data 
    does not contain any digits and that digits are only for those repeat numbers, k.
    
    CONSTRAINTS:
    - 1 <= s.length <= 30
    - s consists of lowercase English letters, digits, and square brackets '[]'
    - s is guaranteed to be a valid input
    - All the integers in s are in the range [1, 300]
    
    EXAMPLES:
    Example 1:
        Input: s = "3[a]2[bc]"
        Output: "aaabcbc"
    
    Example 2:
        Input: s = "2[abc]3[cd]ef"
        Output: "abcabccdcdcdef"
    
    Example 3:
        Input: s = "abc3[cd]xyz"
        Output: "abccdcdcdxyz"
    
    APPROACH: Stack
    
    Use stack to handle nested brackets. Store count and string separately.
    When ']' is encountered, pop and repeat the string.
    
    TIME: O(n), SPACE: O(n)
    """
    stack = []
    current_string = ""
    current_num = 0
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state to stack
            stack.append((current_string, current_num))
            current_string = ""
            current_num = 0
        elif char == ']':
            # Pop and repeat
            prev_string, num = stack.pop()
            current_string = prev_string + current_string * num
        else:
            # Regular character
            current_string += char
    
    return current_string


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all stack and queue problems with comprehensive test cases
    """
    print("=== TESTING DAY 5 PROBLEMS ===\n")
    
    # Test Stack Implementation
    print("1. Stack Implementation Tests:")
    array_stack = ArrayStack()
    linked_stack = LinkedListStack()
    
    # Test both implementations
    for name, stack in [("Array", array_stack), ("LinkedList", linked_stack)]:
        print(f"   {name} Stack:")
        
        # Test operations
        stack.push(1)
        stack.push(2)
        stack.push(3)
        
        print(f"     After pushing 1,2,3: size = {stack.size()}")
        print(f"     Peek: {stack.peek()}")
        print(f"     Pop: {stack.pop()}")
        print(f"     Pop: {stack.pop()}")
        print(f"     Size after 2 pops: {stack.size()}")
        print(f"     Is empty: {stack.is_empty()}")
        print()
    
    # Test Valid Parentheses
    print("2. Valid Parentheses:")
    paren_tests = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True)
    ]
    
    for s, expected in paren_tests:
        result = is_valid_parentheses(s)
        print(f"   Input: '{s}'")
        print(f"   Output: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Daily Temperatures
    print("3. Daily Temperatures:")
    temp_tests = [
        ([73, 74, 75, 71, 69, 72, 76, 73], [1, 1, 4, 2, 1, 1, 0, 0]),
        ([30, 40, 50, 60], [1, 1, 1, 0]),
        ([30, 60, 90], [1, 1, 0])
    ]
    
    for temperatures, expected in temp_tests:
        result = daily_temperatures(temperatures)
        print(f"   Input: {temperatures}")
        print(f"   Output: {result}")
        print(f"   Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Largest Rectangle in Histogram
    print("4. Largest Rectangle in Histogram:")
    histogram_tests = [
        ([2, 1, 5, 6, 2, 3], 10),
        ([2, 4], 4),
        ([1, 1], 2)
    ]
    
    for heights, expected in histogram_tests:
        result = largest_rectangle_area(heights)
        print(f"   Input: {heights}")
        print(f"   Output: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Queue using Stacks
    print("5. Queue using Stacks:")
    queue = MyQueue()
    operations = [
        ("push", 1), ("push", 2), ("peek", None), 
        ("pop", None), ("empty", None)
    ]
    expected_results = [None, None, 1, 1, False]
    
    print("   Operations: push(1), push(2), peek(), pop(), empty()")
    results = []
    for op, val in operations:
        if op == "push":
            queue.push(val)
            results.append(None)
        elif op == "pop":
            results.append(queue.pop())
        elif op == "peek":
            results.append(queue.peek())
        elif op == "empty":
            results.append(queue.empty())
    
    print(f"   Results: {results}")
    print(f"   Expected: {expected_results}")
    print(f"   ✓ Correct" if results == expected_results else f"   ✗ Wrong")
    print()


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_monotonic_stack():
    """
    Visual demonstration of monotonic stack pattern
    """
    print("\n=== MONOTONIC STACK DEMONSTRATION ===")
    
    print("Finding next greater element for [2, 1, 2, 4, 3, 1]:")
    nums = [2, 1, 2, 4, 3, 1]
    stack = []
    result = [-1] * len(nums)
    
    for i, num in enumerate(nums):
        print(f"\nStep {i + 1}: Processing {num} at index {i}")
        print(f"  Stack before: {stack}")
        
        # Pop smaller elements
        while stack and nums[stack[-1]] < num:
            prev_idx = stack.pop()
            result[prev_idx] = num
            print(f"  Found next greater for index {prev_idx} (value {nums[prev_idx]}): {num}")
        
        stack.append(i)
        print(f"  Stack after: {stack}")
        print(f"  Current result: {result}")
    
    print(f"\nFinal result: {result}")
    print("Meaning: [4, 2, 4, -1, -1, -1]")


def stack_vs_recursion():
    """
    Show relationship between stack and recursion
    """
    print("\n=== STACK VS RECURSION ===")
    
    def factorial_recursive(n):
        if n <= 1:
            return 1
        return n * factorial_recursive(n - 1)
    
    def factorial_iterative(n):
        stack = []
        # Simulate function calls
        while n > 1:
            stack.append(n)
            n -= 1
        
        result = 1
        while stack:
            result *= stack.pop()
        
        return result
    
    n = 5
    rec_result = factorial_recursive(n)
    iter_result = factorial_iterative(n)
    
    print(f"Factorial of {n}:")
    print(f"  Recursive: {rec_result}")
    print(f"  Using explicit stack: {iter_result}")
    print(f"  Both use stack - recursive uses call stack, iterative uses explicit stack")


def queue_applications():
    """
    Demonstrate common queue applications
    """
    print("\n=== QUEUE APPLICATIONS ===")
    
    print("1. BFS Level-by-level traversal simulation:")
    # Simulate BFS on tree: 1 -> [2,3] -> [4,5,6,7]
    queue = deque([1])
    level = 0
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            
            # Add children (simulated)
            if node == 1:
                queue.extend([2, 3])
            elif node == 2:
                queue.extend([4, 5])
            elif node == 3:
                queue.extend([6, 7])
        
        print(f"   Level {level}: {current_level}")
        level += 1
        
        if level > 2:  # Limit for demo
            break
    
    print("\n2. Task Scheduling simulation:")
    task_queue = deque(['Task A', 'Task B', 'Task C'])
    completed = []
    
    while task_queue:
        current_task = task_queue.popleft()
        completed.append(current_task)
        print(f"   Processing: {current_task}")
    
    print(f"   Completed in order: {completed}")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Demonstrate monotonic stack
    demonstrate_monotonic_stack()
    
    # Stack vs recursion
    stack_vs_recursion()
    
    # Queue applications
    queue_applications()
    
    print("\n" + "="*70)
    print("DAY 5 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. Stack (LIFO): Perfect for nested structures, function calls")
    print("2. Queue (FIFO): Ideal for level-by-level processing, task scheduling")
    print("3. Monotonic stack: Powerful pattern for next/previous greater/smaller")
    print("4. Stack applications: parentheses, expression evaluation, DFS")
    print("5. Queue applications: BFS, sliding window, scheduling")
    print("6. Implementation trade-offs: array vs linked list")
    print("7. Deque: Flexible structure supporting both stack and queue operations")
    print("\nTransition: Day 5→6 - Review and integration")
    print("- Stacks/queues support tree and graph traversals")
    print("- Review all Week 1 patterns and their connections")
    print("- Prepare for Week 2: Trees and advanced data structures")
    print("\nNext: Day 6 - Week 1 Review & Mock Interview") 