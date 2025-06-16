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


# Problem 1: Valid Parentheses - Classic stack problem
def is_valid_parentheses(s):
    """
    Check if parentheses/brackets are valid and properly nested
    
    Stack approach: Push opening brackets, pop and match closing brackets
    This is the fundamental stack pattern
    
    Time: O(n), Space: O(n)
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


# Problem 2: Daily Temperatures - Monotonic stack pattern
def daily_temperatures(temperatures):
    """
    Find number of days until warmer temperature for each day
    
    Monotonic stack: Maintain decreasing temperature stack
    When we find warmer temperature, pop all colder days and record distances
    
    Time: O(n), Space: O(n)
    """
    result = [0] * len(temperatures)
    stack = []  # Store indices
    
    for i, temp in enumerate(temperatures):
        # Pop all temperatures that are colder than current
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        
        stack.append(i)
    
    return result


# Problem 3: Largest Rectangle in Histogram - Advanced monotonic stack
def largest_rectangle_area(heights):
    """
    Find area of largest rectangle in histogram
    
    Monotonic stack approach:
    1. Maintain increasing heights in stack
    2. When decreasing height found, calculate rectangles
    3. Each popped height can extend back to previous smaller height
    
    Time: O(n), Space: O(n)
    """
    stack = []
    max_area = 0
    index = 0
    
    while index < len(heights):
        # If current bar is higher, push to stack
        if not stack or heights[index] >= heights[stack[-1]]:
            stack.append(index)
            index += 1
        else:
            # Pop and calculate area with popped bar as smallest
            top = stack.pop()
            area = (heights[top] * 
                   ((index - stack[-1] - 1) if stack else index))
            max_area = max(max_area, area)
    
    # Pop remaining bars and calculate area
    while stack:
        top = stack.pop()
        area = (heights[top] * 
               ((index - stack[-1] - 1) if stack else index))
        max_area = max(max_area, area)
    
    return max_area


# Problem 4: Implement Queue using Stacks - Structural understanding
class MyQueue:
    """
    Implement queue using two stacks
    
    Strategy: Use two stacks to reverse the order twice
    - input_stack: for enqueue operations
    - output_stack: for dequeue operations
    
    Amortized O(1) for all operations
    """
    
    def __init__(self):
        self.input_stack = []
        self.output_stack = []
    
    def push(self, x):
        """Add element to back of queue"""
        self.input_stack.append(x)
    
    def pop(self):
        """Remove element from front of queue"""
        self._move_to_output()
        return self.output_stack.pop()
    
    def peek(self):
        """Get front element"""
        self._move_to_output()
        return self.output_stack[-1]
    
    def empty(self):
        """Check if queue is empty"""
        return not self.input_stack and not self.output_stack
    
    def _move_to_output(self):
        """Move elements from input to output stack if needed"""
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())


# Problem 5: Next Greater Element - Monotonic stack pattern
def next_greater_element(nums1, nums2):
    """
    Find next greater element for each element in nums1 within nums2
    
    Approach:
    1. Use monotonic stack to find next greater for all elements in nums2
    2. Store results in hash map
    3. Look up results for nums1 elements
    
    Time: O(n + m), Space: O(n)
    """
    # Build next greater mapping for nums2
    next_greater = {}
    stack = []
    
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # Default to -1 for elements with no greater element
    for num in stack:
        next_greater[num] = -1
    
    # Build result for nums1
    return [next_greater[num] for num in nums1]


# ADVANCED PROBLEMS

def sliding_window_maximum(nums, k):
    """
    Find maximum in each sliding window of size k
    
    Deque approach: Maintain decreasing order of elements
    Front of deque always contains maximum of current window
    
    Time: O(n), Space: O(k)
    """
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements from back
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum to result (window size reached)
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


def evaluate_rpn(tokens):
    """
    Evaluate Reverse Polish Notation expression
    
    Stack approach: Push numbers, pop operands for operations
    
    Time: O(n), Space: O(n)
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            # Pop two operands
            b = stack.pop()
            a = stack.pop()
            
            # Perform operation
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            else:  # division
                # Handle negative division (truncate towards zero)
                result = int(a / b)
            
            stack.append(result)
        else:
            # Push number
            stack.append(int(token))
    
    return stack[0]


def decode_string(s):
    """
    Decode string with pattern k[encoded_string]
    
    Example: "3[a2[c]]" -> "accaccacc"
    
    Stack approach: Use stack to handle nested brackets
    
    Time: O(max_k * n), Space: O(max_k * n)
    """
    stack = []
    current_num = 0
    current_string = ""
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state to stack
            stack.append((current_string, current_num))
            current_string = ""
            current_num = 0
        elif char == ']':
            # Pop and decode
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