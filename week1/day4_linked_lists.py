"""
=============================================================================
                        WEEK 1 - DAY 4: LINKED LISTS
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. LINKED LIST FUNDAMENTALS
   - Node structure: data + pointer to next node
   - Dynamic size (grows/shrinks during runtime)
   - Non-contiguous memory allocation
   - No direct access by index (must traverse from head)

2. TYPES OF LINKED LISTS
   - Singly linked: each node points to next
   - Doubly linked: each node has next and prev pointers
   - Circular linked: last node points back to first

3. LINKED LIST OPERATIONS & COMPLEXITY
   - Access: O(n) - must traverse from head
   - Search: O(n) - linear scan required
   - Insertion: O(1) if position known, O(n) to find position
   - Deletion: O(1) if node reference known, O(n) to find node

4. COMMON PATTERNS & TECHNIQUES
   - Dummy head node: simplifies edge cases
   - Two pointers: fast/slow for cycle detection, finding middle
   - Recursion vs iteration: both viable for most operations
   - In-place operations: reverse, merge without extra space

5. POINTER MANIPULATION PRINCIPLES
   - Always check for null pointers
   - Save references before breaking links
   - Draw diagrams to visualize pointer changes
   - Handle edge cases: empty list, single node

6. TRANSITION FROM HASH TABLES
   - Hash tables can store linked list nodes for fast lookup
   - Combine fast access with dynamic structure
   - Example: LRU cache uses hash table + doubly linked list

=============================================================================
"""


# Linked List Node Definition
class ListNode:
    """Standard linked list node structure"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        """String representation for debugging"""
        return f"ListNode({self.val})"


# Utility functions for testing
def create_linked_list(values):
    """Create linked list from list of values"""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head


def linked_list_to_list(head):
    """Convert linked list to Python list for easy testing"""
    result = []
    current = head
    visited = set()  # Detect cycles
    
    while current:
        if id(current) in visited:
            result.append(f"CYCLE_TO_{current.val}")
            break
        visited.add(id(current))
        result.append(current.val)
        current = current.next
    
    return result


# Problem 1: Reverse Linked List - Fundamental pointer manipulation
def reverse_list_iterative(head):
    """
    Reverse linked list iteratively
    
    Key technique: Three pointers (prev, current, next)
    This is the foundation of pointer manipulation
    
    Time: O(n), Space: O(1)
    """
    prev = None
    current = head
    
    while current:
        # Save next node before breaking link
        next_temp = current.next
        
        # Reverse the link
        current.next = prev
        
        # Move pointers forward
        prev = current
        current = next_temp
    
    return prev  # prev is new head


def reverse_list_recursive(head):
    """
    Reverse linked list recursively
    
    Demonstrates recursive thinking with linked lists
    
    Time: O(n), Space: O(n) due to recursion stack
    """
    # Base case
    if not head or not head.next:
        return head
    
    # Recursively reverse rest of list
    new_head = reverse_list_recursive(head.next)
    
    # Reverse current connection
    head.next.next = head
    head.next = None
    
    return new_head


# Problem 2: Merge Two Sorted Lists - Classic merge technique
def merge_two_lists(l1, l2):
    """
    Merge two sorted linked lists
    
    Technique: Dummy head to simplify edge cases
    This pattern appears in many linked list problems
    
    Time: O(n + m), Space: O(1)
    """
    # Create dummy head to simplify logic
    dummy = ListNode(0)
    current = dummy
    
    # Merge while both lists have nodes
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    # Attach remaining nodes
    current.next = l1 if l1 else l2
    
    return dummy.next


def merge_two_lists_recursive(l1, l2):
    """
    Recursive approach for merging
    
    Elegant but uses O(n) space for recursion
    
    Time: O(n + m), Space: O(n + m)
    """
    # Base cases
    if not l1:
        return l2
    if not l2:
        return l1
    
    # Choose smaller head and recurse
    if l1.val <= l2.val:
        l1.next = merge_two_lists_recursive(l1.next, l2)
        return l1
    else:
        l2.next = merge_two_lists_recursive(l1, l2.next)
        return l2


# Problem 3: Remove Nth Node From End - Two pointer technique
def remove_nth_from_end(head, n):
    """
    Remove nth node from end of list
    
    Two pointer technique: fast pointer gets n steps head start
    When fast reaches end, slow is at nth from end
    
    Time: O(L) where L is list length, Space: O(1)
    """
    # Use dummy head for edge cases (removing first node)
    dummy = ListNode(0)
    dummy.next = head
    
    fast = slow = dummy
    
    # Move fast pointer n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next
    
    # Move both pointers until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next
    
    # Remove the nth node
    slow.next = slow.next.next
    
    return dummy.next


# Problem 4: Linked List Cycle Detection - Floyd's algorithm
def has_cycle(head):
    """
    Detect if linked list has cycle
    
    Floyd's Cycle Detection (Tortoise and Hare):
    - Slow pointer moves 1 step, fast moves 2 steps
    - If cycle exists, fast will eventually meet slow
    
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True


def detect_cycle_start(head):
    """
    Find the node where cycle begins
    
    Extended Floyd's algorithm:
    1. Detect cycle with fast/slow pointers
    2. Move one pointer to head, keep other at meeting point
    3. Move both one step at a time until they meet
    
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect cycle
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Phase 2: Find cycle start
    start = head
    while start != slow:
        start = start.next
        slow = slow.next
    
    return start


# Problem 5: Merge k Sorted Lists - Divide and conquer + heap approaches
def merge_k_lists_divide_conquer(lists):
    """
    Merge k sorted linked lists using divide and conquer
    
    Strategy: Recursively merge pairs of lists
    
    Time: O(n log k) where n is total nodes, k is number of lists
    Space: O(log k) for recursion
    """
    if not lists:
        return None
    
    def merge_two_lists_helper(l1, l2):
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        current.next = l1 if l1 else l2
        return dummy.next
    
    def merge_helper(lists, start, end):
        if start == end:
            return lists[start]
        if start > end:
            return None
        
        mid = (start + end) // 2
        left = merge_helper(lists, start, mid)
        right = merge_helper(lists, mid + 1, end)
        
        return merge_two_lists_helper(left, right)
    
    return merge_helper(lists, 0, len(lists) - 1)


import heapq

def merge_k_lists_heap(lists):
    """
    Merge k sorted lists using min-heap
    
    Strategy: Use heap to always get minimum element
    
    Time: O(n log k), Space: O(k)
    """
    if not lists:
        return None
    
    # Initialize heap with first node from each list
    heap = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        
        # Add node to result
        current.next = node
        current = current.next
        
        # Add next node from same list to heap
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next


# ADVANCED PROBLEMS FOR DEEPER UNDERSTANDING

def reorder_list(head):
    """
    Reorder list: L0 → L1 → ... → Ln-1 → Ln becomes
                  L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...
    
    Strategy:
    1. Find middle of list
    2. Reverse second half
    3. Merge two halves alternately
    
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second_half = slow.next
    slow.next = None
    
    prev = None
    current = second_half
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    # Merge two halves
    first = head
    second = prev
    
    while second:
        first_next = first.next
        second_next = second.next
        
        first.next = second
        second.next = first_next
        
        first = first_next
        second = second_next


def add_two_numbers(l1, l2):
    """
    Add two numbers represented as linked lists (digits in reverse order)
    
    Example: (2 -> 4 -> 3) + (5 -> 6 -> 4) = (7 -> 0 -> 8)
             Represents: 342 + 465 = 807
    
    Time: O(max(m, n)), Space: O(max(m, n))
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        current.next = ListNode(digit)
        current = current.next
        
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next


def copy_random_list(head):
    """
    Deep copy linked list with random pointers
    
    Each node has next and random pointer
    
    Strategy:
    1. Create new nodes and interleave with original
    2. Set random pointers for new nodes
    3. Separate the two lists
    
    Time: O(n), Space: O(1) excluding output
    """
    if not head:
        return None
    
    # Step 1: Create new nodes interleaved with original
    current = head
    while current:
        new_node = ListNode(current.val)
        new_node.next = current.next
        current.next = new_node
        current = new_node.next
    
    # Step 2: Set random pointers for new nodes
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate the lists
    dummy = ListNode(0)
    new_current = dummy
    current = head
    
    while current:
        new_current.next = current.next
        current.next = current.next.next
        new_current = new_current.next
        current = current.next
    
    return dummy.next


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all linked list problems with comprehensive test cases
    """
    print("=== TESTING DAY 4 PROBLEMS ===\n")
    
    # Test Reverse Linked List
    print("1. Reverse Linked List:")
    test_cases = [
        [1, 2, 3, 4, 5],
        [1, 2],
        [1],
        []
    ]
    
    for values in test_cases:
        original = create_linked_list(values)
        head_iter = create_linked_list(values)
        head_rec = create_linked_list(values)
        
        result_iter = reverse_list_iterative(head_iter)
        result_rec = reverse_list_recursive(head_rec)
        
        expected = values[::-1] if values else []
        actual_iter = linked_list_to_list(result_iter)
        actual_rec = linked_list_to_list(result_rec)
        
        print(f"   Input: {values}")
        print(f"   Iterative: {actual_iter}")
        print(f"   Recursive: {actual_rec}")
        print(f"   Expected: {expected}")
        success = actual_iter == expected and actual_rec == expected
        print(f"   ✓ Correct" if success else f"   ✗ Wrong")
        print()
    
    # Test Merge Two Sorted Lists
    print("2. Merge Two Sorted Lists:")
    merge_tests = [
        ([1, 2, 4], [1, 3, 4], [1, 1, 2, 3, 4, 4]),
        ([], [], []),
        ([], [0], [0])
    ]
    
    for l1_vals, l2_vals, expected in merge_tests:
        l1 = create_linked_list(l1_vals)
        l2 = create_linked_list(l2_vals)
        
        result = merge_two_lists(l1, l2)
        actual = linked_list_to_list(result)
        
        print(f"   l1: {l1_vals}, l2: {l2_vals}")
        print(f"   Output: {actual}")
        print(f"   Expected: {expected}")
        print(f"   ✓ Correct" if actual == expected else f"   ✗ Wrong")
        print()
    
    # Test Remove Nth From End
    print("3. Remove Nth Node From End:")
    remove_tests = [
        ([1, 2, 3, 4, 5], 2, [1, 2, 3, 5]),
        ([1], 1, []),
        ([1, 2], 1, [1])
    ]
    
    for values, n, expected in remove_tests:
        head = create_linked_list(values)
        result = remove_nth_from_end(head, n)
        actual = linked_list_to_list(result)
        
        print(f"   Input: {values}, n: {n}")
        print(f"   Output: {actual}")
        print(f"   Expected: {expected}")
        print(f"   ✓ Correct" if actual == expected else f"   ✗ Wrong")
        print()
    
    # Test Cycle Detection
    print("4. Linked List Cycle Detection:")
    # Create cycle: 3 -> 2 -> 0 -> -4 -> (back to 2)
    head = create_linked_list([3, 2, 0, -4])
    # Create cycle by connecting last node to second node
    if head and head.next:
        current = head
        while current.next:
            current = current.next
        current.next = head.next  # Create cycle
        
        has_cycle_result = has_cycle(head)
        cycle_start = detect_cycle_start(head)
        
        print(f"   List with cycle at position 1")
        print(f"   Has cycle: {has_cycle_result}")
        print(f"   Cycle starts at value: {cycle_start.val if cycle_start else None}")
        print(f"   ✓ Correct" if has_cycle_result and cycle_start.val == 2 else f"   ✗ Wrong")
    
    # Test without cycle
    head_no_cycle = create_linked_list([1, 2, 3, 4])
    has_cycle_result = has_cycle(head_no_cycle)
    cycle_start = detect_cycle_start(head_no_cycle)
    
    print(f"   List without cycle: [1, 2, 3, 4]")
    print(f"   Has cycle: {has_cycle_result}")
    print(f"   Cycle start: {cycle_start}")
    print(f"   ✓ Correct" if not has_cycle_result and cycle_start is None else f"   ✗ Wrong")
    print()


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_pointer_manipulation():
    """
    Visual demonstration of pointer manipulation
    """
    print("\n=== POINTER MANIPULATION DEMONSTRATION ===")
    
    # Create simple list: 1 -> 2 -> 3
    head = create_linked_list([1, 2, 3])
    print("Original list: 1 -> 2 -> 3")
    
    print("\nReversing step by step:")
    prev = None
    current = head
    step = 1
    
    while current:
        print(f"\nStep {step}:")
        print(f"  prev: {prev.val if prev else None}")
        print(f"  current: {current.val}")
        print(f"  current.next: {current.next.val if current.next else None}")
        
        # Save next
        next_temp = current.next
        print(f"  Saving next: {next_temp.val if next_temp else None}")
        
        # Reverse link
        current.next = prev
        print(f"  After reversing: current.next = {prev.val if prev else None}")
        
        # Move pointers
        prev = current
        current = next_temp
        print(f"  Moving pointers: prev = {prev.val}, current = {current.val if current else None}")
        
        step += 1
    
    print(f"\nFinal result: {linked_list_to_list(prev)}")


def two_pointer_techniques():
    """
    Demonstrate various two-pointer techniques in linked lists
    """
    print("\n=== TWO POINTER TECHNIQUES ===")
    
    # Finding middle of list
    print("1. Finding Middle of List:")
    head = create_linked_list([1, 2, 3, 4, 5])
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    print(f"   List: [1, 2, 3, 4, 5]")
    print(f"   Middle: {slow.val}")
    
    # Finding nth from end
    print("\n2. Finding Nth from End:")
    head = create_linked_list([1, 2, 3, 4, 5])
    n = 2
    fast = slow = head
    
    # Move fast n steps ahead
    for _ in range(n):
        fast = fast.next
    
    # Move both until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next
    
    print(f"   List: [1, 2, 3, 4, 5]")
    print(f"   2nd from end: {slow.val}")


def memory_management_tips():
    """
    Important memory management considerations
    """
    print("\n=== MEMORY MANAGEMENT TIPS ===")
    
    print("1. Always check for null pointers before dereferencing")
    print("2. Save references before breaking links")
    print("3. In languages with manual memory management:")
    print("   - Free deleted nodes to prevent memory leaks")
    print("   - Be careful with dangling pointers")
    print("4. Python handles garbage collection automatically")
    print("5. Circular references can cause memory leaks in some scenarios")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Demonstrate pointer manipulation
    demonstrate_pointer_manipulation()
    
    # Show two-pointer techniques
    two_pointer_techniques()
    
    # Memory management tips
    memory_management_tips()
    
    print("\n" + "="*70)
    print("DAY 4 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. Master pointer manipulation: always save references first")
    print("2. Dummy head nodes simplify edge case handling")
    print("3. Two pointers: fast/slow for cycles, middle, nth from end")
    print("4. Floyd's algorithm: elegant cycle detection in O(1) space")
    print("5. Merge operations: fundamental for many complex problems")
    print("6. Recursive vs iterative: understand space trade-offs")
    print("7. Draw diagrams when solving pointer problems")
    print("\nTransition: Day 4→5 - From linked lists to stacks/queues")
    print("- Linked lists can implement stacks and queues")
    print("- Pointer manipulation skills transfer to tree problems")
    print("- Next: Linear data structures for ordering and sequencing")
    print("\nNext: Day 5 - Stacks & Queues") 