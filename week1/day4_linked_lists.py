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


# =============================================================================
# PROBLEM 1: REVERSE LINKED LIST (EASY) - 30 MIN
# =============================================================================

def reverse_list_iterative(head):
    """
    PROBLEM: Reverse Linked List
    
    Given the head of a singly linked list, reverse the list, and return the reversed list.
    
    CONSTRAINTS:
    - The number of nodes in the list is the range [0, 5000]
    - -5000 <= Node.val <= 5000
    
    EXAMPLES:
    Example 1:
        Input: head = [1,2,3,4,5]
        Output: [5,4,3,2,1]
    
    Example 2:
        Input: head = [1,2]
        Output: [2,1]
    
    Example 3:
        Input: head = []
        Output: []
    
    APPROACH: Iterative with Three Pointers
    
    Key technique: Three pointers (prev, current, next)
    This is the foundation of pointer manipulation
    
    TIME: O(n), SPACE: O(1)
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
    APPROACH: Recursive
    
    Demonstrates recursive thinking with linked lists
    
    TIME: O(n), SPACE: O(n) due to recursion stack
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


# =============================================================================
# PROBLEM 2: MERGE TWO SORTED LISTS (EASY) - 30 MIN
# =============================================================================

def merge_two_lists(l1, l2):
    """
    PROBLEM: Merge Two Sorted Lists
    
    You are given the heads of two sorted linked lists list1 and list2.
    
    Merge the two lists in a one sorted list. The list should be made by splicing 
    together the nodes of the first two lists.
    
    Return the head of the merged linked list.
    
    CONSTRAINTS:
    - The number of nodes in both lists is in the range [0, 50]
    - -100 <= Node.val <= 100
    - Both list1 and list2 are sorted in non-decreasing order
    
    EXAMPLES:
    Example 1:
        Input: list1 = [1,2,4], list2 = [1,3,4]
        Output: [1,1,2,3,4,4]
    
    Example 2:
        Input: list1 = [], list2 = []
        Output: []
    
    Example 3:
        Input: list1 = [], list2 = [0]
        Output: [0]
    
    APPROACH: Dummy Head with Two Pointers
    
    Technique: Dummy head to simplify edge cases
    This pattern appears in many linked list problems
    
    TIME: O(n + m), SPACE: O(1)
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
    APPROACH: Recursive
    
    Elegant but uses O(n) space for recursion
    
    TIME: O(n + m), SPACE: O(n + m)
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


# =============================================================================
# PROBLEM 3: REMOVE NTH NODE FROM END OF LIST (MEDIUM) - 45 MIN
# =============================================================================

def remove_nth_from_end(head, n):
    """
    PROBLEM: Remove Nth Node From End of List
    
    Given the head of a linked list, remove the nth node from the end of the list 
    and return its head.
    
    CONSTRAINTS:
    - The number of nodes in the list is sz
    - 1 <= sz <= 30
    - 0 <= Node.val <= 100
    - 1 <= n <= sz
    
    EXAMPLES:
    Example 1:
        Input: head = [1,2,3,4,5], n = 2
        Output: [1,2,3,5]
        Explanation: Remove the 2nd node from the end (node with value 4)
    
    Example 2:
        Input: head = [1], n = 1
        Output: []
    
    Example 3:
        Input: head = [1,2], n = 1
        Output: [1]
    
    APPROACH: Two Pointers (Fast and Slow)
    
    Two pointer technique: fast pointer gets n steps head start
    When fast reaches end, slow is at the node before target
    
    TIME: O(n), SPACE: O(1)
    """
    # Create dummy head to handle edge cases
    dummy = ListNode(0)
    dummy.next = head
    
    # Initialize two pointers
    fast = slow = dummy
    
    # Move fast pointer n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next
    
    # Move both pointers until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next
    
    # Remove the nth node from end
    slow.next = slow.next.next
    
    return dummy.next


# =============================================================================
# PROBLEM 4: LINKED LIST CYCLE (EASY) - 30 MIN
# =============================================================================

def has_cycle(head):
    """
    PROBLEM: Linked List Cycle
    
    Given head, the head of a linked list, determine if the linked list has a cycle in it.
    
    There is a cycle in a linked list if there is some node in the list that can be 
    reached again by continuously following the next pointer.
    
    Return true if there is a cycle in the linked list. Otherwise, return false.
    
    CONSTRAINTS:
    - The number of the nodes in the list is in the range [0, 10^4]
    - -10^5 <= Node.val <= 10^5
    
    EXAMPLES:
    Example 1:
        Input: head = [3,2,0,-4], pos = 1 (tail connects to node index 1)
        Output: true
        Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed)
    
    Example 2:
        Input: head = [1,2], pos = 0 (tail connects to node index 0)
        Output: true
    
    Example 3:
        Input: head = [1], pos = -1 (no cycle)
        Output: false
    
    APPROACH: Floyd's Cycle Detection (Tortoise and Hare)
    
    Use two pointers moving at different speeds. If there's a cycle,
    the fast pointer will eventually meet the slow pointer.
    
    TIME: O(n), SPACE: O(1)
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


# =============================================================================
# PROBLEM 5: LINKED LIST CYCLE II (MEDIUM) - 45 MIN
# =============================================================================

def detect_cycle_start(head):
    """
    PROBLEM: Linked List Cycle II
    
    Given the head of a linked list, return the node where the cycle begins. 
    If there is no cycle, return null.
    
    There is a cycle in a linked list if there is some node in the list that can be 
    reached again by continuously following the next pointer.
    
    Do not modify the linked list.
    
    CONSTRAINTS:
    - The number of the nodes in the list is in the range [0, 10^4]
    - -10^5 <= Node.val <= 10^5
    
    EXAMPLES:
    Example 1:
        Input: head = [3,2,0,-4], pos = 1
        Output: tail connects to node index 1
    
    Example 2:
        Input: head = [1,2], pos = 0
        Output: tail connects to node index 0
    
    Example 3:
        Input: head = [1], pos = -1
        Output: no cycle
    
    APPROACH: Floyd's Algorithm Extended
    
    1. Use fast/slow pointers to detect cycle
    2. If cycle exists, start another pointer from head
    3. Move both at same speed until they meet at cycle start
    
    Mathematical proof: Distance from head to cycle start equals
    distance from meeting point to cycle start
    
    TIME: O(n), SPACE: O(1)
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect if cycle exists
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Phase 2: Find cycle start
    # Start another pointer from head
    start = head
    
    # Move both pointers at same speed
    while start != slow:
        start = start.next
        slow = slow.next
    
    return start


# =============================================================================
# PROBLEM 6: MERGE K SORTED LISTS (HARD) - 60 MIN
# =============================================================================

def merge_k_lists_divide_conquer(lists):
    """
    PROBLEM: Merge k Sorted Lists
    
    You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
    
    Merge all the linked-lists into one sorted linked-list and return it.
    
    CONSTRAINTS:
    - k == lists.length
    - 0 <= k <= 10^4
    - 0 <= lists[i].length <= 500
    - -10^4 <= lists[i][j] <= 10^4
    - lists[i] is sorted in ascending order
    - The sum of lists[i].length will not exceed 10^4
    
    EXAMPLES:
    Example 1:
        Input: lists = [[1,4,5],[1,3,4],[2,6]]
        Output: [1,1,2,3,4,4,5,6]
    
    Example 2:
        Input: lists = []
        Output: []
    
    Example 3:
        Input: lists = [[]]
        Output: []
    
    APPROACH 1: Divide and Conquer
    
    Recursively merge pairs of lists until only one remains
    
    TIME: O(n log k) where n = total nodes, k = number of lists
    SPACE: O(log k) for recursion
    """
    if not lists:
        return None
    
    def merge_two_lists_helper(l1, l2):
        """Helper function to merge two sorted lists"""
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
        """Divide and conquer helper"""
        if start == end:
            return lists[start]
        if start > end:
            return None
        
        mid = (start + end) // 2
        left = merge_helper(lists, start, mid)
        right = merge_helper(lists, mid + 1, end)
        
        return merge_two_lists_helper(left, right)
    
    return merge_helper(lists, 0, len(lists) - 1)


def merge_k_lists_heap(lists):
    """
    APPROACH 2: Min Heap
    
    Use min heap to always get the smallest element among all lists
    
    TIME: O(n log k), SPACE: O(k)
    """
    import heapq
    
    if not lists:
        return None
    
    # Create min heap with first node from each non-empty list
    heap = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        
        # Add current smallest to result
        current.next = node
        current = current.next
        
        # Add next node from same list to heap
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next


# =============================================================================
# PROBLEM 7: REORDER LIST (MEDIUM) - 45 MIN
# =============================================================================

def reorder_list(head):
    """
    PROBLEM: Reorder List
    
    You are given the head of a singly linked-list. The list can be represented as:
    L0 → L1 → … → Ln - 1 → Ln
    
    Reorder the list to be on the following form:
    L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
    
    You may not modify the values in the list's nodes. Only nodes themselves may be changed.
    
    CONSTRAINTS:
    - The number of nodes in the list is in the range [1, 5 * 10^4]
    - 1 <= Node.val <= 1000
    
    EXAMPLES:
    Example 1:
        Input: head = [1,2,3,4]
        Output: [1,4,2,3]
    
    Example 2:
        Input: head = [1,2,3,4,5]
        Output: [1,5,2,4,3]
    
    APPROACH: Find Middle + Reverse + Merge
    
    1. Find middle of list using slow/fast pointers
    2. Reverse second half
    3. Merge two halves alternately
    
    TIME: O(n), SPACE: O(1)
    """
    if not head or not head.next:
        return
    
    # Step 1: Find middle using slow/fast pointers
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Step 2: Reverse second half
    second_half = slow.next
    slow.next = None  # Split the list
    
    # Reverse second half
    prev = None
    current = second_half
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    second_half = prev
    
    # Step 3: Merge two halves alternately
    first_half = head
    while second_half:
        # Save next nodes
        first_next = first_half.next
        second_next = second_half.next
        
        # Connect nodes
        first_half.next = second_half
        second_half.next = first_next
        
        # Move to next pair
        first_half = first_next
        second_half = second_next


# =============================================================================
# PROBLEM 8: ADD TWO NUMBERS (MEDIUM) - 45 MIN
# =============================================================================

def add_two_numbers(l1, l2):
    """
    PROBLEM: Add Two Numbers
    
    You are given two non-empty linked lists representing two non-negative integers. 
    The digits are stored in reverse order, and each of their nodes contains a single digit. 
    Add the two numbers and return the sum as a linked list.
    
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    
    CONSTRAINTS:
    - The number of nodes in each linked list is in the range [1, 100]
    - 0 <= Node.val <= 9
    - It is guaranteed that the list represents a number that does not have leading zeros
    
    EXAMPLES:
    Example 1:
        Input: l1 = [2,4,3], l2 = [5,6,4]
        Output: [7,0,8]
        Explanation: 342 + 465 = 807
    
    Example 2:
        Input: l1 = [0], l2 = [0]
        Output: [0]
    
    Example 3:
        Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
        Output: [8,9,9,9,0,0,0,1]
    
    APPROACH: Digit-by-Digit Addition with Carry
    
    Simulate elementary school addition with carry
    
    TIME: O(max(m, n)), SPACE: O(max(m, n))
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        # Get values (0 if node is None)
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        # Calculate sum and carry
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        # Create new node
        current.next = ListNode(digit)
        current = current.next
        
        # Move to next nodes
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next


# =============================================================================
# PROBLEM 9: COPY LIST WITH RANDOM POINTER (MEDIUM) - 45 MIN
# =============================================================================

def copy_random_list(head):
    """
    PROBLEM: Copy List with Random Pointer
    
    A linked list of length n is given such that each node contains an additional random 
    pointer, which could point to any node in the list, or null.
    
    Construct a deep copy of the list. The deep copy should consist of exactly n brand new 
    nodes, where each new node has its value set to the value of its corresponding original 
    node. Both the next and random pointers of the new nodes should point to new nodes in 
    the copied list.
    
    Return the head of the copied linked list.
    
    CONSTRAINTS:
    - 0 <= n <= 1000
    - -10^4 <= Node.val <= 10^4
    - Node.random is null or is pointing to some node in the linked list
    
    EXAMPLES:
    Example 1:
        Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
        Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
    
    Example 2:
        Input: head = [[1,1],[2,1]]
        Output: [[1,1],[2,1]]
    
    Example 3:
        Input: head = [[3,null],[3,0],[3,null]]
        Output: [[3,null],[3,0],[3,null]]
    
    APPROACH: Hash Map for Node Mapping
    
    Use hash map to maintain mapping between original and copied nodes
    
    TIME: O(n), SPACE: O(n)
    """
    if not head:
        return None
    
    # Hash map to store mapping from original to copied nodes
    node_map = {}
    
    # First pass: Create all nodes
    current = head
    while current:
        node_map[current] = ListNode(current.val)
        current = current.next
    
    # Second pass: Set next and random pointers
    current = head
    while current:
        if current.next:
            node_map[current].next = node_map[current.next]
        if current.random:
            node_map[current].random = node_map[current.random]
        current = current.next
    
    return node_map[head]


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