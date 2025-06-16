"""
=============================================================================
                  WEEK 2 - DAY 11: HEAPS & PRIORITY QUEUES
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. HEAP FUNDAMENTALS
   - Complete binary tree with heap property
   - Max heap: Parent ≥ children, Min heap: Parent ≤ children
   - Array representation: parent at i, children at 2i+1, 2i+2
   - Height: O(log n), operations: O(log n)
   - Not sorted, but partially ordered

2. HEAP OPERATIONS
   - Insert (heapify up): Add at end, bubble up
   - Extract min/max (heapify down): Remove root, bubble down
   - Peek: Access root in O(1)
   - Build heap: O(n) from unsorted array
   - Heap sort: O(n log n) using extract operations

3. PRIORITY QUEUE APPLICATIONS
   - Task scheduling by priority
   - Dijkstra's shortest path algorithm
   - Huffman coding for compression
   - Event simulation systems
   - Top K problems and streaming data

4. PYTHON HEAPQ MODULE
   - Built-in min heap implementation
   - heappush, heappop, heapify functions
   - For max heap: negate values or use wrapper
   - Maintains heap invariant automatically

5. ADVANCED HEAP TECHNIQUES
   - K-way merge using multiple heaps
   - Sliding window maximum/minimum
   - Top K frequent elements
   - Median maintenance with two heaps
   - Heap-based graph algorithms

6. TRANSITION FROM TREES
   - Heaps are specialized binary trees
   - Tree traversal knowledge applies
   - Different ordering from BST
   - Array implementation more efficient than linked nodes

=============================================================================
"""

import heapq
from collections import Counter, defaultdict
from typing import List


# Problem 1: Implement Min Heap - Core heap operations
class MinHeap:
    """
    Min heap implementation with basic operations
    
    Demonstrates fundamental heap mechanics
    """
    
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def has_parent(self, i):
        return self.parent(i) >= 0
    
    def has_left_child(self, i):
        return self.left_child(i) < len(self.heap)
    
    def has_right_child(self, i):
        return self.right_child(i) < len(self.heap)
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def peek(self):
        """Get minimum element without removing"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def insert(self, val):
        """
        Insert element and maintain heap property
        
        Time: O(log n), Space: O(1)
        """
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_min(self):
        """
        Remove and return minimum element
        
        Time: O(log n), Space: O(1)
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()  # Move last to root
        self._heapify_down(0)
        
        return min_val
    
    def _heapify_up(self, index):
        """Bubble up to maintain heap property"""
        while (self.has_parent(index) and 
               self.heap[index] < self.heap[self.parent(index)]):
            self.swap(index, self.parent(index))
            index = self.parent(index)
    
    def _heapify_down(self, index):
        """Bubble down to maintain heap property"""
        while self.has_left_child(index):
            smaller_child_index = self.left_child(index)
            
            if (self.has_right_child(index) and
                self.heap[self.right_child(index)] < self.heap[smaller_child_index]):
                smaller_child_index = self.right_child(index)
            
            if self.heap[index] < self.heap[smaller_child_index]:
                break
            
            self.swap(index, smaller_child_index)
            index = smaller_child_index
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0


# Problem 2: Kth Largest Element - Heap selection problem
def find_kth_largest_sort(nums, k):
    """
    Find kth largest using sorting
    
    Simple but not optimal approach
    
    Time: O(n log n), Space: O(1)
    """
    nums.sort(reverse=True)
    return nums[k-1]


def find_kth_largest_heap(nums, k):
    """
    Find kth largest using min heap
    
    Maintain heap of size k with k largest elements
    
    Time: O(n log k), Space: O(k)
    """
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        
        # Keep only k largest elements
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]  # Root is kth largest


def find_kth_largest_quickselect(nums, k):
    """
    Find kth largest using quickselect
    
    Average O(n), worst case O(n²)
    
    Time: O(n) average, Space: O(1)
    """
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        store_index = left
        for i in range(left, right):
            if nums[i] > pivot_value:  # For kth largest
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        
        # Move pivot to final position
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index
    
    def quickselect(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        # Choose random pivot
        import random
        pivot_index = random.randint(left, right)
        
        pivot_index = partition(left, right, pivot_index)
        
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return quickselect(left, pivot_index - 1, k_smallest)
        else:
            return quickselect(pivot_index + 1, right, k_smallest)
    
    return quickselect(0, len(nums) - 1, k - 1)


# Problem 3: Top K Frequent Elements - Frequency + heap
def top_k_frequent_heap(nums, k):
    """
    Find k most frequent elements using heap
    
    Time: O(n log k), Space: O(n)
    """
    # Count frequencies
    count = Counter(nums)
    
    # Use min heap to keep k most frequent
    heap = []
    
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]


def top_k_frequent_bucket_sort(nums, k):
    """
    Find k most frequent using bucket sort
    
    Time: O(n), Space: O(n)
    """
    count = Counter(nums)
    
    # Bucket sort by frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    # Iterate from highest frequency to lowest
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result


# Problem 4: Merge k Sorted Lists - Multi-way merge
class ListNode:
    """Simple linked list node for demonstration"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __lt__(self, other):
        return self.val < other.val


def merge_k_sorted_lists(lists):
    """
    Merge k sorted linked lists using min heap
    
    Time: O(n log k), Space: O(k)
    where n = total number of nodes, k = number of lists
    """
    heap = []
    
    # Add first node from each list
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        
        current.next = node
        current = current.next
        
        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next


def merge_k_sorted_arrays(arrays):
    """
    Merge k sorted arrays using min heap
    
    More practical version for interview
    
    Time: O(n log k), Space: O(k)
    """
    heap = []
    
    # Add first element from each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))  # (value, array_idx, element_idx)
    
    result = []
    
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same array
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))
    
    return result


# Problem 5: Find Median from Data Stream - Two heaps technique
class MedianFinder:
    """
    Find median in streaming data using two heaps
    
    Max heap for smaller half, min heap for larger half
    """
    
    def __init__(self):
        self.small = []  # Max heap (negate values)
        self.large = []  # Min heap
    
    def add_num(self, num):
        """
        Add number to data structure
        
        Time: O(log n), Space: O(1)
        """
        # Add to appropriate heap
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)  # Negate for max heap
        else:
            heapq.heappush(self.large, num)
        
        # Balance heaps (difference ≤ 1)
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        elif len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def find_median(self):
        """
        Find median of all numbers
        
        Time: O(1), Space: O(1)
        """
        if len(self.small) > len(self.large):
            return -self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        else:
            return (-self.small[0] + self.large[0]) / 2.0


# Problem 6: Sliding Window Maximum - Monotonic deque + heap comparison
def sliding_window_maximum_heap(nums, k):
    """
    Find maximum in each sliding window using heap
    
    Time: O(n log k), Space: O(k)
    """
    result = []
    heap = []
    
    for i, num in enumerate(nums):
        # Add current element
        heapq.heappush(heap, (-num, i))  # Max heap with index
        
        # Remove elements outside window
        while heap and heap[0][1] <= i - k:
            heapq.heappop(heap)
        
        # Add maximum to result
        if i >= k - 1:
            result.append(-heap[0][0])
    
    return result


from collections import deque

def sliding_window_maximum_deque(nums, k):
    """
    Find maximum using monotonic deque (optimal)
    
    Time: O(n), Space: O(k)
    """
    dq = deque()  # Store indices
    result = []
    
    for i, num in enumerate(nums):
        # Remove elements outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements (maintain decreasing order)
        while dq and nums[dq[-1]] <= num:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum to result
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


# ADVANCED PROBLEMS FOR EXTRA PRACTICE

def kth_smallest_in_sorted_matrix(matrix, k):
    """
    Find kth smallest element in sorted matrix using heap
    
    Time: O(k log min(k, n)), Space: O(min(k, n))
    """
    n = len(matrix)
    heap = []
    
    # Add first element from each row
    for i in range(min(k, n)):
        heapq.heappush(heap, (matrix[i][0], i, 0))
    
    for _ in range(k):
        val, row, col = heapq.heappop(heap)
        
        # Add next element from same row
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    
    return val


def reorganize_string(s):
    """
    Reorganize string so no two adjacent characters are same
    
    Use max heap to always place most frequent character
    
    Time: O(n log k), Space: O(k) where k = unique characters
    """
    count = Counter(s)
    
    # Max heap of (frequency, character)
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)
    
    result = []
    prev_freq, prev_char = 0, ''
    
    while heap:
        freq, char = heapq.heappop(heap)
        
        result.append(char)
        
        # Add back previous character if it still has frequency
        if prev_freq < 0:
            heapq.heappush(heap, (prev_freq, prev_char))
        
        # Update previous
        prev_freq, prev_char = freq + 1, char
    
    result_str = ''.join(result)
    return result_str if len(result_str) == len(s) else ""


class MaxStack:
    """
    Stack that supports push, pop, top, and popMax operations
    
    Use two heaps to track maximum elements
    """
    
    def __init__(self):
        self.stack = []
        self.heap = []  # Max heap
        self.removed = set()  # Track removed elements
        self.counter = 0  # Unique identifier
    
    def push(self, x):
        """Push element onto stack"""
        self.counter += 1
        self.stack.append((x, self.counter))
        heapq.heappush(self.heap, (-x, -self.counter))
    
    def pop(self):
        """Remove and return top element"""
        while self.stack and self.stack[-1][1] in self.removed:
            self.stack.pop()
        
        if not self.stack:
            return None
        
        val, counter = self.stack.pop()
        self.removed.add(counter)
        return val
    
    def top(self):
        """Return top element without removing"""
        while self.stack and self.stack[-1][1] in self.removed:
            self.stack.pop()
        
        return self.stack[-1][0] if self.stack else None
    
    def pop_max(self):
        """Remove and return maximum element"""
        while self.heap and -self.heap[0][1] in self.removed:
            heapq.heappop(self.heap)
        
        if not self.heap:
            return None
        
        val, counter = heapq.heappop(self.heap)
        self.removed.add(-counter)
        return -val


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all heap problems with comprehensive test cases
    """
    print("=== TESTING DAY 11 PROBLEMS ===\n")
    
    # Test Min Heap
    print("1. Min Heap Implementation:")
    min_heap = MinHeap()
    
    elements = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    for elem in elements:
        min_heap.insert(elem)
    
    print(f"   Inserted: {elements}")
    
    extracted = []
    while not min_heap.is_empty():
        extracted.append(min_heap.extract_min())
    
    print(f"   Extracted in order: {extracted}")
    print(f"   ✓ Sorted: {extracted == sorted(elements)}")
    print()
    
    # Test Kth Largest Element
    print("2. Kth Largest Element:")
    test_array = [3, 2, 1, 5, 6, 4]
    k = 2
    
    result_heap = find_kth_largest_heap(test_array.copy(), k)
    result_quickselect = find_kth_largest_quickselect(test_array.copy(), k)
    
    print(f"   Array: {test_array}, k={k}")
    print(f"   Kth largest (heap): {result_heap} (expected: 5)")
    print(f"   Kth largest (quickselect): {result_quickselect} (expected: 5)")
    print()
    
    # Test Top K Frequent
    print("3. Top K Frequent Elements:")
    freq_array = [1, 1, 1, 2, 2, 3]
    k_freq = 2
    
    top_k_heap = top_k_frequent_heap(freq_array, k_freq)
    top_k_bucket = top_k_frequent_bucket_sort(freq_array, k_freq)
    
    print(f"   Array: {freq_array}, k={k_freq}")
    print(f"   Top k frequent (heap): {sorted(top_k_heap)}")
    print(f"   Top k frequent (bucket): {sorted(top_k_bucket)}")
    print(f"   Expected: [1, 2]")
    print()
    
    # Test Merge K Sorted Arrays
    print("4. Merge K Sorted Arrays:")
    sorted_arrays = [
        [1, 4, 5],
        [1, 3, 4],
        [2, 6]
    ]
    
    merged = merge_k_sorted_arrays(sorted_arrays)
    print(f"   Input arrays: {sorted_arrays}")
    print(f"   Merged result: {merged}")
    print(f"   Expected: [1, 1, 2, 3, 4, 4, 5, 6]")
    print()
    
    # Test Median Finder
    print("5. Find Median from Data Stream:")
    median_finder = MedianFinder()
    
    stream = [1, 2, 3, 4, 5]
    medians = []
    
    for num in stream:
        median_finder.add_num(num)
        medians.append(median_finder.find_median())
    
    print(f"   Stream: {stream}")
    print(f"   Medians: {medians}")
    print(f"   Expected: [1, 1.5, 2, 2.5, 3]")
    print()
    
    # Test Sliding Window Maximum
    print("6. Sliding Window Maximum:")
    window_array = [1, 3, -1, -3, 5, 3, 6, 7]
    window_k = 3
    
    max_heap = sliding_window_maximum_heap(window_array, window_k)
    max_deque = sliding_window_maximum_deque(window_array, window_k)
    
    print(f"   Array: {window_array}, k={window_k}")
    print(f"   Window maxima (heap): {max_heap}")
    print(f"   Window maxima (deque): {max_deque}")
    print(f"   Expected: [3, 3, 5, 5, 6, 7]")


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_heap_properties():
    """
    Visual demonstration of heap properties and operations
    """
    print("\n=== HEAP PROPERTIES DEMONSTRATION ===")
    
    print("Min Heap Structure (array representation):")
    print("         1")
    print("       /   \\")
    print("      3     2")
    print("     / \\   / \\")
    print("    7   4 5   6")
    print("   / \\")
    print("  9   8")
    
    heap_array = [1, 3, 2, 7, 4, 5, 6, 9, 8]
    print(f"\nArray: {heap_array}")
    
    print("\nIndex relationships:")
    for i in range(len(heap_array)):
        parent = (i - 1) // 2 if i > 0 else None
        left = 2 * i + 1 if 2 * i + 1 < len(heap_array) else None
        right = 2 * i + 2 if 2 * i + 2 < len(heap_array) else None
        
        print(f"  Index {i} (val={heap_array[i]}): parent={parent}, left={left}, right={right}")
    
    print("\nHeap Property Verification:")
    is_valid = True
    for i in range(len(heap_array)):
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < len(heap_array) and heap_array[i] > heap_array[left]:
            print(f"  ✗ Violation: parent {heap_array[i]} > left child {heap_array[left]}")
            is_valid = False
        
        if right < len(heap_array) and heap_array[i] > heap_array[right]:
            print(f"  ✗ Violation: parent {heap_array[i]} > right child {heap_array[right]}")
            is_valid = False
    
    if is_valid:
        print("  ✓ Valid min heap!")


def heap_vs_other_structures():
    """
    Compare heap with other data structures
    """
    print("\n=== HEAP VS OTHER STRUCTURES ===")
    
    operations = [
        ("Insert", "O(log n)", "O(log n)", "O(1)", "O(n)"),
        ("Delete Min/Max", "O(log n)", "O(log n)", "O(n)", "O(1)"),
        ("Find Min/Max", "O(1)", "O(log n)", "O(n)", "O(1)"),
        ("Search", "O(n)", "O(log n)", "O(n)", "O(n)"),
        ("Build from array", "O(n)", "O(n log n)", "O(n)", "O(n)")
    ]
    
    print("Operation        | Heap     | BST      | Array    | Sorted Array")
    print("-----------------|----------|----------|----------|-------------")
    for op, heap, bst, array, sorted_arr in operations:
        print(f"{op:16} | {heap:8} | {bst:8} | {array:8} | {sorted_arr}")
    
    print("\nWhen to use each:")
    print("  Heap: Priority queue, top K problems, streaming median")
    print("  BST: Range queries, ordered traversal, dynamic sorted data")
    print("  Array: Random access, cache locality, simple problems")
    print("  Sorted Array: Static data, binary search, space efficiency")


def heap_problem_patterns():
    """
    Common patterns in heap-based problems
    """
    print("\n=== HEAP PROBLEM PATTERNS ===")
    
    patterns = {
        "Top K Problems": [
            "Kth largest/smallest element",
            "Top K frequent elements",
            "K closest points to origin",
            "K pairs with smallest sums"
        ],
        "Two Heaps Technique": [
            "Find median in data stream",
            "Sliding window median",
            "Split array into two equal sum parts"
        ],
        "Multi-way Merge": [
            "Merge K sorted lists/arrays",
            "Smallest range covering K lists",
            "Kth smallest in sorted matrix"
        ],
        "Scheduling/Priority": [
            "Task scheduler",
            "Meeting rooms",
            "CPU scheduling",
            "Event processing"
        ],
        "Sliding Window": [
            "Sliding window maximum",
            "Maximum in all subarrays",
            "Constrained subset sum"
        ]
    }
    
    for pattern, examples in patterns.items():
        print(f"\n{pattern}:")
        for example in examples:
            print(f"  • {example}")
    
    print("\nKey Insights:")
    print("  • Use min heap for Kth largest problems")
    print("  • Use max heap for Kth smallest problems")
    print("  • Two heaps can maintain running median")
    print("  • Heap + hash map for complex priority management")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Demonstrate heap properties
    demonstrate_heap_properties()
    
    # Compare with other structures
    heap_vs_other_structures()
    
    # Show problem patterns
    heap_problem_patterns()
    
    print("\n" + "="*70)
    print("DAY 11 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. Heaps provide O(log n) insert/delete with O(1) peek")
    print("2. Two heaps technique powerful for median/partition problems")
    print("3. Multi-way merge pattern uses heap for efficiency")
    print("4. Top K problems often solved optimally with heaps")
    print("5. Python heapq provides min heap; negate for max heap")
    print("6. Consider heap vs sorting trade-offs for K-selection")
    print("7. Streaming data problems often benefit from heap solutions")
    print("\nTransition: Day 11→12 - Comprehensive review")
    print("- Integration of all Week 2 concepts")
    print("- Complex problems combining multiple techniques")
    print("- Performance comparison and optimization strategies")
    print("\nNext: Day 12 - Week 2 Review & Assessment") 