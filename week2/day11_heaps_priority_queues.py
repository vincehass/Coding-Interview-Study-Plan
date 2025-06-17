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


# =============================================================================
# PROBLEM 1: IMPLEMENT MIN HEAP (MEDIUM) - 45 MIN
# =============================================================================

class MinHeap:
    """
    PROBLEM: Implement Min Heap
    
    Design a min heap data structure that supports the following operations:
    - insert(val): Insert a value into the heap
    - extract_min(): Remove and return the minimum element
    - peek(): Return the minimum element without removing it
    - size(): Return the number of elements in the heap
    - is_empty(): Check if the heap is empty
    
    CONSTRAINTS:
    - 1 <= val <= 10^9
    - At most 10^4 calls will be made to each function
    - extract_min() and peek() will not be called on empty heap
    
    EXAMPLES:
    Example 1:
        Operations: ["insert", "insert", "peek", "extract_min", "size"]
        Values: [3, 1, null, null, null]
        Output: [null, null, 1, 1, 1]
        Explanation:
        - insert(3): heap = [3]
        - insert(1): heap = [1, 3] 
        - peek(): return 1
        - extract_min(): return 1, heap = [3]
        - size(): return 1
        
    Example 2:
        Operations: ["insert", "insert", "insert", "extract_min", "extract_min"]
        Values: [5, 2, 8, null, null]
        Output: [null, null, null, 2, 5]
    
    APPROACH: Array-Based Binary Heap
    
    Use a complete binary tree represented as an array:
    - Parent of node at index i is at (i-1)//2
    - Left child of node at index i is at 2*i+1
    - Right child of node at index i is at 2*i+2
    
    Maintain heap property: parent <= children
    - Insert: Add at end, bubble up (heapify up)
    - Extract: Replace root with last element, bubble down (heapify down)
    
    TIME: O(log n) for insert/extract, O(1) for peek, SPACE: O(n) for storage
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
        
        TIME: O(log n), SPACE: O(1)
        """
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_min(self):
        """
        Remove and return minimum element
        
        TIME: O(log n), SPACE: O(1)
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


# =============================================================================
# PROBLEM 2: KTH LARGEST ELEMENT IN AN ARRAY (MEDIUM) - 45 MIN
# =============================================================================

def find_kth_largest_sort(nums, k):
    """
    PROBLEM: Kth Largest Element in an Array
    
    Given an integer array nums and an integer k, return the kth largest element in the array.
    Note that it is the kth largest element in the sorted order, not the kth distinct element.
    Can you solve it without sorting?
    
    CONSTRAINTS:
    - 1 <= k <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    
    EXAMPLES:
    Example 1:
        Input: nums = [3,2,1,5,6,4], k = 2
        Output: 5
        Explanation: When sorted in descending order: [6,5,4,3,2,1], the 2nd largest is 5.
        
    Example 2:
        Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
        Output: 4
        Explanation: When sorted in descending order: [6,5,5,4,3,3,2,2,1], the 4th largest is 4.
        
    Example 3:
        Input: nums = [1], k = 1
        Output: 1
    
    APPROACH 1: Sorting
    
    Simple approach: sort array in descending order and return kth element.
    
    TIME: O(n log n), SPACE: O(1)
    """
    nums.sort(reverse=True)
    return nums[k-1]


def find_kth_largest_heap(nums, k):
    """
    APPROACH 2: Min Heap of Size K
    
    Maintain a min heap containing the k largest elements seen so far.
    The root of this heap will be the kth largest element.
    
    Why min heap? We want to quickly remove the smallest among the k largest elements
    when we encounter a larger element.
    
    TIME: O(n log k), SPACE: O(k)
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
    APPROACH 3: Quickselect Algorithm
    
    Based on quicksort's partitioning. Instead of sorting entire array,
    we only recurse on the side that contains the kth largest element.
    
    Average case: O(n), Worst case: O(n²)
    
    TIME: O(n) average, SPACE: O(1)
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
        
        # Move pivot to final place
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


# =============================================================================
# PROBLEM 3: TOP K FREQUENT ELEMENTS (MEDIUM) - 45 MIN
# =============================================================================

def top_k_frequent_heap(nums, k):
    """
    PROBLEM: Top K Frequent Elements
    
    Given an integer array nums and an integer k, return the k most frequent elements.
    You may return the answer in any order.
    
    CONSTRAINTS:
    - 1 <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    - k is in the range [1, the number of unique elements in the array]
    - It's guaranteed that the answer is unique
    
    EXAMPLES:
    Example 1:
        Input: nums = [1,1,1,2,2,3], k = 2
        Output: [1,2]
        Explanation: 1 appears 3 times, 2 appears 2 times, 3 appears 1 time.
        The 2 most frequent elements are 1 and 2.
        
    Example 2:
        Input: nums = [1], k = 1
        Output: [1]
        
    Example 3:
        Input: nums = [4,1,-1,2,-1,2,3], k = 2
        Output: [-1,2]
    
    APPROACH 1: Min Heap
    
    1. Count frequency of each element
    2. Use min heap to maintain top k frequent elements
    3. Heap size never exceeds k, so space efficient
    
    TIME: O(n log k), SPACE: O(n + k)
    """
    # Count frequencies
    count = Counter(nums)
    
    # Use min heap to keep top k elements
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]


def top_k_frequent_bucket_sort(nums, k):
    """
    APPROACH 2: Bucket Sort
    
    Since frequency is bounded by array length, we can use bucket sort.
    Create buckets for each possible frequency (0 to n).
    
    TIME: O(n), SPACE: O(n)
    """
    count = Counter(nums)
    
    # Create buckets for each possible frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    # Place elements in buckets by frequency
    for num, freq in count.items():
        buckets[freq].append(num)
    
    # Collect top k elements from highest frequency buckets
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result


# =============================================================================
# PROBLEM 4: MERGE K SORTED LISTS (HARD) - 60 MIN
# =============================================================================

class ListNode:
    """Definition for singly-linked list"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __lt__(self, other):
        return self.val < other.val

def merge_k_sorted_lists(lists):
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
        Explanation: The linked-lists are:
        [
          1->4->5,
          1->3->4,
          2->6
        ]
        merging them into one sorted list: 1->1->2->3->4->4->5->6
        
    Example 2:
        Input: lists = []
        Output: []
        
    Example 3:
        Input: lists = [[]]
        Output: []
    
    APPROACH: Min Heap
    
    Use a min heap to always get the smallest current element among all lists.
    
    1. Add the first node of each non-empty list to the heap
    2. Extract minimum node, add it to result
    3. If the extracted node has a next node, add it to heap
    4. Repeat until heap is empty
    
    TIME: O(n log k) where n is total nodes and k is number of lists
    SPACE: O(k) for the heap
    """
    if not lists:
        return None
    
    # Initialize heap with first node of each list
    heap = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        # Add next node from same list if exists
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next


def merge_k_sorted_arrays(arrays):
    """
    PROBLEM VARIATION: Merge K Sorted Arrays
    
    Similar to merge k sorted lists but with arrays.
    
    TIME: O(n log k), SPACE: O(k)
    """
    if not arrays:
        return []
    
    # Heap stores (value, array_index, element_index)
    heap = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))
    
    result = []
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same array if exists
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))
    
    return result


# =============================================================================
# PROBLEM 5: FIND MEDIAN FROM DATA STREAM (HARD) - 60 MIN
# =============================================================================

class MedianFinder:
    """
    PROBLEM: Find Median from Data Stream
    
    The median is the middle value in an ordered integer list. If the size of the list is even,
    there is no middle value, and the median is the mean of the two middle values.
    
    Implement the MedianFinder class:
    - MedianFinder() initializes the MedianFinder object
    - void addNum(int num) adds the integer num from the data stream to the data structure
    - double findMedian() returns the median of all elements so far
    
    CONSTRAINTS:
    - -10^5 <= num <= 10^5
    - There will be at least one element in the data structure before calling findMedian
    - At most 5 * 10^4 calls will be made to addNum and findMedian
    
    EXAMPLES:
    Example 1:
        Input: ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
               [[], [1], [2], [], [3], []]
        Output: [null, null, null, 1.5, null, 2.0]
        Explanation:
        MedianFinder medianFinder = new MedianFinder();
        medianFinder.addNum(1);    // arr = [1]
        medianFinder.addNum(2);    // arr = [1, 2]
        medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
        medianFinder.addNum(3);    // arr = [1, 2, 3]
        medianFinder.findMedian(); // return 2.0
    
    APPROACH: Two Heaps
    
    Use two heaps to maintain the median:
    - Max heap (left_half): stores smaller half of numbers
    - Min heap (right_half): stores larger half of numbers
    
    Invariants:
    1. max_heap size >= min_heap size (at most 1 larger)
    2. All elements in max_heap <= all elements in min_heap
    3. Median is either max_heap.top() or average of both tops
    
    TIME: O(log n) for addNum, O(1) for findMedian, SPACE: O(n)
    """
    
    def __init__(self):
        # Max heap for smaller half (negate values for max heap simulation)
        self.left_half = []  # max heap
        # Min heap for larger half
        self.right_half = []  # min heap
    
    def add_num(self, num):
        """
        Add number while maintaining heap invariants
        
        TIME: O(log n), SPACE: O(1)
        """
        # Always add to left_half first
        heapq.heappush(self.left_half, -num)
        
        # Move largest from left_half to right_half
        if self.left_half and self.right_half and (-self.left_half[0] > self.right_half[0]):
            val = -heapq.heappop(self.left_half)
            heapq.heappush(self.right_half, val)
        
        # Balance heap sizes
        if len(self.left_half) > len(self.right_half) + 1:
            val = -heapq.heappop(self.left_half)
            heapq.heappush(self.right_half, val)
        elif len(self.right_half) > len(self.left_half):
            val = heapq.heappop(self.right_half)
            heapq.heappush(self.left_half, -val)
    
    def find_median(self):
        """
        Get median of all numbers added so far
        
        TIME: O(1), SPACE: O(1)
        """
        if len(self.left_half) > len(self.right_half):
            return -self.left_half[0]
        else:
            return (-self.left_half[0] + self.right_half[0]) / 2.0


# =============================================================================
# PROBLEM 6: SLIDING WINDOW MAXIMUM (HARD) - 60 MIN
# =============================================================================

def sliding_window_maximum_heap(nums, k):
    """
    PROBLEM: Sliding Window Maximum
    
    You are given an array of integers nums, there is a sliding window of size k which is moving
    from the very left of the array to the very right. You can only see the k numbers in the window.
    Each time the sliding window moves right by one position.
    
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
    
    APPROACH 1: Max Heap with Lazy Removal
    
    Use a max heap to track elements in current window.
    Since we can't efficiently remove arbitrary elements from heap,
    we use lazy removal - keep outdated elements and ignore them when they appear at top.
    
    TIME: O(n log k), SPACE: O(k)
    """
    if not nums:
        return []
    
    heap = []
    result = []
    
    for i in range(len(nums)):
        # Add current element to heap
        heapq.heappush(heap, (-nums[i], i))
        
        # Remove elements outside current window
        while heap and heap[0][1] <= i - k:
            heapq.heappop(heap)
        
        # Add maximum to result if window is full
        if i >= k - 1:
            result.append(-heap[0][0])
    
    return result


def sliding_window_maximum_deque(nums, k):
    """
    APPROACH 2: Monotonic Deque
    
    Use a deque to maintain indices of elements in decreasing order of their values.
    The front of deque always contains index of maximum element in current window.
    
    TIME: O(n), SPACE: O(k)
    """
    from collections import deque
    
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices of smaller elements (they can't be maximum)
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum to result if window is full
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


# =============================================================================
# PROBLEM 7: KTHSMALLEST ELEMENT IN A SORTED MATRIX (MEDIUM) - 45 MIN
# =============================================================================

def kth_smallest_in_sorted_matrix(matrix, k):
    """
    PROBLEM: Kth Smallest Element in a Sorted Matrix
    
    Given an n x n matrix where each of the rows and columns is sorted in ascending order,
    return the kth smallest element in the matrix.
    
    Note that it is the kth smallest element in the sorted order, not the kth distinct element.
    
    CONSTRAINTS:
    - n == matrix.length == matrix[i].length
    - 1 <= n <= 300
    - -10^9 <= matrix[i][j] <= 10^9
    - All the rows and columns of matrix are guaranteed to be sorted in non-decreasing order
    - 1 <= k <= n^2
    
    EXAMPLES:
    Example 1:
        Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
        Output: 13
        Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13
        
    Example 2:
        Input: matrix = [[-5]], k = 1
        Output: -5
    
    APPROACH: Min Heap
    
    Use min heap to efficiently find kth smallest element.
    Start with first element of each row, then expand.
    
    TIME: O(k log n), SPACE: O(n)
    """
    n = len(matrix)
    heap = []
    
    # Add first element of each row
    for i in range(n):
        heapq.heappush(heap, (matrix[i][0], i, 0))
    
    # Extract k-1 elements
    for _ in range(k - 1):
        val, row, col = heapq.heappop(heap)
        
        # Add next element from same row if exists
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    
    return heap[0][0]


# =============================================================================
# PROBLEM 8: REORGANIZE STRING (MEDIUM) - 45 MIN
# =============================================================================

def reorganize_string(s):
    """
    PROBLEM: Reorganize String
    
    Given a string s, rearrange the characters of s so that any two adjacent characters are not the same.
    Return any possible rearrangement of s or return "" if not possible.
    
    CONSTRAINTS:
    - 1 <= s.length <= 500
    - s consists of lowercase English letters
    
    EXAMPLES:
    Example 1:
        Input: s = "aab"
        Output: "aba"
        
    Example 2:
        Input: s = "aaab"
        Output: ""
        Explanation: It's impossible to rearrange so no two adjacent characters are the same.
        
    Example 3:
        Input: s = "aabcc"
        Output: "abcac" (or other valid arrangements)
    
    APPROACH: Max Heap with Greedy Strategy
    
    1. Count frequency of each character
    2. Use max heap to always pick the most frequent available character
    3. Alternate between most frequent and second most frequent characters
    4. If at any point we can't find a valid next character, return ""
    
    TIME: O(n log k) where k is number of unique characters, SPACE: O(k)
    """
    # Count character frequencies
    count = Counter(s)
    
    # Check if reorganization is possible
    # Most frequent character can appear at most (n+1)//2 times
    if max(count.values()) > (len(s) + 1) // 2:
        return ""
    
    # Max heap of (-frequency, character)
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)
    
    result = []
    prev_freq, prev_char = 0, ''
    
    while heap:
        # Get most frequent character
        freq, char = heapq.heappop(heap)
        result.append(char)
        
        # Put back previous character if it still has remaining count
        if prev_freq < 0:
            heapq.heappush(heap, (prev_freq, prev_char))
        
        # Update previous character info
        prev_freq, prev_char = freq + 1, char  # freq is negative
    
    return ''.join(result) if len(result) == len(s) else ""


# =============================================================================
# PROBLEM 9: MAX STACK (HARD) - 60 MIN
# =============================================================================

class MaxStack:
    """
    PROBLEM: Max Stack
    
    Design a max stack data structure that supports the stack operations and supports finding the stack's maximum element.
    
    Implement the MaxStack class:
    - MaxStack() Initializes the stack object
    - void push(int x) Pushes element x onto the stack
    - int pop() Removes the element on top of the stack and returns it
    - int top() Gets the element on the top of the stack without removing it
    - int peekMax() Retrieves the maximum element in the stack without removing it
    - int popMax() Retrieves the maximum element in the stack and removes it
    
    CONSTRAINTS:
    - -10^7 <= x <= 10^7
    - At most 10^5 calls will be made to push, pop, top, peekMax, and popMax
    - There will be at least one element in the stack when pop, top, peekMax, or popMax is called
    
    EXAMPLES:
    Example 1:
        Input: ["MaxStack", "push", "push", "push", "top", "popMax", "top", "peekMax", "pop", "top"]
               [[], [5], [1], [5], [], [], [], [], [], []]
        Output: [null, null, null, null, 5, 5, 1, 5, 1, 5]
        Explanation:
        MaxStack stk = new MaxStack();
        stk.push(5);   // [5] the top of the stack and the maximum number is 5.
        stk.push(1);   // [5, 1] the top of the stack is 1, but the maximum is 5.
        stk.push(5);   // [5, 1, 5] the top of the stack is 5, which is also the maximum, because 5 == 5.
        stk.top();     // return 5, [5, 1, 5] the stack did not change.
        stk.popMax();  // return 5, [5, 1] the stack is changed now, and the top is different from the max.
        stk.top();     // return 1, [5, 1] the stack did not change.
        stk.peekMax(); // return 5, [5, 1] the stack did not change.
        stk.pop();     // return 1, [5] the top of the stack and the max element is now 5.
        stk.top();     // return 5, [5] the stack did not change.
    
    APPROACH: Two Stacks with Heap
    
    Use a stack for normal operations and a max heap to track maximum elements.
    For popMax(), use a temporary stack to restore order after removing max element.
    
    TIME: O(1) for push/pop/top/peekMax, O(log n) for popMax, SPACE: O(n)
    """
    
    def __init__(self):
        self.stack = []
        self.max_heap = []
        self.id_counter = 0
    
    def push(self, x):
        """Push element onto stack"""
        self.id_counter += 1
        self.stack.append((x, self.id_counter))
        heapq.heappush(self.max_heap, (-x, -self.id_counter))
    
    def pop(self):
        """Remove and return top element"""
        val, _ = self.stack.pop()
        return val
    
    def top(self):
        """Get top element without removing"""
        return self.stack[-1][0]
    
    def peek_max(self):
        """Get maximum element without removing"""
        # Clean up heap of removed elements
        while self.max_heap:
            max_val, max_id = self.max_heap[0]
            max_val, max_id = -max_val, -max_id
            
            # Check if this element is still in stack
            if self.stack and self.stack[-1] == (max_val, max_id):
                return max_val
            
            # Check if element exists anywhere in stack
            found = False
            for val, id_num in self.stack:
                if val == max_val and id_num == max_id:
                    found = True
                    break
            
            if found:
                return max_val
            else:
                heapq.heappop(self.max_heap)
        
        return self.stack[-1][0] if self.stack else None
    
    def pop_max(self):
        """Remove and return maximum element"""
        # Find maximum value
        max_val = self.peek_max()
        
        # Use temporary stack to remove max element
        temp_stack = []
        
        # Pop elements until we find the maximum (rightmost occurrence)
        while self.stack:
            val, id_num = self.stack.pop()
            if val == max_val:
                # Found the max element, restore other elements
                while temp_stack:
                    self.stack.append(temp_stack.pop())
                return max_val
            temp_stack.append((val, id_num))
        
        # This shouldn't happen if peek_max() works correctly
        return None


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