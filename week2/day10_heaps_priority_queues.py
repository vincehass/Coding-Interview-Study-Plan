# =============================================================================
# PROBLEM 1: KTH LARGEST ELEMENT IN AN ARRAY (MEDIUM) - 45 MIN
# =============================================================================

def find_kth_largest_heap(nums, k):
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
    
    Example 2:
        Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
        Output: 4
    
    APPROACH 1: Min Heap of Size K
    
    Maintain a min heap of size k containing the k largest elements
    
    TIME: O(n log k), SPACE: O(k)
    """
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]


def find_kth_largest_quickselect(nums, k):
    """
    APPROACH 2: Quickselect Algorithm
    
    Optimized approach using partition from quicksort
    
    TIME: O(n) average, O(n²) worst case, SPACE: O(1)
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
# PROBLEM 2: TOP K FREQUENT ELEMENTS (MEDIUM) - 45 MIN
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
    
    Example 2:
        Input: nums = [1], k = 1
        Output: [1]
    
    APPROACH 1: Min Heap
    
    Use min heap to maintain top k frequent elements
    
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
    
    Since frequency is bounded by array length, use bucket sort
    
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
# PROBLEM 3: MERGE K SORTED LISTS (HARD) - 60 MIN
# =============================================================================

def merge_k_lists_heap(lists):
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
        merging them into one sorted list:
        1->1->2->3->4->4->5->6
    
    Example 2:
        Input: lists = []
        Output: []
    
    Example 3:
        Input: lists = [[]]
        Output: []
    
    APPROACH: Min Heap
    
    Use min heap to always get the smallest element among all lists
    
    TIME: O(n log k), SPACE: O(k)
    """
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
# PROBLEM 4: FIND MEDIAN FROM DATA STREAM (HARD) - 60 MIN
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
    
    Use max heap for smaller half, min heap for larger half
    
    TIME: O(log n) for addNum, O(1) for findMedian, SPACE: O(n)
    """
    
    def __init__(self):
        """Initialize data structure"""
        self.small = []  # Max heap for smaller half (negate values)
        self.large = []  # Min heap for larger half
    
    def addNum(self, num):
        """Add number to data structure"""
        # Add to appropriate heap
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)  # Negate for max heap
        else:
            heapq.heappush(self.large, num)
        
        # Balance heaps
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        elif len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def findMedian(self):
        """Return median of all elements"""
        if len(self.small) > len(self.large):
            return -self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        else:
            return (-self.small[0] + self.large[0]) / 2.0


# =============================================================================
# PROBLEM 5: SLIDING WINDOW MAXIMUM (HARD) - 60 MIN
# =============================================================================

def max_sliding_window_heap(nums, k):
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
    
    APPROACH 1: Max Heap with Lazy Deletion
    
    Use max heap but handle stale elements lazily
    
    TIME: O(n log k), SPACE: O(k)
    """
    if not nums or k == 0:
        return []
    
    result = []
    heap = []  # (negative_value, index) for max heap
    
    for i in range(len(nums)):
        # Add current element
        heapq.heappush(heap, (-nums[i], i))
        
        # Remove elements outside window
        while heap and heap[0][1] <= i - k:
            heapq.heappop(heap)
        
        # Add maximum to result if window is full
        if i >= k - 1:
            result.append(-heap[0][0])
    
    return result


def max_sliding_window_deque(nums, k):
    """
    APPROACH 2: Monotonic Deque (Optimal)
    
    Use deque to maintain indices in decreasing order of their values
    
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
# PROBLEM 6: TASK SCHEDULER (MEDIUM) - 45 MIN
# =============================================================================

def least_interval(tasks, n):
    """
    PROBLEM: Task Scheduler
    
    Given a characters array tasks, representing the tasks a CPU needs to do, where each letter 
    represents a different task. Tasks could be done in any order. Each task is done in one unit of time. 
    For each unit of time, the CPU could complete either one task or just be idle.
    
    However, there is a non-negative integer n that represents the cooldown period between two same tasks 
    (the same letter in the array), that is that there must be at least n units of time between any two same tasks.
    
    Return the least number of units of time that the CPU will take to finish all the given tasks.
    
    CONSTRAINTS:
    - 1 <= task.length <= 10^4
    - tasks[i] is upper-case English letter
    - The integer n is in the range [0, 100]
    
    EXAMPLES:
    Example 1:
        Input: tasks = ["A","A","A","B","B","B"], n = 2
        Output: 8
        Explanation: A -> B -> idle -> A -> B -> idle -> A -> B
        There is at least 2 units of time between any two same tasks.
    
    Example 2:
        Input: tasks = ["A","A","A","B","B","B"], n = 0
        Output: 6
        Explanation: On this case any permutation of size 6 would work since n = 0.
        ["A","A","A","B","B","B"]
        ["A","B","A","B","A","B"]
        ["B","B","B","A","A","A"]
        ...
        And so on.
    
    Example 3:
        Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
        Output: 16
        Explanation: One possible solution is
        A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A
    
    APPROACH: Greedy with Max Heap
    
    Always schedule the most frequent task that's available
    
    TIME: O(n), SPACE: O(1) - at most 26 different tasks
    """
    # Count task frequencies
    task_counts = Counter(tasks)
    
    # Use max heap for frequencies
    heap = [-count for count in task_counts.values()]
    heapq.heapify(heap)
    
    time = 0
    
    while heap:
        cycle = []
        
        # Try to schedule n+1 tasks (including cooldown)
        for _ in range(n + 1):
            if heap:
                cycle.append(-heapq.heappop(heap))
        
        # Put back tasks that still have remaining count
        for count in cycle:
            if count > 1:
                heapq.heappush(heap, -(count - 1))
        
        # Add time for this cycle
        time += n + 1 if heap else len(cycle)
    
    return time


# =============================================================================
# PROBLEM 7: UGLY NUMBER II (MEDIUM) - 45 MIN
# =============================================================================

def nth_ugly_number(n):
    """
    PROBLEM: Ugly Number II
    
    An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.
    
    Given an integer n, return the nth ugly number.
    
    CONSTRAINTS:
    - 1 <= n <= 1690
    
    EXAMPLES:
    Example 1:
        Input: n = 10
        Output: 12
        Explanation: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12] is the sequence of the first 10 ugly numbers.
    
    Example 2:
        Input: n = 1
        Output: 1
        Explanation: 1 has no prime factors, therefore all of its prime factors are limited to 2, 3, and 5.
    
    APPROACH 1: Min Heap
    
    Generate ugly numbers in order using min heap
    
    TIME: O(n log n), SPACE: O(n)
    """
    heap = [1]
    seen = {1}
    
    for _ in range(n):
        ugly = heapq.heappop(heap)
        
        # Generate next ugly numbers
        for factor in [2, 3, 5]:
            new_ugly = ugly * factor
            if new_ugly not in seen:
                seen.add(new_ugly)
                heapq.heappush(heap, new_ugly)
    
    return ugly


def nth_ugly_number_dp(n):
    """
    APPROACH 2: Dynamic Programming (Optimal)
    
    Use three pointers to generate ugly numbers in order
    
    TIME: O(n), SPACE: O(n)
    """
    ugly = [1] * n
    i2 = i3 = i5 = 0
    
    next_2 = 2
    next_3 = 3
    next_5 = 5
    
    for i in range(1, n):
        ugly[i] = min(next_2, next_3, next_5)
        
        if ugly[i] == next_2:
            i2 += 1
            next_2 = ugly[i2] * 2
        
        if ugly[i] == next_3:
            i3 += 1
            next_3 = ugly[i3] * 3
        
        if ugly[i] == next_5:
            i5 += 1
            next_5 = ugly[i5] * 5
    
    return ugly[n - 1]


# =============================================================================
# PROBLEM 8: MEETING ROOMS II (MEDIUM) - 45 MIN
# =============================================================================

def min_meeting_rooms(intervals):
    """
    PROBLEM: Meeting Rooms II
    
    Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], 
    return the minimum number of conference rooms required.
    
    CONSTRAINTS:
    - 1 <= intervals.length <= 10^4
    - 0 <= starti < endi <= 10^6
    
    EXAMPLES:
    Example 1:
        Input: intervals = [[0,30],[5,10],[15,20]]
        Output: 2
    
    Example 2:
        Input: intervals = [[7,10],[2,4]]
        Output: 1
    
    APPROACH: Min Heap for End Times
    
    Track when each room becomes available
    
    TIME: O(n log n), SPACE: O(n)
    """
    if not intervals:
        return 0
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # Min heap to track end times of ongoing meetings
    heap = []
    
    for start, end in intervals:
        # Remove meetings that have ended
        while heap and heap[0] <= start:
            heapq.heappop(heap)
        
        # Add current meeting's end time
        heapq.heappush(heap, end)
    
    return len(heap)


# =============================================================================
# PROBLEM 9: REORGANIZE STRING (MEDIUM) - 45 MIN
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
    
    APPROACH: Max Heap with Greedy Selection
    
    Always place the most frequent available character
    
    TIME: O(n log k), SPACE: O(k) where k is number of unique characters
    """
    # Count character frequencies
    char_count = Counter(s)
    
    # Check if reorganization is possible
    max_freq = max(char_count.values())
    if max_freq > (len(s) + 1) // 2:
        return ""
    
    # Use max heap for frequencies
    heap = [(-freq, char) for char, freq in char_count.items()]
    heapq.heapify(heap)
    
    result = []
    prev_freq, prev_char = 0, ''
    
    while heap:
        # Get most frequent character
        freq, char = heapq.heappop(heap)
        result.append(char)
        
        # Put back previous character if it still has count
        if prev_freq < 0:
            heapq.heappush(heap, (prev_freq, prev_char))
        
        # Update for next iteration
        prev_freq, prev_char = freq + 1, char
    
    return ''.join(result) if len(result) == len(s) else "" 