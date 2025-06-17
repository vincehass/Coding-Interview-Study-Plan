"""
=============================================================================
                    WEEK 1 - DAY 2: STRINGS & PATTERN MATCHING
                           Meta Interview Preparation
=============================================================================

THEORY SECTION (1 Hour)
======================

1. STRING FUNDAMENTALS
   - Strings are immutable in Python (creates new string on modification)
   - Common operations: slicing O(k), concatenation O(n+m), search O(n*m)
   - String comparison: lexicographic ordering, case sensitivity

2. SLIDING WINDOW TECHNIQUE
   - Efficient pattern for substring problems
   - Two types:
     a) Fixed size window: maintain window of constant size
     b) Variable size window: expand/contract based on conditions
   - Reduces time from O(n²) or O(n³) to O(n)

3. SLIDING WINDOW PATTERNS:
   - Longest substring with unique characters
   - Minimum window covering substring
   - Maximum sum subarray of size k
   - Find all anagrams in string

4. PATTERN MATCHING STRATEGIES:
   - Character frequency counting (use hash maps)
   - Two pointers for palindrome checking
   - Rolling hash for efficient pattern search
   - KMP algorithm for pattern matching (advanced)

5. TRANSITION TO HASH TABLES:
   - Character frequency → Hash map
   - Anagram detection → Sorted string or frequency map
   - Substring problems → Sliding window + hash map

=============================================================================
"""

from collections import defaultdict, Counter


# =============================================================================
# PROBLEM 1: VALID PALINDROME (EASY) - 30 MIN
# =============================================================================

def is_palindrome(s):
    """
    PROBLEM: Valid Palindrome
    
    A phrase is a palindrome if, after converting all uppercase letters into 
    lowercase letters and removing all non-alphanumeric characters, it reads 
    the same forward and backward.
    
    Given a string s, return true if it is a palindrome, or false otherwise.
    
    CONSTRAINTS:
    - 1 <= s.length <= 2 * 10^5
    - s consists only of printable ASCII characters
    
    EXAMPLES:
    Example 1:
        Input: s = "A man, a plan, a canal: Panama"
        Output: true
        Explanation: "amanaplanacanalpanama" is a palindrome
    
    Example 2:
        Input: s = "race a car"
        Output: false
        Explanation: "raceacar" is not a palindrome
    
    Example 3:
        Input: s = " "
        Output: true
        Explanation: After removing non-alphanumeric characters, s becomes an empty string
    
    APPROACH: Two Pointers
    
    Use two pointers from both ends, skip non-alphanumeric characters,
    and compare characters case-insensitively.
    
    TIME: O(n), SPACE: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1
        
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True


# =============================================================================
# PROBLEM 2: LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS (MEDIUM) - 45 MIN
# =============================================================================

def length_of_longest_substring(s):
    """
    PROBLEM: Longest Substring Without Repeating Characters
    
    Given a string s, find the length of the longest substring without 
    repeating characters.
    
    CONSTRAINTS:
    - 0 <= s.length <= 5 * 10^4
    - s consists of English letters, digits, symbols and spaces
    
    EXAMPLES:
    Example 1:
        Input: s = "abcabcbb"
        Output: 3
        Explanation: The answer is "abc", with the length of 3
    
    Example 2:
        Input: s = "bbbbb"
        Output: 1
        Explanation: The answer is "b", with the length of 1
    
    Example 3:
        Input: s = "pwwkew"
        Output: 3
        Explanation: The answer is "wke", with the length of 3
    
    APPROACH: Sliding Window with Set
    
    1. Expand window by moving right pointer
    2. If duplicate found, contract from left until no duplicates
    3. Track maximum length seen
    
    TIME: O(n), SPACE: O(min(m, n)) where m is charset size
    """
    if not s:
        return 0
    
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Contract window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        # Add current character and update max
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length


def length_of_longest_substring_optimized(s):
    """
    APPROACH: Sliding Window with HashMap (Optimized)
    
    Instead of moving left pointer one by one, jump directly to position
    after the last occurrence of the duplicate character.
    
    TIME: O(n), SPACE: O(min(m, n))
    """
    char_map = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            # Jump left pointer to position after last occurrence
            left = char_map[s[right]] + 1
        
        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length


# =============================================================================
# PROBLEM 3: MINIMUM WINDOW SUBSTRING (HARD) - 60 MIN
# =============================================================================

def min_window(s, t):
    """
    PROBLEM: Minimum Window Substring
    
    Given two strings s and t of lengths m and n respectively, return the 
    minimum window substring of s such that every character in t (including 
    duplicates) is included in the window. If there is no such substring, 
    return the empty string "".
    
    CONSTRAINTS:
    - m == s.length
    - n == t.length
    - 1 <= m, n <= 10^5
    - s and t consist of uppercase and lowercase English letters
    
    EXAMPLES:
    Example 1:
        Input: s = "ADOBECODEBANC", t = "ABC"
        Output: "BANC"
        Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t
    
    Example 2:
        Input: s = "a", t = "a"
        Output: "a"
        Explanation: The entire string s is the minimum window
    
    Example 3:
        Input: s = "a", t = "aa"
        Output: ""
        Explanation: Both 'a's from t must be included in the window
    
    APPROACH: Sliding Window with Character Frequency
    
    Template for minimum window problems:
    1. Expand window until valid (contains all required)
    2. Contract window while maintaining validity
    3. Track minimum valid window
    
    TIME: O(|s| + |t|), SPACE: O(|s| + |t|)
    """
    if not s or not t or len(s) < len(t):
        return ""
    
    # Count characters in t
    t_count = Counter(t)
    required = len(t_count)
    
    # Sliding window variables
    left = right = 0
    formed = 0  # Number of unique chars in current window with desired frequency
    
    # Dictionary to keep count of characters in current window
    window_counts = defaultdict(int)
    
    # Result: (window length, left, right)
    ans = float('inf'), None, None
    
    while right < len(s):
        # Add character from right to window
        char = s[right]
        window_counts[char] += 1
        
        # Check if this character's frequency matches desired frequency
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        # Contract window from left
        while left <= right and formed == required:
            char = s[left]
            
            # Update result if this window is smaller
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            
            # Remove character from left
            window_counts[char] -= 1
            if char in t_count and window_counts[char] < t_count[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]


# =============================================================================
# PROBLEM 4: GROUP ANAGRAMS (MEDIUM) - 45 MIN
# =============================================================================

def group_anagrams(strs):
    """
    PROBLEM: Group Anagrams
    
    Given an array of strings strs, group the anagrams together. You can 
    return the answer in any order.
    
    An Anagram is a word or phrase formed by rearranging the letters of a 
    different word or phrase, typically using all the original letters exactly once.
    
    CONSTRAINTS:
    - 1 <= strs.length <= 10^4
    - 0 <= strs[i].length <= 100
    - strs[i] consists of lowercase English letters
    
    EXAMPLES:
    Example 1:
        Input: strs = ["eat","tea","tan","ate","nat","bat"]
        Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
    
    Example 2:
        Input: strs = [""]
        Output: [[""]]
    
    Example 3:
        Input: strs = ["a"]
        Output: [["a"]]
    
    APPROACH 1: Sort Each String as Key
    
    Anagrams will have the same sorted string. Use sorted string as key
    in hash map to group anagrams together.
    
    TIME: O(n * m log m) where n = number of strings, m = average length
    SPACE: O(n * m)
    """
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Sort the string to create a key
        key = ''.join(sorted(s))
        anagram_map[key].append(s)
    
    return list(anagram_map.values())


def group_anagrams_frequency(strs):
    """
    APPROACH 2: Character Frequency as Key
    
    Use character frequency tuple as key instead of sorting.
    Can be more efficient for very long strings.
    
    TIME: O(n * m) where n = number of strings, m = average length
    SPACE: O(n * m)
    """
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Create frequency array for 26 lowercase letters
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        
        # Use tuple of counts as key
        key = tuple(count)
        anagram_map[key].append(s)
    
    return list(anagram_map.values())


# =============================================================================
# PROBLEM 5: VALID PARENTHESES (EASY) - 30 MIN
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
# COMPREHENSIVE TEST CASES
# =============================================================================

def test_all_problems():
    """
    Test all implemented solutions with comprehensive test cases
    """
    print("=" * 60)
    print("           WEEK 1 - DAY 2: TESTING RESULTS")
    print("=" * 60)
    
    # Test Valid Palindrome
    print("\n🧪 PROBLEM 1: VALID PALINDROME")
    test_cases_palindrome = [
        ("A man, a plan, a canal: Panama", True),
        ("race a car", False),
        (" ", True),
        ("", True),
        ("Madam", True)
    ]
    
    for i, (s, expected) in enumerate(test_cases_palindrome, 1):
        result = is_palindrome(s)
        print(f"   Test Case {i}:")
        print(f"   Input: s = \"{s}\"")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print(f"   ✅ PASS" if result == expected else f"   ❌ FAIL")
        print()
    
    # Test Longest Substring Without Repeating Characters
    print("🧪 PROBLEM 2: LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS")
    test_cases_longest = [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("", 0),
        ("dvdf", 3)
    ]
    
    for i, (s, expected) in enumerate(test_cases_longest, 1):
        result1 = length_of_longest_substring(s)
        result2 = length_of_longest_substring_optimized(s)
        print(f"   Test Case {i}:")
        print(f"   Input: s = \"{s}\"")
        print(f"   Expected: {expected}")
        print(f"   Basic: {result1}, Optimized: {result2}")
        print(f"   ✅ PASS" if result1 == expected and result2 == expected else f"   ❌ FAIL")
        print()
    
    # Test Minimum Window Substring
    print("🧪 PROBLEM 3: MINIMUM WINDOW SUBSTRING")
    test_cases_min_window = [
        ("ADOBECODEBANC", "ABC", "BANC"),
        ("a", "a", "a"),
        ("a", "aa", ""),
        ("ab", "b", "b")
    ]
    
    for i, (s, t, expected) in enumerate(test_cases_min_window, 1):
        result = min_window(s, t)
        print(f"   Test Case {i}:")
        print(f"   Input: s = \"{s}\", t = \"{t}\"")
        print(f"   Expected: \"{expected}\"")
        print(f"   Got: \"{result}\"")
        print(f"   ✅ PASS" if result == expected else f"   ❌ FAIL")
        print()
    
    # Test Group Anagrams
    print("🧪 PROBLEM 4: GROUP ANAGRAMS")
    test_cases_anagrams = [
        (["eat","tea","tan","ate","nat","bat"], [["bat"],["nat","tan"],["ate","eat","tea"]]),
        ([""], [[""]]),
        (["a"], [["a"]])
    ]
    
    for i, (strs, expected) in enumerate(test_cases_anagrams, 1):
        result = group_anagrams(strs)
        # Sort for comparison since order doesn't matter
        result_sorted = [sorted(group) for group in sorted(result)]
        expected_sorted = [sorted(group) for group in sorted(expected)]
        
        print(f"   Test Case {i}:")
        print(f"   Input: strs = {strs}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print(f"   ✅ PASS" if result_sorted == expected_sorted else f"   ❌ FAIL")
        print()
    
    # Test Valid Parentheses
    print("🧪 PROBLEM 5: VALID PARENTHESES")
    test_cases_parentheses = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True)
    ]
    
    for i, (s, expected) in enumerate(test_cases_parentheses, 1):
        result = is_valid_parentheses(s)
        print(f"   Test Case {i}:")
        print(f"   Input: s = \"{s}\"")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print(f"   ✅ PASS" if result == expected else f"   ❌ FAIL")
        print()


# =============================================================================
# EDUCATIONAL HELPER FUNCTIONS
# =============================================================================

def demonstrate_sliding_window():
    """
    Visual demonstration of sliding window technique
    """
    print("\n" + "=" * 60)
    print("               SLIDING WINDOW DEMONSTRATION")
    print("=" * 60)
    
    s = "abcabcbb"
    print(f"🔍 Finding longest substring without repeating characters in: \"{s}\"")
    
    char_set = set()
    left = 0
    max_length = 0
    max_substring = ""
    
    for right in range(len(s)):
        print(f"\nStep {right + 1}: Processing s[{right}] = '{s[right]}'")
        
        # Contract window until no duplicates
        while s[right] in char_set:
            print(f"  Duplicate found! Removing s[{left}] = '{s[left]}'")
            char_set.remove(s[left])
            left += 1
        
        # Add current character
        char_set.add(s[right])
        current_length = right - left + 1
        current_substring = s[left:right+1]
        
        print(f"  Current window: [{left}, {right}] = \"{current_substring}\"")
        print(f"  Length: {current_length}")
        
        if current_length > max_length:
            max_length = current_length
            max_substring = current_substring
            print(f"  🎯 New maximum length: {max_length}")
    
    print(f"\n✅ Final result: \"{max_substring}\" with length {max_length}")


def pattern_matching_complexity_analysis():
    """
    Analysis of different string pattern matching approaches
    """
    print("\n" + "=" * 60)
    print("              PATTERN MATCHING COMPLEXITY")
    print("=" * 60)
    
    print("\n📊 SUBSTRING SEARCH ALGORITHMS:")
    print("\n🔴 Naive Approach:")
    print("  - Check every position: O(n*m)")
    print("  - Space: O(1)")
    print("  - Simple but inefficient")
    
    print("\n🟡 Sliding Window:")
    print("  - For specific patterns: O(n)")
    print("  - Space: O(k) where k is window size")
    print("  - Great for constraint-based problems")
    
    print("\n🟢 KMP Algorithm:")
    print("  - General pattern search: O(n + m)")
    print("  - Space: O(m)")
    print("  - Optimal for exact pattern matching")
    
    print("\n🔵 Rolling Hash:")
    print("  - Average case: O(n + m)")
    print("  - Space: O(1)")
    print("  - Good for multiple pattern search")


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*60)
    print("               EDUCATIONAL DEMONSTRATIONS")
    print("="*60)
    
    # Demonstrate sliding window
    demonstrate_sliding_window()
    
    # Show complexity analysis
    pattern_matching_complexity_analysis()
    
    print("\n" + "="*60)
    print("                   DAY 2 COMPLETE")
    print("="*60)
    print("🎯 KEY TAKEAWAYS:")
    print("1. Sliding window reduces O(n²) to O(n) for substring problems")
    print("2. Use hash maps for character frequency tracking")
    print("3. Two pointers work well for palindrome problems")
    print("4. Stack is perfect for bracket matching problems")
    print("5. Anagrams can be detected by sorting or frequency counting")
    print("6. Always consider string immutability in Python")
    print("\n🚀 NEXT: Day 3 - Hash Tables & Sets") 