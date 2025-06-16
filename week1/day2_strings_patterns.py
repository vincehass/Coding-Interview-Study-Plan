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


# Problem 1: Valid Palindrome - Introduction to string manipulation
def is_palindrome(s):
    """
    Check if string is palindrome ignoring non-alphanumeric characters
    
    Approach: Two pointers from ends, skip non-alphanumeric
    This introduces string processing and two pointers on strings
    
    Time: O(n), Space: O(1)
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


# Problem 2: Longest Substring Without Repeating Characters - Classic sliding window
def length_of_longest_substring(s):
    """
    Find length of longest substring without repeating characters
    
    Sliding Window Approach:
    1. Expand window by moving right pointer
    2. If duplicate found, contract from left until no duplicates
    3. Track maximum length seen
    
    This is the fundamental sliding window pattern
    
    Time: O(n), Space: O(min(m, n)) where m is charset size
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
    Optimized version using hash map to track last seen positions
    Instead of moving left pointer one by one, jump directly
    
    Time: O(n), Space: O(min(m, n))
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


# Problem 3: Minimum Window Substring - Advanced sliding window
def min_window(s, t):
    """
    Find minimum window in s that contains all characters of t
    
    Template for minimum window problems:
    1. Expand window until valid (contains all required)
    2. Contract window while maintaining validity
    3. Track minimum valid window
    
    Time: O(|s| + |t|), Space: O(|s| + |t|)
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


# Problem 4: Group Anagrams - Hash map with string patterns
def group_anagrams(strs):
    """
    Group strings that are anagrams of each other
    
    Approach 1: Sort each string as key
    Approach 2: Character frequency as key
    
    This bridges string processing to hash table usage
    
    Time: O(n * m log m) where n = number of strings, m = average length
    Space: O(n * m)
    """
    anagram_groups = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())


def group_anagrams_frequency(strs):
    """
    Alternative approach using character frequency as key
    
    Time: O(n * m) where n = number of strings, m = average length
    Space: O(n * m)
    """
    anagram_groups = defaultdict(list)
    
    for s in strs:
        # Create frequency tuple as key
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        key = tuple(count)
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())


# Problem 5: Valid Parentheses - Stack introduction (transitioning to next topic)
def is_valid_parentheses(s):
    """
    Check if parentheses are valid (properly paired and nested)
    
    This introduces stack concept for next day's study
    Stack pattern: Last in, first out for matching pairs
    
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


# ADVANCED PROBLEMS FOR EXTRA PRACTICE

def find_all_anagrams(s, p):
    """
    Find all start indices of anagrams of p in s
    
    Sliding window with fixed size |p|
    Demonstrates fixed-size sliding window pattern
    
    Time: O(|s|), Space: O(1) since alphabet size is constant
    """
    if len(p) > len(s):
        return []
    
    result = []
    p_count = Counter(p)
    window_count = Counter()
    
    # Initialize window
    for i in range(len(p)):
        window_count[s[i]] += 1
    
    # Check first window
    if window_count == p_count:
        result.append(0)
    
    # Slide window
    for i in range(len(p), len(s)):
        # Add new character
        window_count[s[i]] += 1
        
        # Remove old character
        left_char = s[i - len(p)]
        window_count[left_char] -= 1
        if window_count[left_char] == 0:
            del window_count[left_char]
        
        # Check if current window is anagram
        if window_count == p_count:
            result.append(i - len(p) + 1)
    
    return result


def longest_palindromic_substring(s):
    """
    Find longest palindromic substring
    
    Expand around centers approach
    Demonstrates palindrome checking patterns
    
    Time: O(n²), Space: O(1)
    """
    if not s:
        return ""
    
    start = 0
    max_len = 1
    
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    for i in range(len(s)):
        # Odd length palindromes (center at i)
        len1 = expand_around_center(i, i)
        # Even length palindromes (center between i and i+1)
        len2 = expand_around_center(i, i + 1)
        
        current_max = max(len1, len2)
        if current_max > max_len:
            max_len = current_max
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_len]


# COMPREHENSIVE TESTING SUITE
def test_all_problems():
    """
    Test all string problems with comprehensive test cases
    """
    print("=== TESTING DAY 2 PROBLEMS ===\n")
    
    # Test Valid Palindrome
    print("1. Valid Palindrome Tests:")
    palindrome_tests = [
        ("A man, a plan, a canal: Panama", True),
        ("race a car", False),
        ("", True),
        ("Madam", True),
        ("No 'x' in Nixon", True)
    ]
    
    for s, expected in palindrome_tests:
        result = is_palindrome(s)
        print(f"   Input: '{s}'")
        print(f"   Output: {result}, Expected: {expected}")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Longest Substring Without Repeating Characters
    print("2. Longest Substring Without Repeating Characters:")
    substring_tests = [
        ("abcabcbb", 3),  # "abc"
        ("bbbbb", 1),     # "b"
        ("pwwkew", 3),    # "wke"
        ("", 0),
        ("au", 2)
    ]
    
    for s, expected in substring_tests:
        result1 = length_of_longest_substring(s)
        result2 = length_of_longest_substring_optimized(s)
        print(f"   Input: '{s}'")
        print(f"   Basic: {result1}, Optimized: {result2}, Expected: {expected}")
        print(f"   ✓ Correct" if result1 == result2 == expected else f"   ✗ Wrong")
        print()
    
    # Test Minimum Window Substring
    print("3. Minimum Window Substring:")
    window_tests = [
        ("ADOBECODEBANC", "ABC", "BANC"),
        ("a", "a", "a"),
        ("a", "aa", ""),
        ("ab", "b", "b")
    ]
    
    for s, t, expected in window_tests:
        result = min_window(s, t)
        print(f"   s: '{s}', t: '{t}'")
        print(f"   Output: '{result}', Expected: '{expected}'")
        print(f"   ✓ Correct" if result == expected else f"   ✗ Wrong")
        print()
    
    # Test Group Anagrams
    print("4. Group Anagrams:")
    anagram_tests = [
        (["eat", "tea", "tan", "ate", "nat", "bat"], 
         [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]),
        ([""], [[""]]),
        (["a"], [["a"]])
    ]
    
    for strs, expected in anagram_tests:
        result1 = group_anagrams(strs)
        result2 = group_anagrams_frequency(strs)
        # Sort for comparison
        result1_sorted = [sorted(group) for group in sorted(result1)]
        result2_sorted = [sorted(group) for group in sorted(result2)]
        expected_sorted = [sorted(group) for group in sorted(expected)]
        
        print(f"   Input: {strs}")
        print(f"   Output (sorted): {result1_sorted}")
        print(f"   Expected (sorted): {expected_sorted}")
        success = result1_sorted == expected_sorted and result2_sorted == expected_sorted
        print(f"   ✓ Correct" if success else f"   ✗ Wrong")
        print()
    
    # Test Valid Parentheses
    print("5. Valid Parentheses:")
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


# EDUCATIONAL DEMONSTRATIONS
def demonstrate_sliding_window():
    """
    Visual demonstration of sliding window technique
    """
    print("\n=== SLIDING WINDOW DEMONSTRATION ===")
    s = "abcabcbb"
    print(f"Finding longest substring without repeating chars in: '{s}'")
    
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        print(f"\nStep {right + 1}: Processing '{s[right]}' at position {right}")
        
        # Show current window
        print(f"  Current window: '{s[left:right+1]}' (left={left}, right={right})")
        print(f"  Characters in set: {char_set}")
        
        # Contract if duplicate
        while s[right] in char_set:
            print(f"  Duplicate '{s[right]}' found! Removing '{s[left]}' from left")
            char_set.remove(s[left])
            left += 1
            print(f"  New window: '{s[left:right+1]}' (left={left})")
        
        char_set.add(s[right])
        current_length = right - left + 1
        max_length = max(max_length, current_length)
        
        print(f"  Added '{s[right]}', window length: {current_length}, max so far: {max_length}")


def pattern_matching_complexity_analysis():
    """
    Compare different approaches for pattern matching
    """
    print("\n=== PATTERN MATCHING COMPLEXITY ANALYSIS ===")
    
    print("\nProblem: Find substring pattern in text")
    print("\nNaive Approach:")
    print("  - Check each position: O(n*m) time")
    print("  - Space: O(1)")
    print("  - For each position, compare m characters")
    
    print("\nSliding Window (when applicable):")
    print("  - Single pass with hash map: O(n) time")
    print("  - Space: O(k) where k is pattern size")
    print("  - Maintains character frequencies")
    
    print("\nRolling Hash:")
    print("  - Compute hash in O(1) for each position: O(n) time")
    print("  - Space: O(1)")
    print("  - Collision handling needed")
    
    print("\nKMP Algorithm:")
    print("  - Preprocessing + matching: O(n + m) time")
    print("  - Space: O(m) for failure function")
    print("  - Optimal for exact pattern matching")


# STRING MANIPULATION UTILITIES
def string_utilities_demo():
    """
    Demonstrate common string manipulation techniques
    """
    print("\n=== STRING MANIPULATION TECHNIQUES ===")
    
    s = "Hello World"
    print(f"Original string: '{s}'")
    
    # Common operations
    print("\nCommon Operations:")
    print(f"  Length: {len(s)}")
    print(f"  Lowercase: '{s.lower()}'")
    print(f"  Uppercase: '{s.upper()}'")
    print(f"  Replace: '{s.replace('World', 'Python')}'")
    print(f"  Split: {s.split()}")
    print(f"  Strip whitespace: '{s.strip()}'")
    
    # Character operations
    print("\nCharacter Operations:")
    print(f"  First char: '{s[0]}'")
    print(f"  Last char: '{s[-1]}'")
    print(f"  Slice [0:5]: '{s[0:5]}'")
    print(f"  Reverse: '{s[::-1]}'")
    
    # ASCII operations
    print("\nASCII Operations:")
    print(f"  ord('A'): {ord('A')}")
    print(f"  chr(65): '{chr(65)}'")
    print(f"  'A' to index: {ord('A') - ord('A')}")  # 0-based indexing


if __name__ == "__main__":
    # Run all tests
    test_all_problems()
    
    # Educational demonstrations
    print("\n" + "="*70)
    print("EDUCATIONAL DEMONSTRATIONS")
    print("="*70)
    
    # Demonstrate sliding window
    demonstrate_sliding_window()
    
    # Show complexity analysis
    pattern_matching_complexity_analysis()
    
    # String utilities
    string_utilities_demo()
    
    print("\n" + "="*70)
    print("DAY 2 COMPLETE - KEY TAKEAWAYS:")
    print("="*70)
    print("1. Sliding window reduces O(n²) to O(n) for substring problems")
    print("2. Two types: fixed-size and variable-size windows")
    print("3. Hash maps for character frequency counting")
    print("4. String immutability in Python affects space complexity")
    print("5. Palindrome checking with two pointers")
    print("6. Anagram detection using sorting or frequency counting")
    print("7. Stack pattern introduced for parentheses matching")
    print("\nTransition: Day 2→3 - From strings to hash tables")
    print("- Character frequency → Hash map operations")
    print("- Sliding window + hash map → Advanced hash table usage")
    print("\nNext: Day 3 - Hash Tables & Sets") 