"""
=============================================================================
                        DAY 17: TRIES & STRING ALGORITHMS
                           Meta Interview Preparation
                              Week 3 - Day 17
=============================================================================

FOCUS: Prefix trees, string matching
TIME ALLOCATION: 4 hours
- Theory (1 hour): Trie construction, string algorithms
- Problems (3 hours): Trie and string matching problems

TOPICS COVERED:
- Trie construction and operations
- Prefix matching applications
- Space-time tradeoffs
- String search algorithms

=============================================================================
"""

from typing import List, Dict, Set
from collections import defaultdict


# =============================================================================
# THEORY SECTION (1 HOUR)
# =============================================================================

"""
TRIE (PREFIX TREE):

Structure:
- Tree where each node represents a character
- Root represents empty string
- Path from root to node represents a prefix
- Leaf nodes (or marked nodes) represent complete words

Operations:
- INSERT: O(m) where m is word length
- SEARCH: O(m) where m is word length
- PREFIX_SEARCH: O(p) where p is prefix length

Applications:
- Autocomplete systems
- Spell checkers
- Word games
- IP routing tables
- Efficient string matching
"""


# =============================================================================
# TRIE IMPLEMENTATION
# =============================================================================

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None  # Store complete word for some problems


class Trie:
    """
    PROBLEM: Implement Trie (Prefix Tree)
    
    Implement a trie with insert, search, and startsWith methods.
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """Insert a word into the trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word
    
    def search(self, word: str) -> bool:
        """Returns true if word is in the trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix: str) -> bool:
        """Returns true if there is any word that starts with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


# =============================================================================
# PROBLEM 1: IMPLEMENT TRIE (MEDIUM) - 45 MIN
# =============================================================================

# Already implemented above as the Trie class


# =============================================================================
# PROBLEM 2: WORD SEARCH II (HARD) - 75 MIN
# =============================================================================

def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    """
    PROBLEM: Word Search II
    
    Given an m x n board of characters and a list of strings words, 
    return all words on the board.
    
    Each word must be constructed from letters of sequentially adjacent cells, 
    where adjacent cells are horizontally or vertically neighboring.
    
    Example:
    Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], 
           words = ["oath","pea","eat","rain"]
    Output: ["eat","oath"]
    
    TIME: O(M * N * 4^L), SPACE: O(W * L)
    """
    # Build trie from words
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    rows, cols = len(board), len(board[0])
    result = set()
    
    def dfs(r, c, node, visited):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if (r, c) in visited:
            return
        
        char = board[r][c]
        if char not in node.children:
            return
        
        node = node.children[char]
        if node.is_end and node.word:
            result.add(node.word)
        
        visited.add((r, c))
        
        # Explore 4 directions
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            dfs(r + dr, c + dc, node, visited)
        
        visited.remove((r, c))
    
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root, set())
    
    return list(result)


# =============================================================================
# PROBLEM 3: ADD AND SEARCH WORD (MEDIUM) - 45 MIN
# =============================================================================

class WordDictionary:
    """
    PROBLEM: Design Add and Search Words Data Structure
    
    Design a data structure that supports adding new words and finding if 
    a string matches any previously added string. The search can contain '.' 
    which can match any letter.
    
    TIME: O(m) for add, O(m * 26^k) for search where k is number of dots
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def add_word(self, word: str) -> None:
        """Add a word to the data structure"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        """Search for a word (may contain '.' wildcards)"""
        def dfs(node, index):
            if index == len(word):
                return node.is_end
            
            char = word[index]
            if char == '.':
                # Try all possible characters
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)
        
        return dfs(self.root, 0)


# =============================================================================
# PROBLEM 4: LONGEST COMMON PREFIX (EASY) - 30 MIN
# =============================================================================

def longest_common_prefix(strs: List[str]) -> str:
    """
    PROBLEM: Longest Common Prefix
    
    Write a function to find the longest common prefix string amongst 
    an array of strings. If there is no common prefix, return "".
    
    Example:
    Input: strs = ["flower","flow","flight"]
    Output: "fl"
    
    TIME: O(S) where S is sum of all characters, SPACE: O(1)
    """
    if not strs:
        return ""
    
    # Start with first string as potential prefix
    prefix = strs[0]
    
    for i in range(1, len(strs)):
        # Reduce prefix until it matches current string
        while not strs[i].startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix


def longest_common_prefix_trie(strs: List[str]) -> str:
    """
    VARIANT: Using Trie approach
    TIME: O(S), SPACE: O(S)
    """
    if not strs:
        return ""
    
    trie = Trie()
    for s in strs:
        trie.insert(s)
    
    # Traverse trie while there's only one path
    node = trie.root
    prefix = ""
    
    while len(node.children) == 1 and not node.is_end:
        char = next(iter(node.children))
        prefix += char
        node = node.children[char]
    
    return prefix


# =============================================================================
# PROBLEM 5: REPLACE WORDS (MEDIUM) - 45 MIN
# =============================================================================

def replace_words(dictionary: List[str], sentence: str) -> str:
    """
    PROBLEM: Replace Words
    
    In English, we have a concept called root, which can be followed by some 
    other word to form another longer word - let's call this word successor.
    
    Given a dictionary consisting of many roots and a sentence consisting of 
    words separated by spaces, replace all the successors in the sentence 
    with the root forming it.
    
    Example:
    Input: dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
    Output: "the cat was rat by the bat"
    
    TIME: O(D + S), SPACE: O(D)
    """
    trie = Trie()
    for root in dictionary:
        trie.insert(root)
    
    def find_root(word):
        node = trie.root
        prefix = ""
        
        for char in word:
            if char not in node.children:
                return word  # No root found
            node = node.children[char]
            prefix += char
            if node.is_end:
                return prefix  # Found a root
        
        return word  # No root found
    
    words = sentence.split()
    return " ".join(find_root(word) for word in words)


# =============================================================================
# PRACTICE PROBLEMS & TEST CASES
# =============================================================================

def test_day17_problems():
    """Test all Day 17 problems"""
    
    print("=" * 60)
    print("         DAY 17: TRIES & STRING ALGORITHMS")
    print("=" * 60)
    
    # Test Trie Implementation
    print("\n🧪 Testing Trie Implementation")
    trie = Trie()
    trie.insert("apple")
    print(f"Insert 'apple'")
    print(f"Search 'apple': {trie.search('apple')} (Expected: True)")
    print(f"Search 'app': {trie.search('app')} (Expected: False)")
    print(f"Starts with 'app': {trie.starts_with('app')} (Expected: True)")
    
    trie.insert("app")
    print(f"Insert 'app'")
    print(f"Search 'app': {trie.search('app')} (Expected: True)")
    
    # Test Word Search II
    print("\n🧪 Testing Word Search II")
    board = [
        ["o","a","a","n"],
        ["e","t","a","e"],
        ["i","h","k","r"],
        ["i","f","l","v"]
    ]
    words = ["oath","pea","eat","rain"]
    found_words = find_words(board, words)
    print(f"Word Search II: {found_words} (Expected: ['eat', 'oath'])")
    
    # Test Add and Search Word
    print("\n🧪 Testing Add and Search Word")
    word_dict = WordDictionary()
    word_dict.add_word("bad")
    word_dict.add_word("dad")
    word_dict.add_word("mad")
    
    print(f"Search 'pad': {word_dict.search('pad')} (Expected: False)")
    print(f"Search 'bad': {word_dict.search('bad')} (Expected: True)")
    print(f"Search '.ad': {word_dict.search('.ad')} (Expected: True)")
    print(f"Search 'b..': {word_dict.search('b..')} (Expected: True)")
    
    # Test Longest Common Prefix
    print("\n🧪 Testing Longest Common Prefix")
    lcp1 = longest_common_prefix(["flower","flow","flight"])
    print(f"LCP ['flower','flow','flight']: '{lcp1}' (Expected: 'fl')")
    
    lcp2 = longest_common_prefix_trie(["dog","racecar","car"])
    print(f"LCP Trie ['dog','racecar','car']: '{lcp2}' (Expected: '')")
    
    # Test Replace Words
    print("\n🧪 Testing Replace Words")
    replaced = replace_words(["cat","bat","rat"], "the cattle was rattled by the battery")
    print(f"Replace Words: '{replaced}'")
    print(f"Expected: 'the cat was rat by the bat'")
    
    print("\n" + "=" * 60)
    print("           DAY 17 TESTING COMPLETED")
    print("=" * 60)


# =============================================================================
# DAILY PRACTICE ROUTINE
# =============================================================================

def main():
    """
    Day 17 Practice Routine:
    1. Review theory (1 hour)
    2. Solve problems (3 hours)
    3. Test solutions
    4. Review and optimize
    """
    
    print("🚀 Starting Day 17: Tries & String Algorithms")
    print("\n📚 Theory Topics:")
    print("- Trie construction and operations")
    print("- Prefix matching applications")
    print("- Space-time tradeoffs")
    print("- String search optimization")
    
    print("\n💻 Practice Problems:")
    print("1. Implement Trie (Medium) - 45 min")
    print("2. Word Search II (Hard) - 75 min")
    print("3. Add and Search Word (Medium) - 45 min")
    print("4. Longest Common Prefix (Easy) - 30 min")
    print("5. Replace Words (Medium) - 45 min")
    
    print("\n🧪 Running Tests...")
    test_day17_problems()
    
    print("\n✅ Day 17 Complete!")
    print("📈 Next: Day 18 - Week 3 Review & Complex Problems")


if __name__ == "__main__":
    main() 