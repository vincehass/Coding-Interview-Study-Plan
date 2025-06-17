# Coding-Interview-Study-Plan

This 4-week intensive preparation plan is specifically tailored for Meta technical interviews, focusing on data structures and algorithms with comprehensive dynamic programming coverage. The plan allocates 4 hours daily across 6 days per week (24 hours/week, 96 hours total).

# Meta Technical Interview Preparation - 4 Week Study Plan

## Repository Structure

```
meta-interview-prep/
├── README.md
├── week1/
│   ├── day1/
│   ├── day2/
│   ├── day3/
│   ├── day4/
│   ├── day5/
│   └── day6/
├── week2/
├── week3/
│   ├── day13_graph_representation_bfs.py
│   ├── day14_graph_dfs_backtracking.py
│   ├── day15_graph_algorithms.py
│   ├── day16_union_find_advanced.py
│   ├── day17_tries_string_algorithms.py
│   └── day18_week3_review_complex.py
├── week4/
│   ├── day19_1d_dynamic_programming.py
│   ├── day20_2d_dynamic_programming.py
│   ├── day21_advanced_dp_patterns.py
│   ├── day22_dp_optimization_techniques.py
│   ├── day23_dp_trees_graphs.py
│   └── day24_final_review_mock_interviews.py
├── practice/
│   ├── week1/
│   │   ├── solution.py        # Complete Week 1 solutions with variants
│   │   └── single_problem_practice.py
│   ├── week2/
│   │   └── solution.py        # Complete Week 2 solutions with variants
│   ├── week3/
│   │   └── solution.py        # Complete Week 3 solutions with variants
│   └── week4/
│       └── solution.py        # Complete Week 4 solutions with variants
├── solutions/
│   ├── week1/
│   ├── week2/
│   ├── week3/
│   └── week4/
├── templates/
└── resources/
```

## Overview

This 4-week intensive preparation plan is specifically tailored for Meta technical interviews, focusing on data structures and algorithms with comprehensive dynamic programming coverage. The plan allocates 4 hours daily across 6 days per week (24 hours/week, 96 hours total).

## 🎯 Complete Solution Files Available

**IMPORTANT**: This repository now includes comprehensive solution files for all 4 weeks with extensive variants, test cases, and multiple approaches for each problem. These files provide production-ready code that can be run immediately for interview practice.

### ✅ Week 1 Solution File: `practice/week1/solution.py`

**Topics**: Arrays, Hash Tables, Strings, Linked Lists, Stacks/Queues

**Problems & Variants**:

- **Two Sum** (with Two Sum All Pairs, Two Sum Sorted variants)
- **Three Sum** (with Three Sum Closest, Four Sum variants)
- **Container With Most Water** (with Trapping Rain Water variant)
- **Longest Substring Without Repeating Characters** (with K distinct characters, repeating character replacement variants)
- **Group Anagrams** (with character counting optimization)
- **Valid Parentheses** (with minimum remove to make valid variant)
- **Top K Frequent Elements** (heap and bucket sort approaches)
- **Longest Consecutive Sequence** (with path return variant)
- **Reverse Linked List** (iterative, recursive, and range reversal variants)
- **Merge Two Sorted Lists** (with Merge K Sorted Lists variant)
- **Linked List Cycle** (detection and finding start position)
- **Daily Temperatures** (with Next Greater Element variants)
- **Sliding Window Maximum** (with Sliding Window Minimum variant)

**Features**:

- 50+ comprehensive test cases with edge cases
- Multiple approaches per problem (O(n) vs O(n²), different data structures)
- Detailed time/space complexity analysis
- Educational comments explaining algorithms
- Self-contained utility functions for testing

**Verification**: ✅ All 50+ test cases passed successfully

### ✅ Week 2 Solution File: `practice/week2/solution.py`

**Topics**: Binary Trees, BSTs, Binary Search, Heaps

**Problems & Variants**:

- **Tree Traversals**: Inorder (recursive & iterative), Preorder, Postorder, Level Order
- **Tree Properties**: Max/Min Depth, Same Tree, Symmetric Tree, Path Sum (with all paths variant), Diameter
- **BST Operations**: Validate BST (bounds & inorder approaches), Search BST (recursive & iterative), Insert/Delete
- **BST Advanced**: Kth Smallest Element, BST to Sorted Array, Lowest Common Ancestor
- **Binary Search**: Basic Binary Search (recursive & iterative), Search Range, Search Insert Position
- **Rotated Arrays**: Search in Rotated Array, Find Minimum in Rotated Array
- **Binary Search Variants**: Find Peak Element, Integer Square Root
- **Heap Problems**: Kth Largest Element (heap & quickselect), Top K Frequent (heap & bucket sort)
- **Advanced Heap**: Median Finder, Merge K Sorted Lists, Sliding Window Median

**Features**:

- 40+ comprehensive test cases covering all tree/search scenarios
- TreeNode class with utility functions for tree creation/manipulation
- Multiple algorithmic approaches (recursive vs iterative, different optimizations)
- Complete heap implementations with multiple strategies
- Self-contained testing framework

**Verification**: ✅ All 40+ test cases passed successfully

### ✅ Week 3 Solution File: `practice/week3/solution.py`

**Topics**: Graph Traversal, Shortest Path, Union Find, Trie, Advanced Graph Problems

**Problems & Variants**:

- **Graph Traversal**: Number of Islands (DFS & BFS), Max Area of Island, Surrounded Regions
- **Graph Problems**: Clone Graph (DFS & BFS), Course Schedule I & II (cycle detection & topological sort)
- **Advanced Traversal**: Pacific Atlantic Water Flow, Word Ladder (BFS & bidirectional BFS)
- **Shortest Path**: Shortest Path in Binary Matrix, Network Delay Time (Dijkstra), Cheapest Flights (Bellman-Ford)
- **Union Find**: Complete implementation with path compression and union by rank
- **Union Find Applications**: Connected Components, Accounts Merge, Redundant Connection
- **Trie**: Complete Trie implementation with search/prefix operations
- **Trie Applications**: Word Search II, Replace Words, Word Search I
- **Advanced Graph**: Alien Dictionary (topological sort), Critical Connections (Tarjan's algorithm)
- **Graph Algorithms**: Minimum Spanning Tree (Kruskal), Strongly Connected Components (Kosaraju)

**Features**:

- 35+ comprehensive test cases covering all graph scenarios
- Complete Union Find implementation with optimizations
- Full Trie implementation with applications
- Advanced graph algorithms (Tarjan's, Kosaraju's, Kruskal's)
- Utility functions for graph creation and manipulation

**Verification**: ✅ All 35+ test cases passed successfully

### ✅ Week 4 Solution File: `practice/week4/solution.py`

**Topics**: Dynamic Programming (1D, 2D, Advanced patterns)

**Problems & Variants**:

- **1D DP**: Climbing Stairs (with K steps variant), House Robber I & II (circular), Decode Ways
- **Coin Problems**: Coin Change I & II (minimum coins vs combinations)
- **Subsequences**: Longest Increasing Subsequence (O(n log n) & O(n²) approaches)
- **2D DP**: Unique Paths (with obstacles variant), Longest Common Subsequence, Edit Distance
- **Grid Problems**: Minimum Path Sum, Maximal Square (space-optimized)
- **String DP**: Word Break I & II (memoization), Longest Palindromic Substring
- **Advanced Patterns**: Maximum Subarray (Kadane's), Maximum Product Subarray
- **Counting DP**: Palindromic Substrings, Target Sum (0/1 knapsack variant)
- **Subset Problems**: Partition Equal Subset Sum

**Features**:

- 30+ comprehensive test cases covering all DP patterns
- Space-optimized implementations (2D to 1D reductions)
- Multiple DP approaches (top-down vs bottom-up)
- Advanced optimization techniques
- Clear complexity analysis for each approach

**Verification**: ✅ All 30+ test cases passed successfully

## 🎯 NEW: Complete Daily Practice Files

### ✅ Week 3 Daily Practice Files

**Complete 6-day structured practice with comprehensive problem sets:**

#### **Day 13: Graph Representation & BFS** (`week3/day13_graph_representation_bfs.py`)

- **Theory**: Graph basics, BFS algorithm implementation
- **Problems**: Number of Islands, Binary Tree Level Order Traversal, Word Ladder, Rotting Oranges, Clone Graph
- **Features**: Complete BFS implementations, graph utilities, comprehensive test cases

#### **Day 14: Graph DFS & Backtracking** (`week3/day14_graph_dfs_backtracking.py`)

- **Theory**: DFS algorithms, backtracking patterns
- **Problems**: Path Sum II, Generate Parentheses, Letter Combinations, Word Search, N-Queens
- **Features**: Recursive DFS, backtracking templates, state management

#### **Day 15: Graph Algorithms** (`week3/day15_graph_algorithms.py`)

- **Theory**: Cycle detection, topological sorting
- **Problems**: Course Schedule I & II, Bipartite Graph Detection, Alien Dictionary
- **Features**: Advanced graph algorithms, cycle detection, topological sort

#### **Day 16: Union-Find & Advanced Graph** (`week3/day16_union_find_advanced.py`)

- **Theory**: Union-Find with path compression and union by rank
- **Problems**: Connected Components, Accounts Merge, Redundant Connection, Most Stones Removed
- **Features**: Optimized Union-Find implementation, connectivity problems

#### **Day 17: Tries & String Algorithms** (`week3/day17_tries_string_algorithms.py`)

- **Theory**: Trie implementation and operations
- **Problems**: Word Search II, Add and Search Word, Longest Common Prefix, Replace Words
- **Features**: Complete Trie implementation, prefix matching, string algorithms

#### **Day 18: Week 3 Review & Complex Problems** (`week3/day18_week3_review_complex.py`)

- **Integration**: Multi-concept problems combining graphs, tries, and advanced algorithms
- **Problems**: Critical Connections (Tarjan's), Minimum Height Trees, Word Ladder II, Graph Valid Tree
- **Features**: Advanced algorithms, comprehensive review, mock interview preparation

### ✅ Week 4 Daily Practice Files

**Complete 6-day Dynamic Programming mastery with advanced techniques:**

#### **Day 19: 1D Dynamic Programming** (`week4/day19_1d_dynamic_programming.py`)

- **Theory**: DP fundamentals, memoization vs tabulation
- **Problems**: Climbing Stairs, House Robber (I & II), Maximum Subarray, Coin Change, Longest Increasing Subsequence
- **Features**: Multiple DP approaches, space optimization, classic 1D patterns

#### **Day 20: 2D Dynamic Programming** (`week4/day20_2d_dynamic_programming.py`)

- **Theory**: Grid-based DP, string matching algorithms
- **Problems**: Unique Paths, Minimum Path Sum, Longest Common Subsequence, Edit Distance, Maximal Square
- **Features**: Grid traversal DP, string matching patterns, space optimization techniques

#### **Day 21: Advanced DP Patterns** (`week4/day21_advanced_dp_patterns.py`)

- **Theory**: Interval DP, state machine DP, complex patterns
- **Problems**: Palindromic Substrings, Decode Ways, Stock with Cooldown, Burst Balloons, Regular Expression Matching
- **Features**: Advanced DP techniques, interval DP, pattern matching algorithms

#### **Day 22: DP Optimization Techniques** (`week4/day22_dp_optimization_techniques.py`)

- **Theory**: Space/time optimization, matrix exponentiation, monotonic deque
- **Problems**: Perfect Squares, Maximal Rectangle, Sliding Window Maximum, Fibonacci Matrix, Largest Divisible Subset
- **Features**: Mathematical optimizations, advanced data structures, complexity reduction

#### **Day 23: DP on Trees and Graphs** (`week4/day23_dp_trees_graphs.py`)

- **Theory**: Tree DP patterns, graph state management
- **Problems**: Binary Tree Maximum Path Sum, House Robber III, Diameter of Binary Tree, Longest Univalue Path
- **Features**: Tree traversal DP, subtree optimization, path problems on trees

#### **Day 24: Final Review & Mock Interviews** (`week4/day24_final_review_mock_interviews.py`)

- **Integration**: Comprehensive 4-week review, mock interview simulation
- **Problems**: LRU Cache, Time Based Key-Value Store, Sliding Window Maximum, Word Break II
- **Features**: System design components, interview simulation, complete preparation summary

## 🚀 How to Use the Solution Files

### Running Individual Week Solutions

```bash
# Week 1 - Arrays, Strings, Hash Tables, Linked Lists, Stacks/Queues
cd practice/week1
python solution.py

# Week 2 - Trees, BST, Binary Search, Heaps
cd practice/week2
python solution.py

# Week 3 - Graphs, Union Find, Trie
cd practice/week3
python solution.py

# Week 4 - Dynamic Programming
cd practice/week4
python solution.py
```

### Running Daily Practice Files

```bash
# Week 3 Daily Practice
python week3/day13_graph_representation_bfs.py
python week3/day14_graph_dfs_backtracking.py
python week3/day15_graph_algorithms.py
python week3/day16_union_find_advanced.py
python week3/day17_tries_string_algorithms.py
python week3/day18_week3_review_complex.py

# Week 4 Daily Practice
python week4/day19_1d_dynamic_programming.py
python week4/day20_2d_dynamic_programming.py
python week4/day21_advanced_dp_patterns.py
python week4/day22_dp_optimization_techniques.py
python week4/day23_dp_trees_graphs.py
python week4/day24_final_review_mock_interviews.py
```

### Features of Solution Files

- **Multiple Variants**: Each core problem includes 2-4 related variants
- **Different Approaches**: Recursive vs iterative, different optimizations
- **Comprehensive Testing**: Edge cases, large inputs, boundary conditions
- **Production Ready**: Clean, commented, interview-ready code
- **Educational**: Detailed complexity analysis and algorithm explanations
- **Self-Contained**: No external dependencies, ready to run

### Testing Results Summary

- **Week 1**: 50+ test cases ✅ - Arrays, Strings, Hash Tables, Linked Lists, Stacks/Queues
- **Week 2**: 40+ test cases ✅ - Trees, BST, Binary Search, Heaps
- **Week 3**: 35+ test cases ✅ - Graphs, Union Find, Trie, Advanced Algorithms
- **Week 4**: 30+ test cases ✅ - Dynamic Programming (1D, 2D, Advanced)

**Total**: 155+ test cases covering all essential Meta interview topics

## Meta Interview Focus Areas

Based on Meta's technical interview patterns:

1. **Arrays & Strings** - High frequency
2. **Hash Tables** - Very common
3. **Trees & Graphs** - Core focus area
4. **Dynamic Programming** - Essential for senior roles
5. **Linked Lists** - Regular appearance
6. **Stacks & Queues** - Supporting data structures
7. **Sorting & Searching** - Fundamental algorithms
8. **Two Pointers & Sliding Window** - Common patterns
9. **System Design Concepts** - For senior roles

## Daily Schedule (4 hours)

- **Hour 1**: Concept review and theory
- **Hour 2**: Guided practice problems
- **Hour 3**: Independent problem solving
- **Hour 4**: Review, optimization, and mock interview

## Week 1: Foundations & Arrays/Strings

### Day 1: Arrays & Two Pointers

**Focus**: Array manipulation, two-pointer technique
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Array fundamentals and complexity analysis
- Two-pointer patterns: opposite ends, same direction
- Common array operations and optimizations

#### Problems (3 hours)

1. **Two Sum** (Easy) - 30 min
2. **Container With Most Water** (Medium) - 45 min
3. **3Sum** (Medium) - 60 min
4. **Remove Duplicates from Sorted Array** (Easy) - 30 min
5. **Trapping Rain Water** (Hard) - 45 min

### Day 2: Strings & Pattern Matching

**Focus**: String manipulation, substring problems
**Time Allocation**: 4 hours

#### Theory (1 hour)

- String operations and complexity
- Sliding window technique
- Pattern matching algorithms

#### Problems (3 hours)

1. **Valid Palindrome** (Easy) - 30 min
2. **Longest Substring Without Repeating Characters** (Medium) - 45 min
3. **Minimum Window Substring** (Hard) - 60 min
4. **Group Anagrams** (Medium) - 45 min
5. **Valid Parentheses** (Easy) - 30 min

### Day 3: Hash Tables & Sets

**Focus**: Hash-based data structures, frequency counting
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Hash table implementation and collision handling
- Time/space complexity analysis
- When to use hash tables vs other structures

#### Problems (3 hours)

1. **Two Sum** (revisit with hash table) - 20 min
2. **Subarray Sum Equals K** (Medium) - 45 min
3. **Top K Frequent Elements** (Medium) - 45 min
4. **First Missing Positive** (Hard) - 60 min
5. **Intersection of Two Arrays** (Easy) - 30 min

### Day 4: Linked Lists

**Focus**: Linked list operations, pointers manipulation
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Singly vs doubly linked lists
- Common patterns: fast/slow pointers, dummy nodes
- Memory management considerations

#### Problems (3 hours)

1. **Reverse Linked List** (Easy) - 30 min
2. **Merge Two Sorted Lists** (Easy) - 30 min
3. **Remove Nth Node From End** (Medium) - 45 min
4. **Linked List Cycle** (Easy) - 30 min
5. **Merge k Sorted Lists** (Hard) - 75 min

### Day 5: Stacks & Queues

**Focus**: Stack/queue operations, monotonic structures
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Stack and queue implementations
- Monotonic stack/queue patterns
- When to use each structure

#### Problems (3 hours)

1. **Valid Parentheses** (revisit) - 20 min
2. **Daily Temperatures** (Medium) - 45 min
3. **Largest Rectangle in Histogram** (Hard) - 60 min
4. **Implement Queue using Stacks** (Easy) - 30 min
5. **Next Greater Element** (Easy) - 45 min

### Day 6: Week 1 Review & Mock Interview

**Focus**: Consolidation and practice
**Time Allocation**: 4 hours

#### Activities

- Review all Week 1 concepts (1 hour)
- Solve 2-3 mixed problems (1.5 hours)
- Mock interview simulation (1 hour)
- Plan Week 2 (30 min)

## Week 2: Trees & Binary Search

### Day 7: Binary Trees Basics

**Focus**: Tree traversal, basic operations
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Binary tree properties and types
- Traversal methods: inorder, preorder, postorder
- Recursive vs iterative approaches

#### Problems (3 hours)

1. **Binary Tree Inorder Traversal** (Easy) - 30 min
2. **Maximum Depth of Binary Tree** (Easy) - 30 min
3. **Symmetric Tree** (Easy) - 30 min
4. **Path Sum** (Easy) - 30 min
5. **Binary Tree Level Order Traversal** (Medium) - 60 min
6. **Construct Binary Tree from Preorder and Inorder** (Medium) - 60 min

### Day 8: Binary Search Trees

**Focus**: BST properties, search operations
**Time Allocation**: 4 hours

#### Theory (1 hour)

- BST properties and invariants
- Insert, delete, search operations
- Balanced vs unbalanced trees

#### Problems (3 hours)

1. **Validate Binary Search Tree** (Medium) - 45 min
2. **Kth Smallest Element in BST** (Medium) - 45 min
3. **Lowest Common Ancestor of BST** (Easy) - 30 min
4. **Convert Sorted Array to BST** (Easy) - 30 min
5. **Delete Node in BST** (Medium) - 60 min

### Day 9: Binary Search Algorithm

**Focus**: Search algorithms, sorted arrays
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Binary search implementation
- Search space reduction
- Handling edge cases and duplicates

#### Problems (3 hours)

1. **Binary Search** (Easy) - 30 min
2. **Find First and Last Position** (Medium) - 45 min
3. **Search in Rotated Sorted Array** (Medium) - 45 min
4. **Find Peak Element** (Medium) - 45 min
5. **Search a 2D Matrix** (Medium) - 45 min

### Day 10: Tree Advanced Problems

**Focus**: Complex tree operations
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Tree modification and construction
- Parent-child relationships
- Tree serialization

#### Problems (3 hours)

1. **Serialize and Deserialize Binary Tree** (Hard) - 75 min
2. **Binary Tree Maximum Path Sum** (Hard) - 60 min
3. **Flatten Binary Tree to Linked List** (Medium) - 45 min
4. **Populating Next Right Pointers** (Medium) - 45 min

### Day 11: Heaps & Priority Queues

**Focus**: Heap operations, priority-based problems
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Heap properties: min-heap, max-heap
- Heap operations and complexity
- Priority queue applications

#### Problems (3 hours)

1. **Kth Largest Element in Array** (Medium) - 45 min
2. **Merge k Sorted Lists** (revisit with heap) - 45 min
3. **Top K Frequent Elements** (revisit with heap) - 30 min
4. **Find Median from Data Stream** (Hard) - 75 min
5. **Meeting Rooms II** (Medium) - 45 min

### Day 12: Week 2 Review & Assessment

**Focus**: Tree and search mastery check
**Time Allocation**: 4 hours

#### Activities

- Comprehensive review (1 hour)
- Timed problem solving (2 hours)
- Mock interview with tree problems (1 hour)

## Week 3: Graphs & Advanced Data Structures

### Day 13: Graph Representation & BFS (`week3/day13_graph_representation_bfs.py`)

**Focus**: Graph basics, breadth-first search
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Graph representations: adjacency list, matrix
- BFS algorithm and applications
- Level-by-level processing

#### Problems (3 hours)

1. **Number of Islands** (Medium) - 45 min
2. **Binary Tree Level Order Traversal** (revisit) - 30 min
3. **Word Ladder** (Hard) - 75 min
4. **Rotting Oranges** (Medium) - 45 min
5. **Clone Graph** (Medium) - 45 min

### Day 14: Graph DFS & Backtracking (`week3/day14_graph_dfs_backtracking.py`)

**Focus**: Depth-first search, path exploration
**Time Allocation**: 4 hours

#### Theory (1 hour)

- DFS algorithm and recursion
- Backtracking patterns
- Path tracking and state management

#### Problems (3 hours)

1. **Path Sum II** (Medium) - 45 min
2. **Generate Parentheses** (Medium) - 45 min
3. **Letter Combinations of Phone Number** (Medium) - 45 min
4. **Word Search** (Medium) - 60 min
5. **N-Queens** (Hard) - 75 min

### Day 15: Graph Algorithms (`week3/day15_graph_algorithms.py`)

**Focus**: Cycle detection, topological sort
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Cycle detection in directed/undirected graphs
- Topological sorting
- Graph coloring and bipartite graphs

#### Problems (3 hours)

1. **Course Schedule** (Medium) - 45 min
2. **Course Schedule II** (Medium) - 45 min
3. **Detect Cycle in Directed Graph** (Medium) - 45 min
4. **Is Graph Bipartite?** (Medium) - 45 min
5. **Alien Dictionary** (Hard) - 60 min

### Day 16: Union-Find & Advanced Graph (`week3/day16_union_find_advanced.py`)

**Focus**: Disjoint sets, connectivity problems
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Union-Find data structure
- Path compression and union by rank
- Applications in connectivity problems

#### Problems (3 hours)

1. **Number of Connected Components** (Medium) - 45 min
2. **Accounts Merge** (Medium) - 60 min
3. **Redundant Connection** (Medium) - 45 min
4. **Most Stones Removed** (Medium) - 60 min
5. **Satisfiability of Equality Equations** (Medium) - 45 min

### Day 17: Tries & String Algorithms (`week3/day17_tries_string_algorithms.py`)

**Focus**: Prefix trees, string matching
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Trie construction and operations
- Prefix matching applications
- Space-time tradeoffs

#### Problems (3 hours)

1. **Implement Trie** (Medium) - 45 min
2. **Word Search II** (Hard) - 75 min
3. **Add and Search Word** (Medium) - 45 min
4. **Longest Common Prefix** (Easy) - 30 min
5. **Replace Words** (Medium) - 45 min

### Day 18: Week 3 Review & Complex Problems (`week3/day18_week3_review_complex.py`)

**Focus**: Integration of multiple concepts
**Time Allocation**: 4 hours

#### Problems (4 hours)

1. **Critical Connections in a Network** (Hard) - 90 min
2. **Minimum Height Trees** (Medium) - 60 min
3. **Word Ladder II** (Hard) - 90 min
4. **Alien Dictionary Advanced** (Hard) - 60 min
5. **Graph Valid Tree** (Medium) - 45 min

## Week 4: Dynamic Programming Mastery

### Day 19: 1D Dynamic Programming (`week4/day19_1d_dynamic_programming.py`)

**Focus**: DP fundamentals, memoization vs tabulation
**Time Allocation**: 4 hours

#### Theory (1 hour)

- DP fundamentals and recurrence relations
- Memoization vs tabulation approaches
- Space optimization techniques

#### Problems (3 hours)

1. **Climbing Stairs** (Easy) - 30 min
2. **House Robber** (Medium) - 45 min
3. **House Robber II** (Medium) - 45 min
4. **Maximum Subarray** (Easy) - 30 min
5. **Coin Change** (Medium) - 60 min
6. **Longest Increasing Subsequence** (Medium) - 60 min

### Day 20: 2D Dynamic Programming (`week4/day20_2d_dynamic_programming.py`)

**Focus**: Grid-based DP, string matching
**Time Allocation**: 4 hours

#### Theory (1 hour)

- 2D DP patterns and state transitions
- Grid traversal problems
- String matching algorithms

#### Problems (3 hours)

1. **Unique Paths** (Medium) - 30 min
2. **Minimum Path Sum** (Medium) - 45 min
3. **Longest Common Subsequence** (Medium) - 60 min
4. **Edit Distance** (Hard) - 75 min
5. **Maximal Square** (Medium) - 60 min

### Day 21: Advanced DP Patterns (`week4/day21_advanced_dp_patterns.py`)

**Focus**: Complex DP patterns, optimization
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Interval DP and range problems
- State machine DP patterns
- Multi-dimensional state spaces

#### Problems (3 hours)

1. **Palindromic Substrings** (Medium) - 45 min
2. **Decode Ways** (Medium) - 60 min
3. **Best Time to Buy and Sell Stock with Cooldown** (Medium) - 75 min
4. **Burst Balloons** (Hard) - 90 min
5. **Regular Expression Matching** (Hard) - 90 min

### Day 22: DP Optimization Techniques (`week4/day22_dp_optimization_techniques.py`)

**Focus**: Space/time optimization, advanced techniques
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Space optimization strategies
- Matrix exponentiation for DP
- Monotonic deque optimization

#### Problems (3 hours)

1. **Perfect Squares** (Medium) - 45 min
2. **Maximal Rectangle** (Hard) - 75 min
3. **Sliding Window Maximum** (Hard) - 60 min
4. **Fibonacci with Matrix Exponentiation** (Medium) - 60 min
5. **Largest Divisible Subset** (Medium) - 75 min

### Day 23: DP on Trees and Graphs (`week4/day23_dp_trees_graphs.py`)

**Focus**: Tree DP, graph DP patterns
**Time Allocation**: 4 hours

#### Theory (1 hour)

- Tree DP with subtree states
- Path problems on trees
- Graph DP with memoization

#### Problems (3 hours)

1. **Binary Tree Maximum Path Sum** (Hard) - 60 min
2. **House Robber III** (Medium) - 45 min
3. **Diameter of Binary Tree** (Easy) - 30 min
4. **Longest Univalue Path** (Medium) - 60 min
5. **Minimum Cost to Make Array Non-decreasing** (Hard) - 90 min

### Day 24: Final Review & Mock Interviews (`week4/day24_final_review_mock_interviews.py`)

**Focus**: Comprehensive review and interview simulation
**Time Allocation**: 4 hours

#### Activities (4 hours)

1. **Complete 4-Week Review** - 60 min
2. **Mock Interview Simulation** - 90 min
3. **System Design Components** - 60 min
4. **Interview Strategy & Preparation** - 30 min

#### Featured Problems

1. **LRU Cache** (Medium) - System design component
2. **Time Based Key-Value Store** (Medium) - Advanced data structure
3. **Sliding Window Maximum** (Hard) - Algorithm optimization
4. **Word Break II** (Hard) - Complex DP with backtracking

## Problem Solutions Structure

Each problem includes:

- **Problem Statement**: Clear description with examples
- **Approach**: Step-by-step solution strategy
- **Code**: Clean, commented implementation
- **Complexity Analysis**: Time and space complexity
- **Edge Cases**: Important test cases to consider
- **Follow-up Questions**: Common interviewer extensions

## Templates & Resources

### Coding Templates

- Binary search template
- DFS/BFS templates
- Sliding window template
- Two pointers template
- Backtracking template
- Dynamic programming templates

### Complexity Analysis Guide

- Big O notation reference
- Common complexity patterns
- Space-time tradeoff examples

### Interview Preparation

- Behavioral questions for Meta
- Technical communication tips
- Whiteboard coding strategies
- Code optimization techniques

## Study Tips for Meta Success

1. **Focus on Clean Code**: Meta values readable, maintainable code
2. **Communicate Clearly**: Explain your thought process throughout
3. **Consider Edge Cases**: Always discuss potential issues
4. **Optimize Iteratively**: Start with working solution, then optimize
5. **Practice Whiteboard Coding**: Get comfortable coding without IDE
6. **Time Management**: Aim to solve problems within 20-30 minutes
7. **Ask Clarifying Questions**: Demonstrate requirements gathering skills

## Daily Checklist

- [ ] Complete theory review
- [ ] Solve all assigned problems
- [ ] Write clean, commented code
- [ ] Analyze time/space complexity
- [ ] Review and optimize solutions
- [ ] Practice explaining approach aloud

## Progress Tracking

Track your progress daily:

- Problems solved: **_/_**
- Time taken per problem
- Difficulty level comfort
- Areas needing more practice
- Mock interview scores

## Additional Resources

- Meta Engineering Blog
- Leetcode Meta tagged problems
- System Design Interview books
- Cracking the Coding Interview
- Meta career preparation resources

---

**Remember**: Consistency is key. Stick to the 4-hour daily commitment, focus on understanding over memorization, and practice explaining your solutions clearly. Good luck with your Meta interview!

## Next Steps

1. Clone this repository
2. Set up your development environment
3. **Run the comprehensive solution files to verify your setup**
4. Start with Week 1, Day 1
5. **Use the solution files as reference and practice**
6. **Follow the daily practice files for structured learning**
7. Track your progress daily
8. Adjust the schedule based on your learning pace
9. Schedule mock interviews for weeks 2-4

---

_This study plan is designed specifically for Meta technical interviews based on current interview patterns and feedback from successful candidates. The comprehensive solution files provide 155+ tested solutions with variants across all essential topics, and the daily practice files offer structured 4-hour learning sessions. Adjust the pace according to your background and comfort level._
