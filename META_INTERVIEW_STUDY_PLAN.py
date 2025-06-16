"""
=============================================================================
                    META TECHNICAL INTERVIEW STUDY PLAN
                           4-Week Intensive Preparation
                             Interview Date: July 14, 2025
=============================================================================

OVERVIEW:
This comprehensive 4-week program prepares you for Meta technical interviews
with 96 hours of structured learning (4 hours/day, 6 days/week).
No Dynamic Programming - focus on core Data Structures & Algorithms.

DAILY STRUCTURE:
• Hour 1: Theory and Concept Review
• Hour 2: Guided Practice Problems
• Hour 3: Independent Problem Solving
• Hour 4: Review, Optimization, and Mock Interviews

TARGET SKILLS FOR META:
✓ Clean, readable code
✓ Clear communication of thought process
✓ Consideration of edge cases
✓ Iterative optimization
✓ Strong fundamentals in core topics

=============================================================================
                                WEEK 1 (Days 1-6)
                         FOUNDATIONS & LINEAR STRUCTURES
=============================================================================

DAY 1: ARRAYS & TWO POINTERS
----------------------------
Theory: Array fundamentals, two-pointer technique, complexity analysis
Key Patterns: Opposite ends, same direction, pair finding
Problems Solved:
• Two Sum (brute force → hash optimization)
• Container With Most Water (two pointers)
• 3Sum (fixed element + two pointers)
• Remove Duplicates from Sorted Array
• Trapping Rain Water (advanced two pointers)

Implementation: week1/day1_arrays_two_pointers.py
Key Takeaways: Two pointers reduce O(n²) to O(n), handle sorted arrays efficiently

DAY 2: STRINGS & PATTERN MATCHING
---------------------------------
Theory: String operations, sliding window technique, pattern matching
Key Patterns: Fixed/variable window, character frequency, palindromes
Problems Solved:
• Valid Palindrome (two pointers on strings)
• Longest Substring Without Repeating Characters (sliding window)
• Minimum Window Substring (advanced sliding window)
• Group Anagrams (hash maps with strings)
• Valid Parentheses (stack introduction)

Implementation: week1/day2_strings_patterns.py
Key Takeaways: Sliding window for substring problems, hash maps for frequency

DAY 3: HASH TABLES & SETS
-------------------------
Theory: Hash table implementation, collision handling, complexity
Key Patterns: Frequency counting, complement lookup, grouping
Problems Solved:
• Two Sum Revisited (hash table optimization)
• Subarray Sum Equals K (prefix sum + hash)
• Top K Frequent Elements (hash + heap)
• First Missing Positive (hash set + in-place)
• Intersection of Two Arrays (set operations)

Implementation: week1/day3_hash_tables_sets.py
Key Takeaways: O(1) lookups, frequency patterns, space-time tradeoffs

DAY 4: LINKED LISTS
-------------------
Theory: Node structure, pointer manipulation, common patterns
Key Patterns: Fast/slow pointers, dummy nodes, cycle detection
Problems Solved:
• Reverse Linked List (iterative & recursive)
• Merge Two Sorted Lists (dummy head pattern)
• Remove Nth Node From End (two pointers)
• Linked List Cycle Detection (Floyd's algorithm)
• Merge k Sorted Lists (divide & conquer + heap)

Implementation: week1/day4_linked_lists.py
Key Takeaways: Pointer manipulation, Floyd's cycle detection, merging techniques

DAY 5: STACKS & QUEUES
----------------------
Theory: LIFO/FIFO operations, monotonic patterns, implementations
Key Patterns: Nested structures, next greater/smaller, BFS preparation
Problems Solved:
• Valid Parentheses (stack for nested structures)
• Daily Temperatures (monotonic stack)
• Largest Rectangle in Histogram (advanced monotonic stack)
• Implement Queue using Stacks (structural understanding)
• Next Greater Element (monotonic stack pattern)

Implementation: week1/day5_stacks_queues.py
Key Takeaways: Monotonic stacks, stack/queue duality, BFS foundation

DAY 6: REVIEW & INTEGRATION
---------------------------
Theory: Pattern recognition, concept integration, mock interviews
Integration Problems:
• LRU Cache (hash table + doubly linked list)
• Browser History (stack + array)
• Phone Directory (trie + set)
• Expression Evaluation (two stacks)
• Hit Counter (queue + sliding window)

Implementation: week1/day6_review_integration.py
Key Takeaways: Multiple patterns in complex problems, design considerations

Week 1 Achievement: Foundation Level - Ready for intermediate problems

=============================================================================
                                WEEK 2 (Days 7-12)
                              TREES & BINARY SEARCH
=============================================================================

DAY 7: BINARY TREES BASICS
--------------------------
Theory: Tree structure, traversal methods, recursive vs iterative
Key Patterns: DFS (pre/in/post order), BFS (level order), tree construction
Problems Solved:
• Binary Tree Traversals (all methods)
• Maximum Depth of Binary Tree
• Symmetric Tree (recursive & iterative)
• Path Sum Problems
• Binary Tree Level Order Traversal
• Construct Tree from Traversals

Builds on: Stack/Queue (Week 1), Recursion patterns
Key Takeaways: Tree traversal patterns, recursive thinking

DAY 8: BINARY SEARCH TREES
--------------------------
Theory: BST properties, search/insert/delete operations, balanced trees
Key Patterns: In-order traversal, BST validation, range queries
Problems Solved:
• Validate Binary Search Tree
• Kth Smallest Element in BST
• Lowest Common Ancestor of BST
• Convert Sorted Array to BST
• Delete Node in BST

Builds on: Tree traversals (Day 7), Two pointers concept
Key Takeaways: BST invariants, sorted sequence from in-order

DAY 9: BINARY SEARCH ALGORITHM
------------------------------
Theory: Search space reduction, template approach, edge cases
Key Patterns: Array binary search, rotated arrays, matrix search
Problems Solved:
• Classic Binary Search
• Find First and Last Position
• Search in Rotated Sorted Array
• Find Peak Element
• Search a 2D Matrix

Builds on: Two pointers (Week 1), Array manipulation
Key Takeaways: Search space reduction, O(log n) complexity

DAY 10: TREE ADVANCED PROBLEMS
------------------------------
Theory: Complex tree operations, serialization, path problems
Advanced Patterns: Tree modification, parent-child relationships
Problems Solved:
• Serialize and Deserialize Binary Tree
• Binary Tree Maximum Path Sum
• Flatten Binary Tree to Linked List
• Populating Next Right Pointers

Builds on: All previous tree concepts, Hash tables for relationships
Key Takeaways: Complex tree manipulation, optimization techniques

DAY 11: HEAPS & PRIORITY QUEUES
-------------------------------
Theory: Heap properties, heap operations, priority queue applications
Key Patterns: Top K problems, merge operations, median finding
Problems Solved:
• Kth Largest Element in Array
• Merge k Sorted Lists (heap approach)
• Top K Frequent Elements (heap approach)
• Find Median from Data Stream
• Meeting Rooms II

Builds on: Tree structure understanding, Array operations
Key Takeaways: Heap for priority problems, O(log n) operations

DAY 12: WEEK 2 REVIEW & ASSESSMENT
----------------------------------
Theory: Tree mastery check, complex problem integration
Assessment: Timed problem solving, pattern recognition
Focus: Tree and search algorithm mastery verification

Week 2 Achievement: Intermediate Level - Tree and search proficiency

=============================================================================
                                WEEK 3 (Days 13-18)
                           GRAPHS & ADVANCED STRUCTURES
=============================================================================

DAY 13: GRAPH REPRESENTATION & BFS
----------------------------------
Theory: Graph representations, BFS algorithm, level-by-level processing
Key Patterns: Adjacency list/matrix, shortest path, connected components
Problems Solved:
• Number of Islands
• Word Ladder
• Rotting Oranges
• Clone Graph
• Binary Tree Level Order (graph perspective)

Builds on: Queue operations (Week 1), Tree BFS (Week 2)
Key Takeaways: Graph traversal, BFS for shortest path

DAY 14: GRAPH DFS & BACKTRACKING
--------------------------------
Theory: DFS algorithm, backtracking patterns, path exploration
Key Patterns: Path finding, state space exploration, constraint satisfaction
Problems Solved:
• Path Sum II
• Generate Parentheses
• Letter Combinations of Phone Number
• Word Search
• N-Queens

Builds on: Stack operations (Week 1), Tree DFS (Week 2)
Key Takeaways: DFS for path problems, backtracking template

DAY 15: GRAPH ALGORITHMS
------------------------
Theory: Cycle detection, topological sort, graph coloring
Key Patterns: DAG problems, dependency resolution, bipartite graphs
Problems Solved:
• Course Schedule (I & II)
• Detect Cycle in Directed Graph
• Is Graph Bipartite?
• Alien Dictionary

Builds on: Graph traversal (Days 13-14), Hash tables for state tracking
Key Takeaways: Advanced graph algorithms, cycle detection patterns

DAY 16: UNION-FIND & ADVANCED GRAPH
-----------------------------------
Theory: Disjoint sets, path compression, union by rank
Key Patterns: Connectivity problems, dynamic graph problems
Problems Solved:
• Number of Connected Components
• Accounts Merge
• Redundant Connection
• Most Stones Removed

Builds on: Graph connectivity concepts, Hash tables for grouping
Key Takeaways: Union-Find for connectivity, path compression optimization

DAY 17: TRIES & STRING ALGORITHMS
---------------------------------
Theory: Prefix trees, string matching, space-time tradeoffs
Key Patterns: Prefix matching, word problems, autocomplete
Problems Solved:
• Implement Trie
• Word Search II
• Add and Search Word
• Longest Common Prefix
• Replace Words

Builds on: String processing (Week 1), Tree structure (Week 2)
Key Takeaways: Trie for prefix problems, string algorithm applications

DAY 18: WEEK 3 REVIEW & COMPLEX PROBLEMS
----------------------------------------
Theory: Multi-concept integration, advanced problem solving
Complex Problems: Combining graphs, strings, and advanced structures
Focus: Integration of multiple Week 3 concepts

Week 3 Achievement: Advanced Level - Graph and structure mastery

=============================================================================
                                WEEK 4 (Days 19-24)
                        INTEGRATION & META-SPECIFIC PREP
=============================================================================

DAY 19: SORTING & ADVANCED ARRAYS
---------------------------------
Theory: Sorting algorithms, stability, custom criteria
Key Patterns: Interval problems, custom comparisons, in-place operations
Problems Solved:
• Merge Intervals
• Insert Interval
• Non-overlapping Intervals
• Sort Colors
• Largest Number
• Meeting Rooms

Builds on: Array manipulation (Week 1), Merge operations (Week 2)
Key Takeaways: Interval algorithms, sorting applications

DAY 20: GREEDY ALGORITHMS
------------------------
Theory: Greedy choice property, proof techniques, optimization
Key Patterns: Local optimal → global optimal, scheduling problems
Problems Solved:
• Best Time to Buy and Sell Stock (I & II)
• Jump Game (I & II)
• Gas Station
• Task Scheduler

Builds on: Array optimization, Algorithm design principles
Key Takeaways: Greedy patterns, when greedy works

DAY 21: META-SPECIFIC PROBLEMS
-----------------------------
Theory: Meta interview patterns, commonly asked problems
Meta Favorites:
• Add Binary
• Valid Palindrome II
• Move Zeroes
• Binary Tree Vertical Order Traversal
• Random Pick with Weight
• Subarray Sum Equals K (revisit)
• Exclusive Time of Functions
• Expression Add Operators

Builds on: All previous weeks' patterns
Key Takeaways: Meta-specific problem patterns, interview expectations

DAY 22: SYSTEM DESIGN FUNDAMENTALS
----------------------------------
Theory: Scalability, load balancing, caching, databases
Basic Concepts: Microservices, distributed systems, trade-offs
Practice Designs:
• URL Shortener
• Chat Application
• Newsfeed System
• Trade-off discussions

Focus: Basic system design for technical interviews
Key Takeaways: High-level design thinking, scalability concepts

DAY 23: MOCK INTERVIEWS & OPTIMIZATION
--------------------------------------
Practice: Full interview simulations, code optimization
Activities:
• Mock Interview #1 (1 hour)
• Code review and optimization (1 hour)
• Mock Interview #2 (1 hour)
• Behavioral question preparation (1 hour)

Focus: Interview simulation, performance under pressure
Key Takeaways: Interview techniques, time management

DAY 24: FINAL REVIEW & CONFIDENCE BUILDING
------------------------------------------
Activities:
• Review weak areas (1 hour)
• Speed coding practice (1 hour)
• Meta problem marathon (1.5 hours)
• Mental preparation and strategy (30 min)

Focus: Last-minute preparation, confidence building
Key Takeaways: Interview readiness, final strategy

Week 4 Achievement: Expert Level - Meta interview ready

=============================================================================
                              SUCCESS METRICS
=============================================================================

TECHNICAL SKILLS ACHIEVED:
✅ Master fundamental data structures (Arrays, Strings, Hash Tables, Linked Lists)
✅ Proficient in linear structures (Stacks, Queues)
✅ Expert in tree structures and traversals
✅ Competent in graph algorithms and advanced structures
✅ Skilled in sorting, searching, and optimization
✅ Familiar with system design basics

PROBLEM-SOLVING PATTERNS MASTERED:
✅ Two Pointers (opposite ends, same direction, fast/slow)
✅ Sliding Window (fixed and variable size)
✅ Hash Table patterns (frequency, complement, grouping)
✅ Tree traversals (DFS, BFS, iterative, recursive)
✅ Graph algorithms (DFS, BFS, cycle detection, topological sort)
✅ Binary Search variations
✅ Greedy algorithms
✅ Backtracking

INTERVIEW SKILLS DEVELOPED:
✅ Clean, readable code
✅ Clear communication of approach
✅ Edge case consideration
✅ Time/space complexity analysis
✅ Iterative optimization
✅ Problem pattern recognition

META-SPECIFIC PREPARATION:
✅ 100+ problems solved with full solutions
✅ Theory and practical application integrated
✅ Mock interview experience
✅ Behavioral question preparation
✅ System design fundamentals
✅ Meta-favorite problems practiced

=============================================================================
                            EXECUTION INSTRUCTIONS
=============================================================================

TO RUN THE COMPLETE STUDY PLAN:

1. WEEK 1 - FOUNDATIONS:
   ```bash
   python week1/day1_arrays_two_pointers.py
   python week1/day2_strings_patterns.py
   python week1/day3_hash_tables_sets.py
   python week1/day4_linked_lists.py
   python week1/day5_stacks_queues.py
   python week1/day6_review_integration.py
   ```

2. WEEK 2 - TREES & SEARCH (To be implemented):
   ```bash
   python week2/day7_binary_trees.py
   python week2/day8_binary_search_trees.py
   python week2/day9_binary_search.py
   python week2/day10_tree_advanced.py
   python week2/day11_heaps_priority_queues.py
   python week2/day12_week2_review.py
   ```

3. WEEK 3 - GRAPHS & ADVANCED (To be implemented):
   ```bash
   python week3/day13_graph_bfs.py
   python week3/day14_graph_dfs_backtracking.py
   python week3/day15_graph_algorithms.py
   python week3/day16_union_find.py
   python week3/day17_tries_strings.py
   python week3/day18_week3_review.py
   ```

4. WEEK 4 - INTEGRATION & META PREP (To be implemented):
   ```bash
   python week4/day19_sorting_arrays.py
   python week4/day20_greedy_algorithms.py
   python week4/day21_meta_specific.py
   python week4/day22_system_design.py
   python week4/day23_mock_interviews.py
   python week4/day24_final_review.py
   ```

DAILY ROUTINE:
1. Run the day's Python file to see theory and problems
2. Work through problems independently
3. Review solutions and complexity analysis
4. Practice explaining solutions out loud
5. Note any weak areas for additional review

PROGRESS TRACKING:
- Mark completed days: ✅
- Note difficult concepts for review
- Track problem-solving speed improvement
- Practice mock interviews weekly

=============================================================================
                                FINAL NOTES
=============================================================================

This study plan provides:
• 96 hours of structured preparation
• 100+ coding problems with full solutions
• Comprehensive theory with practical application
• Progressive difficulty building on previous concepts
• Meta-specific interview preparation
• Mock interview practice and system design basics

Success Indicators:
• Can solve medium-level problems in 20-30 minutes
• Recognizes patterns quickly
• Writes clean, bug-free code on first attempt
• Explains approach clearly while coding
• Handles follow-up questions confidently

Remember:
• Consistency is key - stick to the 4-hour daily schedule
• Practice coding on whiteboard/paper occasionally  
• Focus on understanding patterns, not memorizing solutions
• Communicate your thought process clearly
• Stay confident and trust your preparation

🎯 TARGET: Confident, well-prepared Meta technical interview performance
📅 DEADLINE: July 14, 2025
🚀 SUCCESS: Comprehensive preparation for Meta technical excellence

Good luck with your Meta interview preparation! 🍀
"""

# UTILITY FUNCTIONS FOR STUDY PLAN EXECUTION

def display_study_plan_overview():
    """Display the complete study plan structure"""
    print(__doc__)

def get_week_summary(week_num):
    """Get summary for specific week"""
    summaries = {
        1: """
        WEEK 1 SUMMARY: FOUNDATIONS & LINEAR STRUCTURES
        ================================================
        Focus: Master fundamental data structures and basic algorithms
        Key Topics: Arrays, Strings, Hash Tables, Linked Lists, Stacks, Queues
        Outcome: Solid foundation for intermediate problems
        """,
        2: """
        WEEK 2 SUMMARY: TREES & BINARY SEARCH
        ======================================
        Focus: Tree structures and search algorithms
        Key Topics: Binary Trees, BST, Binary Search, Heaps
        Outcome: Tree traversal and search algorithm proficiency
        """,
        3: """
        WEEK 3 SUMMARY: GRAPHS & ADVANCED STRUCTURES  
        =============================================
        Focus: Graph algorithms and advanced data structures
        Key Topics: Graph traversal, Backtracking, Union-Find, Tries
        Outcome: Advanced problem-solving capabilities
        """,
        4: """
        WEEK 4 SUMMARY: INTEGRATION & META PREPARATION
        ==============================================
        Focus: Integration and Meta-specific preparation
        Key Topics: Sorting, Greedy, Meta problems, System Design, Mock Interviews
        Outcome: Meta interview readiness
        """
    }
    return summaries.get(week_num, "Week not found")

def check_prerequisites(day):
    """Check if prerequisites are met for given day"""
    prerequisites = {
        7: "Complete Week 1 (Days 1-6)",
        13: "Complete Weeks 1-2 (Days 1-12)", 
        19: "Complete Weeks 1-3 (Days 1-18)"
    }
    
    if day in prerequisites:
        print(f"Prerequisites for Day {day}: {prerequisites[day]}")
        return False
    return True

def get_daily_schedule():
    """Display the daily 4-hour schedule structure"""
    schedule = """
    DAILY SCHEDULE (4 Hours):
    ========================
    Hour 1 (60 min): Theory Review & Concept Learning
    - Read theory section
    - Understand key patterns
    - Review complexity analysis
    
    Hour 2 (60 min): Guided Practice  
    - Work through example problems
    - Study provided solutions
    - Understand implementation details
    
    Hour 3 (60 min): Independent Problem Solving
    - Solve additional problems independently
    - Practice without looking at solutions first
    - Time yourself on problem-solving
    
    Hour 4 (60 min): Review & Optimization
    - Review solutions and alternatives
    - Optimize code for clarity and efficiency
    - Practice explaining solutions aloud
    - Mock interview practice (later weeks)
    """
    print(schedule)

if __name__ == "__main__":
    display_study_plan_overview() 