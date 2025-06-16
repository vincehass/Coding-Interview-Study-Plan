#!/usr/bin/env python3
"""
=============================================================================
                   META INTERVIEW STUDY PLAN RUNNER
                        Interactive Study Guide
=============================================================================

This script provides an interactive interface to navigate and run the 
complete 4-week Meta interview preparation study plan.

Usage:
  python run_study_plan.py
  
Features:
- View complete study plan overview
- Run individual days interactively
- Track progress
- Get daily schedules and tips
- Access pattern recognition guides

=============================================================================
"""

import os
import sys
from datetime import datetime, timedelta

def display_banner():
    """Display the main banner"""
    print("=" * 80)
    print("                    META TECHNICAL INTERVIEW PREPARATION")
    print("                         4-Week Intensive Study Plan")
    print("                           Interview: July 14, 2025")
    print("=" * 80)

def calculate_days_remaining():
    """Calculate days remaining until interview"""
    interview_date = datetime(2025, 7, 14)
    today = datetime.now()
    days_remaining = (interview_date - today).days
    return days_remaining

def display_progress_tracker():
    """Display current progress and timeline"""
    days_remaining = calculate_days_remaining()
    
    print(f"\n📅 TIMELINE TRACKING:")
    print(f"   Interview Date: July 14, 2025")
    print(f"   Days Remaining: {days_remaining}")
    print(f"   Study Plan: 24 days (4 weeks × 6 days)")
    
    if days_remaining >= 24:
        print(f"   Status: ✅ Perfect timing - {days_remaining - 24} buffer days")
    elif days_remaining >= 20:
        print(f"   Status: ⚠️  Tight schedule - focus on core topics")
    else:
        print(f"   Status: 🚨 Intensive mode - prioritize Week 1 & Meta specifics")

def show_daily_schedule():
    """Show the standard daily 4-hour schedule"""
    print("\n⏰ DAILY SCHEDULE (4 Hours):")
    print("   Hour 1: Theory Review & Concept Learning")
    print("   Hour 2: Guided Practice Problems") 
    print("   Hour 3: Independent Problem Solving")
    print("   Hour 4: Review, Optimization & Mock Interviews")

def run_week1_day(day_num):
    """Run a specific day from Week 1"""
    day_files = {
        1: "week1/day1_arrays_two_pointers.py",
        2: "week1/day2_strings_patterns.py", 
        3: "week1/day3_hash_tables_sets.py",
        4: "week1/day4_linked_lists.py",
        5: "week1/day5_stacks_queues.py",
        6: "week1/day6_review_integration.py"
    }
    
    if day_num in day_files:
        file_path = day_files[day_num]
        if os.path.exists(file_path):
            print(f"\n🚀 RUNNING DAY {day_num} STUDY MATERIAL...")
            print("=" * 60)
            os.system(f"python {file_path}")
        else:
            print(f"❌ File not found: {file_path}")
    else:
        print(f"❌ Invalid day number: {day_num}")

def show_week_overview(week_num):
    """Show overview for a specific week"""
    week_info = {
        1: {
            "title": "FOUNDATIONS & LINEAR STRUCTURES",
            "topics": ["Arrays & Two Pointers", "Strings & Patterns", "Hash Tables & Sets", 
                      "Linked Lists", "Stacks & Queues", "Review & Integration"],
            "goal": "Master fundamental data structures and basic algorithms",
            "outcome": "Solid foundation for intermediate problems"
        },
        2: {
            "title": "TREES & BINARY SEARCH", 
            "topics": ["Binary Trees Basics", "Binary Search Trees", "Binary Search Algorithm",
                      "Tree Advanced", "Heaps & Priority Queues", "Week 2 Review"],
            "goal": "Master tree structures and search algorithms",
            "outcome": "Tree traversal and search algorithm proficiency"
        },
        3: {
            "title": "GRAPHS & ADVANCED STRUCTURES",
            "topics": ["Graph BFS", "Graph DFS & Backtracking", "Graph Algorithms",
                      "Union-Find", "Tries & String Algorithms", "Week 3 Review"],
            "goal": "Master graph algorithms and advanced data structures", 
            "outcome": "Advanced problem-solving capabilities"
        },
        4: {
            "title": "INTEGRATION & META PREPARATION",
            "topics": ["Sorting & Arrays", "Greedy Algorithms", "Meta-Specific Problems",
                      "System Design", "Mock Interviews", "Final Review"],
            "goal": "Integration and Meta-specific preparation",
            "outcome": "Meta interview readiness"
        }
    }
    
    if week_num in week_info:
        info = week_info[week_num]
        print(f"\n📚 WEEK {week_num}: {info['title']}")
        print("=" * 60)
        print(f"Goal: {info['goal']}")
        print(f"Outcome: {info['outcome']}")
        print("\nDaily Topics:")
        for i, topic in enumerate(info['topics'], 1):
            status = "✅" if week_num == 1 else "📋"
            print(f"   Day {i + (week_num-1)*6}: {topic} {status}")
    else:
        print(f"❌ Invalid week number: {week_num}")

def show_pattern_guide():
    """Show pattern recognition guide"""
    print("\n🎯 PATTERN RECOGNITION GUIDE:")
    print("=" * 50)
    
    patterns = {
        "Two Pointers": ["Pair sum (sorted array)", "Palindromes", "Remove duplicates"],
        "Sliding Window": ["Longest/shortest substring", "Fixed size subarray", "Anagrams"],
        "Hash Table": ["Frequency counting", "Fast lookups", "Grouping"],
        "Stack": ["Nested structures", "Monotonic patterns", "Expression evaluation"],
        "Queue": ["Level-by-level", "FIFO processing", "BFS"],
        "Linked List": ["Cycle detection", "Merging", "Two pointers"]
    }
    
    for pattern, uses in patterns.items():
        print(f"\n{pattern}:")
        for use in uses:
            print(f"  • {use}")

def show_meta_tips():
    """Show Meta-specific interview tips"""
    print("\n🎯 META INTERVIEW TIPS:")
    print("=" * 40)
    print("✓ Write clean, readable code")
    print("✓ Explain your approach before coding")
    print("✓ Consider edge cases out loud")
    print("✓ Start with brute force, then optimize")
    print("✓ Test your code with examples")
    print("✓ Discuss time/space complexity")
    print("✓ Ask clarifying questions")
    print("✓ Stay calm and think out loud")

def main_menu():
    """Display main menu and handle user input"""
    while True:
        display_banner()
        display_progress_tracker()
        
        print("\n📋 STUDY PLAN MENU:")
        print("   1. View Complete Study Plan Overview")
        print("   2. Run Week 1 Day (Available)")
        print("   3. View Week Overview") 
        print("   4. Show Daily Schedule")
        print("   5. Pattern Recognition Guide")
        print("   6. Meta Interview Tips")
        print("   7. Exit")
        
        choice = input("\n👉 Select option (1-7): ").strip()
        
        if choice == '1':
            print("\n🚀 LOADING COMPLETE STUDY PLAN...")
            os.system("python META_INTERVIEW_STUDY_PLAN.py")
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            print("\n📚 WEEK 1 DAYS AVAILABLE:")
            print("   1. Arrays & Two Pointers")
            print("   2. Strings & Pattern Matching") 
            print("   3. Hash Tables & Sets")
            print("   4. Linked Lists")
            print("   5. Stacks & Queues")
            print("   6. Review & Integration")
            
            try:
                day = int(input("\n👉 Select day (1-6): "))
                run_week1_day(day)
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            try:
                week = int(input("\n👉 Enter week number (1-4): "))
                show_week_overview(week)
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            show_daily_schedule()
            input("\nPress Enter to continue...")
            
        elif choice == '5':
            show_pattern_guide()
            input("\nPress Enter to continue...")
            
        elif choice == '6':
            show_meta_tips()
            input("\nPress Enter to continue...")
            
        elif choice == '7':
            print("\n🎯 Good luck with your Meta interview preparation!")
            print("💪 Stay consistent, practice daily, and trust the process!")
            break
            
        else:
            print("❌ Invalid option. Please select 1-7.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Study session interrupted. See you next time!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your setup and try again.") 