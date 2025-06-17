def two_sum(nums, target):
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i
    return []




def two_sum_2(nums, target):
    #we can use two pointers to solve this problem
    n = len(nums)-1
    left = 0
    right = n
    while left < right:
        if nums[left] + nums[right] == target:
            return [left, right]
        elif nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1
    


if __name__ == "__main__":
    test_nums = [2,7,11,15]
    test_target = 9
    print(two_sum(test_nums, test_target))

    test_nums_2 = [3,2,4]
    test_target_2 = 6
    print(two_sum(test_nums_2, test_target_2))

    test_nums_3 = [3,3]
    test_target_3 = 6
    print(two_sum(test_nums_3, test_target_3))

    test_nums_4 = [3,2,3]
    test_target_4 = 6
    print(two_sum(test_nums_4, test_target_4))

    test_nums_5 = [2,5,5,11]
    test_target_5 = 10
    print(two_sum(test_nums_5, test_target_5))
    test_passed = two_sum(test_nums, test_target) == [0, 1]
    print(f"Test 1 {'passed' if test_passed else 'failed'}")
    
    test_passed = two_sum(test_nums_2, test_target_2) == [1, 2]
    print(f"Test 2 {'passed' if test_passed else 'failed'}")
    
    test_passed = two_sum(test_nums_3, test_target_3) == [0, 1]
    print(f"Test 3 {'passed' if test_passed else 'failed'}")
    
    test_passed = two_sum(test_nums_4, test_target_4) == [0, 2]
    print(f"Test 4 {'passed' if test_passed else 'failed'}")
    
    test_passed = two_sum(test_nums_5, test_target_5) == [1, 2]
    print(f"Test 5 {'passed' if test_passed else 'failed'}")


def three_sum(nums):
    #please give the expected outcome and the goal of the problem
    #the goal of the problem is to find all the triplets that sum to 0
    #the expected outcome is a list of triplets that sum to 0
    #the triplets should be sorted in ascending order
    #the triplets should be unique
    #the triplets should be in the same order as the input list
    #the triplets should be in the same order as the input list
    nums.sort()
    result = []
    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i+1, len(nums)-1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result

if __name__ == "__main__":
    test_nums = [-1,0,1,2,-1,-4]
    print(three_sum(test_nums))