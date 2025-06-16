

def two_sum(nums, target):
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in hash_map:
            print(hash_map[complement])
            return [hash_map[complement], i]
        hash_map[num] = i

    
test_nums = [2,7,11,15]
test_target = 9

print(two_sum(test_nums, test_target))