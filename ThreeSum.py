class Solution(object):
    def threeSum(self, nums):

        nums.sort()
        n = len(nums)
        result = []

        for i in range (n):

            if i > 0 and nums[i] == nums[i - 1]:
                continue

            left, right = i + 1, n - 1

            while left < right:

                current_sum = nums[i] + nums[left] + nums[right]

                if current_sum == 0:

                    result.append([nums[i], nums[left], nums[right]])

                    while left < right and nums[left] == nums[left + 1]:
                        left += 1

                    while left < right and nums[right] == nums [right -1]:
                        right -= 1

                    left += 1
                    right -= 1

                elif current_sum < 0:

                    left += 1

                else:

                    right -= 1

        return result

nums = [-1, 0, 1, 2, -1, 4]
solution = Solution()
print(solution.threeSum(nums))

