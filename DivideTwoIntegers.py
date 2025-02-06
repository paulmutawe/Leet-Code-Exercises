class Solution(object):
    def divide(slef, dividend, divisor):

        INT_MAX = 2**31 -1
        INT_MIN = -2**31

        if dividend == INT_MIN and divisor == -1:
            return INT_MAX
        
        negative = (dividend < 0) ^ (divisor < 0)

        dividend, divisor = abs(dividend), abs(divisor)

        quotient = 0

        while dividend >= divisor:
            temp_divisor, num_shifts = divisor, 1

            while dividend >= (temp_divisor << 1):
                temp_divisor <<= 1
                num_shifts <<= 1

            dividend -= temp_divisor
            quotient += num_shifts

        if negative:
            quotient = -quotient 

        return max(INT_MIN, min(INT_MAX, quotient))
    
solution = Solution()
print(solution.divide(10, 3))
print(solution.divide(7, -3))

