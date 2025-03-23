import unittest

class Solution(object):
    def rotate(self, matrix):

        n = len(matrix)

        for i in range(n):
            for j in range(i + 1, n): 
                temp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp

        for row in matrix:
            row.reverse()

if __name__ == '__main__':
    example_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Original Matrix:")
    for row in example_matrix:
        print(row)

    Solution().rotate(example_matrix)

    print("\nRotated Matrix:")
    for row in example_matrix:
        print(row)

    unittest.main(argv=['first-arg-is-ignored'], exit=False)

class TestRotateMatrix(unittest.TestCase):
    def test_rotate(self):
        sol = Solution()

        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        expected = [
            [7, 4, 1],
            [8, 5, 2],
            [9, 6, 3]
        ]

        sol.rotate(matrix)
        self.assertEqual(matrix, expected)

    def test_rotate_4x4(self):
        sol = Solution()

        matrix = [
            [ 5,  1,  9, 11],
            [ 2,  4,  8, 10],
            [13,  3,  6,  7],
            [15, 14, 12, 16]
        ]

        expected = [
            [15, 13,  2,  5],
            [14,  3,  4,  1],
            [12,  6,  8,  9],
            [16,  7, 10, 11]
        ]

        sol.rotate(matrix)
        self.assertEqual(matrix, expected)