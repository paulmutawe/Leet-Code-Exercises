from typing import List
import unittest


solutions: List[List[str]] = []
n_global: int = 0

def solveNQueens(n: int) -> List[List[str]]:
    
    global solutions, n_global
    solutions = []
    n_global = n
    
    cols = set()
    diag1 = set()
    diag2 = set()
    positions: List[int] =[]
    
    def backtrack(row: int) -> None:
        
        if row == n_global:
            
            solutions.append(build_board(positions))
            return
        
        for col in range(n_global):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            positions.append(col)
            
            backtrack(row + 1)
            
            positions.pop()
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
            
    backtrack(0)
    return solutions

def build_board(positions: List[int]) -> List[str]:
    
    board: List[str] = []
    for row_idx, col in enumerate(positions):
        
        row_str = ''.join(
            'Q' if j == col else '.'
            for j in range(n_global)
        )
        board.append(row_str)
    return board

def print_solutions(sols: List[List[str]]) -> None:
    
    for i, sol in enumerate(sols, start = 1):
        print(f"Solution {i}:")
        for row in sol:
            print(row)
        print()
        
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        
        return solveNQueens(n)
    
class TestNQueens(unittest.TestCase):
    
    def setUp(self):
        self.solver = Solution()
        
    def test_n1(self):
        self.assertEqual(self.solver.solveNQueens(1), [["Q"]])
        
    def test_n2(self):
        self.assertEqual(self.solver.solveNQueens(2), [])
        
    def test_n4(self):
        expected = [
            [".Q..", 
             "...Q", 
             "Q...", 
             "..Q."],
            ["..Q.", 
             "Q...", 
             "...Q", 
             ".Q.."]
        ]
        result = self.solver.solveNQueens(4)
        
        self.assertEqual(sorted(result), sorted(expected))
        
if __name__ == "__main__":
    
    unittest.main()