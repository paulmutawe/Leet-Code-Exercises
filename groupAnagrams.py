#python3 -m venv .venv
#source .venv/bin/activate

class Solution(object):
    def groupAnagrams(self, strs):

        anagram_dict = {}

        for word in strs:

            sorted_word = ''.join(sorted(word))

            if sorted_word in anagram_dict:

                anagram_dict[sorted_word].append(word)

            else:

                anagram_dict[sorted_word] = [word]

        return list(anagram_dict.values())
    
def test_group_anagrams():

    solution = Solution()
    input_strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

    result = solution.groupAnagrams(input_strs)

    expected_groups = [
            {"eat", "tea", "ate"},
            {"tan", "nat"},
            {"bat"}
        ]

    result_as_sets = [set(group) for group in result]

    for group_set in expected_groups:

        assert group_set in result_as_sets, f"Expected group {group_set}  is missing in the result."

    print("All test cases passed!")

class Main:

    def run_solution(self):
        solution = Solution()
        sample_input = ["eat", "tea", "tan", "ate", "nat", "bat"]
        output = solution.groupAnagrams(sample_input)
        print("Grouped Anagrams:", output)

if __name__ == "__main__":
    test_group_anagrams()
    main_instance = Main()
    main_instance.run_solution()