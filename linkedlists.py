class ListNode(object):
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

class Solution(object):
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(0)
        dummy.next = head
        
        slow = dummy
        fast = dummy
        
        for _ in range(n + 1):
            fast = fast.next

        while fast:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next

        return dummy.next
    
def list_to_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head
    
def print_linked_list(head):
    current = head
    result = []
    while current:
        result.append(str(current.val))
        current = current.next
    print("->".join(result))

test_values = [1, 2, 3, 4, 5]
head = list_to_linked_list(test_values)

solution = Solution()
modified_head = solution.removeNthFromEnd(head, 2)

print("Original list:", "->".join(map(str, test_values)))
print("Modified list after removing 2nd node from the end:")
print_linked_list(modified_head)

        