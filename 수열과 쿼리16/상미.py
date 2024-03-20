import sys
input = sys.stdin.readline

# def Query1(lst, a, b):
#     lst[a-1] = b
#     return lst

# def Query2(lst, a, b):
#     minV = 1000000000
#     for i in range(a-1, b):
#         if minV > lst[i]:
#             minV = lst[i]
#             ans = i+1
#     return ans

N = int(input())
lst = list(map(int, input().split()))
tree = [0 for i in range(N*10)]

def segmentTree(tree, node, left, right):
    if left == right:
        tree[node] = lst[left]
        return tree[node]
    mid = (left + right) // 2
    leftV = segmentTree(tree, node*2, left, mid)
    rightV = segmentTree(tree, node*2 + 1, mid+1, right)
    tree[node] = leftV + rightV
    return tree[node]

M = int(input())        # 쿼리의 개수
for _ in range(M):
    x, y, z = map(int, input().split())
    # if x == 1:
    #     Query1(lst, y, z)
    # elif x == 2:
    #     print(Query2(lst, y, z))
    segmentTree(tree, 1, 0, N-1)