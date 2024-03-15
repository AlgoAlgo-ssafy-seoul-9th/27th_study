# 14428_수열과 쿼리16_Sequence and Query16
import sys
input = sys.stdin.readline


def build(node, l, r):
    if l == r:
        seg_tree[node] = (l+1, an_lst[l])
        return

    mid = (l+r)//2
    build(node*2, l, mid)
    build(node*2+1, mid+1, r)
    if seg_tree[node*2][1] == seg_tree[node*2+1][1]:
        if seg_tree[node*2][0] < seg_tree[node*2+1][0]:
            seg_tree[node] = seg_tree[node*2]
        else:
            seg_tree[node] = seg_tree[node*2+1]
    else:
        seg_tree[node] = min(seg_tree[node*2], seg_tree[node*2+1], key=lambda x: x[1])

    return


def query(node, l_idx, r_idx, l_limit, r_limit):
    if r_idx < l_limit or l_idx > r_limit:
        return 0, float('inf')

    if l_idx >= l_limit and r_idx <= r_limit:
        return seg_tree[node]

    mid = (l_idx+r_idx)//2
    left = query(node*2, l_idx, mid, l_limit, r_limit)
    right = query(node*2+1, mid+1, r_idx, l_limit, r_limit)

    return min(left, right, key=lambda x: x[1])


def update(node, l_idx, r_idx, pos, value):
    if l_idx == r_idx:
        seg_tree[node] = (l_idx+1, value)
        return

    mid = (l_idx+r_idx)//2
    if l_idx <= pos <= mid:
        update(node*2, l_idx, mid, pos, value)
    else:
        update(node*2+1, mid+1, r_idx, pos, value)

    if seg_tree[node * 2][1] == seg_tree[node * 2 + 1][1]:
        if seg_tree[node * 2][0] < seg_tree[node * 2 + 1][0]:
            seg_tree[node] = seg_tree[node*2]
        else:
            seg_tree[node] = seg_tree[node*2+1]
    else:
        seg_tree[node] = min(seg_tree[node * 2], seg_tree[node * 2 + 1], key=lambda x: x[1])
    return


N = int(input())
an_lst = list(map(int, input().split()))
M = int(input())

seg_tree = [(0, float('inf'))]*(4*N)
build(1, 0, N-1)

for _ in range(M):
    o, s, e = map(int, input().split())
    if o == 1:
        update(1, 0, N-1, s-1, e)
    else:
        tmp = query(1, 0, N-1, s-1, e-1)
        print(tmp[0])

