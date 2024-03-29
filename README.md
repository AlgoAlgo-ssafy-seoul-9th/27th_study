# 27th_study

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## [수열과 쿼리16](https://www.acmicpc.net/problem/14428)

### [민웅](</수열과 쿼리16/민웅.py>)

```py
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



```

### [상미](</수열과 쿼리16/상미.py>)

```py
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

```

### [성구](</수열과 쿼리16/성구.py>)

```py
# 14428 수열과 쿼리 16
import sys
input = sys.stdin.readline

# 반복문 구현

class SegmentTree:

    def __init__(self, arr:list) -> None:
        self.N = len(arr)
        self.tree = [[0,0] for _ in range(self.N << 1)]
        self.build(arr)
        return

    def build(self, arr:list) -> None:
        for i in range(self.N):
            self.tree[i+self.N] = (arr[i], i+1)

        for i in range(self.N-1, 0, -1):
            self.tree[i] = min(self.tree[i<<1], self.tree[(i<<1)+1], key=lambda x:(x[0],x[1]))

        return

    def update(self, idx:int, val:int) -> None:
        index = idx-1+self.N
        self.tree[index] = (val, idx)

        index >>= 1

        while index:
            self.tree[index] = min(self.tree[index<<1], self.tree[(index<<1)+1], key=lambda x:(x[0],x[1]))
            index >>= 1
        return


    def find(self, start:int, end:int) -> int:
        start += self.N-1
        end += self.N

        result = (10**9+1, 0)

        while start < end:
            if start & 1:
                result = min(result, self.tree[start], key=lambda x:(x[0],x[1]))
                start += 1

            if end & 1:
                end -= 1
                result = min(result, self.tree[end], key=lambda x:(x[0],x[1]))

            start >>= 1
            end >>= 1

        return result[1]


def main():
    N = int(input())
    arr = list(map(int, input().split()))
    ST = SegmentTree(arr)
    for _ in range(int(input())):
        order, i, v = map(int, input().split())
        if order & 1:
            ST.update(i, v)
        else:
            print(ST.find(i, v))



if __name__ == "__main__":
    main()
```

### [영준](</수열과 쿼리16/영준.py>)

```py


```

<br/>

</details>

<br/><br/>

# 지난주 스터디 문제

<details markdown="1">
<summary>접기/펼치기</summary>

## [코드트리 파일관리](https://www.codetree.ai/problems/codetree-file-management/description)

### [민웅](./코드트리%20파일관리/민웅.py)

```py
import sys
import heapq
input = sys.stdin.readline

N = int(input())

n_lst = list(map(int, input().split()))
hq = []

for i in range(N):
    heapq.heappush(hq, n_lst.pop())

check = N
score = 0
while check > 1:
    tmp1 = heapq.heappop(hq)
    tmp2 = heapq.heappop(hq)
    score += tmp1 + tmp2
    heapq.heappush(hq, tmp1+tmp2)
    check -= 1

print(score)
```

### [상미](./코드트리%20파일관리/상미.py)

```py

```

### [성구](./코드트리%20파일관리/성구.py)

```py
import sys, heapq
input = sys.stdin.readline


N = int(input())
arr = list(map(int, input().split()))

que = []
for i in range(N):
    heapq.heappush(que, arr[i])
total = 0
while len(que) >1:
    f1 = heapq.heappop(que)
    f2 = heapq.heappop(que)
    total += f1+f2
    heapq.heappush(que, (f1+f2))

print(total)
```

### [영준](./코드트리%20파일관리/영준.py)

```py
import heapq

N = int(input())
arr = list(map(int, input().split()))

heapq.heapify(arr)

s = 0
while len(arr)>1:
    a = heapq.heappop(arr)
    b = heapq.heappop(arr)
    heapq.heappush(arr, a+b)
    s += a+b

print(s)
```

## [점프점프](https://www.codetree.ai/problems/jump-jump/description)

### [민웅](./점프점프/민웅.py)

```py
import sys

N = int(input())

n_lst = list(map(int, input().split()))

dp = [0]*N

for i in range(N-1):
    tmp = n_lst[i]
    for j in range(i+1, min(i+tmp+1, N)):
        dp[j] = max(dp[j], dp[i]+1)

print(max(dp))
```

### [상미](./점프점프/상미.py)

```py

```

### [성구](./점프점프/성구.py)

```py
import sys
input = sys.stdin.readline

N = int(input())
arr = list(map(int, input().split()))

dp = [0] * N
for i in range(1, min(arr[0]+1, N)):
    dp[i] += 1
for i in range(1, N):
    for j in range(1, min(arr[i]+1, N-i)):
        dp[i+j] = max(dp[i+j], dp[i]+1)
print(dp[-1])
```

### [영준](./점프점프/영준.py)

```py
N = int(input())
arr = list(map(int, input().split()))

D = [0]*(N+1)
for i in range(1, N+1):
    max_jmp = 0
    for j in range(max(0, i-100), i):
        if i-j<=arr[j] and max_jmp < D[j] + 1:
            max_jmp = D[j] + 1
    D[i] = max_jmp

print(max(D[:N]))
```

## [거리두기](https://www.codetree.ai/problems/keeping-distance/description)

### [민웅](./거리두기/민웅.py)

```py
import sys
input = sys.stdin.readline

def check(num):
    cnt = 1
    tmp_sum = 0
    for i in range(N):
        tmp_sum += n_lst[i]
        if tmp_sum > num:
            cnt += 1
            tmp_sum = n_lst[i]
        if cnt > M+1:
            return cnt

    return cnt

N, M = map(int, input().split())
n_lst = list(map(int, input().split()))

max_num = sum(n_lst)
l = max(n_lst)
r = max_num
ans = -1
while l <= r:
    mid = (l+r)//2
    tmp = check(mid)

    if tmp > M+1:
        l = mid + 1
    else:
        ans = mid
        r = mid - 1
    # print(l, r)

print(ans)
```

### [상미](./거리두기/상미.py)

```py

```

### [성구](./거리두기/성구.py)

```py
import sys
input = sys.stdin.readline

N, M = map(int, input().split())
guests = list(map(int, input().split()))

s,e = 0, sum(guests)
minv = 1000000001
while e>s:
    maxv = 0
    mid = (s+e)//2
    count = 0
    people = 0
    for i in range(N):
        people += guests[i]
        if  people >= mid:
            maxv = max(maxv, people-guests[i])
            count += 1
            people = guests[i]
    else:
        maxv = max(maxv, people)


    if count <= M:
        minv = min(minv, maxv)
        e = mid
    else:
        s = mid+1

print(minv)
```

### [영준](./거리두기/영준.py)

```py

```

## [회사 문화 1](https://www.acmicpc.net/problem/14267)

### [민웅](./회사문화/민웅.py)

```py
# 14267_회사 문화1_Business Culture
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline


def dfs(i, s):
    s += score[i]
    dp[i] += s
    for node in adjL[i]:
        dfs(node, s)


N, M = map(int, input().split())

parent = list(map(int, input().split()))
adjL = [[] for _ in range(N+1)]

for i in range(1, N):
    tmp = parent[i]
    adjL[tmp].append(i+1)

score = [0]*(N+1)
dp = [0]*(N+1)

for _ in range(M):
    node, s = map(int, input().split())
    score[node] += s

dfs(1, 0)
# print(score)
print(*dp[1:])

```

### [상미](./회사문화/상미.py)

```py

```

### [성구](./회사문화/성구.py)

```py
# 14267 회사 문화 1
import sys
input = sys.stdin.readline


def solution():
    N, M = map(int, input().split())
    parent = tuple(map(int, input().split()))
    child = [[] for _ in range(N)]
    start = 0
    for i in range(N):
        if parent[i] == -1:
            start = i
            continue
        child[parent[i]-1].append(i)

    claps = [0] * N
    for _ in range(M):
        i, w = map(int, input().split())
        claps[i-1] += w

    stack = [start]
    while stack:
        spot = stack.pop()
        for node in child[spot]:
            stack.append(node)
            claps[node] += claps[spot]

    print(*claps)
    return


if __name__ == "__main__":
    solution()
```

### [영준](./회사문화/영준.py)

```py

```

## [사회망 서비스](https://www.acmicpc.net/problem/2533)

### [민웅](./사회망%20서비스/민웅.py)

```py
# 2533_사회망서비스_Social Network Service
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

def dfs(i):
    visited[i] = 1
    dp[i][0] = 0
    dp[i][1] = 1
    for node in adjL[i]:
        if not visited[node]:
            dfs(node)
            dp[i][0] += dp[node][1]
            dp[i][1] += min(dp[node][0], dp[node][1])


N = int(input())

adjL = [[] for _ in range(N+1)]

for _ in range(N-1):
    u, v = map(int, input().split())
    adjL[u].append(v)
    adjL[v].append(u)

dp = [[0, 0] for _ in range(N+1)]
visited = [0]*(N+1)

dfs(1)

print(min(dp[1]))
```

### [상미](./사회망%20서비스/상미.py)

```py

```

### [성구](./사회망%20서비스성구.py)

```py
# 2533 사회망 서비스
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline


N = int(input())
graph = [[] for _ in range(N+1)]
for _ in range(N-1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

def dfs(x:int, parent:int):
    zero, one = 0, 1
    for node in graph[x]:
        if node == parent:
            continue
        pz, po = dfs(node, x)
        zero += po
        one += min(pz, po)
    return zero, one

print(min(dfs(1,0)))
```

### [영준](./사회망%20서비스/영준.py)

```py

```

</details>

<br/><br/>

# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>

### [세그먼트 트리](https://coyote.tistory.com/9)

![예시사진](./images/graph.png)

</details>
