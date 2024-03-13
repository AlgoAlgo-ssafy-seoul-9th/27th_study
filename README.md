# 27th_study

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## [미정](https://www.acmicpc.net/problem/11286)


### [민웅](./절댓값%20힙/민웅.py)

```py


```

### [상미](./절댓값%20힙/상미.py)

```py


```

### [성구](./절댓값%20힙/성구.py)

```py

```

### [영준](./절댓값%20힙/영준.py)

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

```

### [영준](./코드트리%20파일관리/영준.py)

```py

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

```

### [영준](./점프점프/영준.py)

```py

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

```

### [영준](./사회망%20서비스/영준.py)

```py

```

</details>

<br/><br/>

# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>



</details>
