# 27th_study

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## [미정](https://www.acmicpc.net/problem/11286)

### ❗ heapq 쓰지 않고 만들어 쓰기 ❗

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

## [미로 탈출하기](https://www.codetree.ai/problems/escape-the-maze/description)

### [민웅](./미로%20탈출하기/민웅.py)

```py

```

### [상미](./미로%20탈출하기/상미.py)

```py

```

### [성구](./미로%20탈출하기/성구.py)

```py

```

### [영준](./미로%20탈출하기/영준.py)

```py


```

</details>

<br/><br/>

# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>

[알고리즘 설명](https://l1m3kun.tistory.com/entry/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%9A%B0%EC%84%A0%EC%88%9C%EC%9C%84-%ED%81%90%EC%99%80-%ED%9E%99Priority-Queue-Heap)

</details>
