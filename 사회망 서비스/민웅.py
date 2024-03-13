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