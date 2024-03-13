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
