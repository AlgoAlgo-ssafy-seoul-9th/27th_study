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