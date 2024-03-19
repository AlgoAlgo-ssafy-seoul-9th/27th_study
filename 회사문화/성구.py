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