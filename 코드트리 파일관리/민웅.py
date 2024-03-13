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