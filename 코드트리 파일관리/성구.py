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