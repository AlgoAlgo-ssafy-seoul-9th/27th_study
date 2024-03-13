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
