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