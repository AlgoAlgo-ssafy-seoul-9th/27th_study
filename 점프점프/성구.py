import sys
input = sys.stdin.readline

N = int(input())
arr = list(map(int, input().split()))

dp = [0] * N
for i in range(1, min(arr[0]+1, N)):
    dp[i] += 1
for i in range(1, N):
    for j in range(1, min(arr[i]+1, N-i)):
        dp[i+j] = max(dp[i+j], dp[i]+1)
print(dp[-1])