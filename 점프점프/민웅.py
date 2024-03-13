import sys

N = int(input())

n_lst = list(map(int, input().split()))

dp = [0]*N

for i in range(N-1):
    tmp = n_lst[i]
    for j in range(i+1, min(i+tmp+1, N)):
        dp[j] = max(dp[j], dp[i]+1)

print(max(dp))