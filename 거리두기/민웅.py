import sys
input = sys.stdin.readline

def check(num):
    cnt = 1
    tmp_sum = 0
    for i in range(N):
        tmp_sum += n_lst[i]
        if tmp_sum > num:
            cnt += 1
            tmp_sum = n_lst[i]
        if cnt > M+1:
            return cnt
    
    return cnt

N, M = map(int, input().split())
n_lst = list(map(int, input().split()))

max_num = sum(n_lst)
l = max(n_lst)
r = max_num
ans = -1
while l <= r:
    mid = (l+r)//2
    tmp = check(mid)

    if tmp > M+1:
        l = mid + 1
    else:
        ans = mid
        r = mid - 1
    # print(l, r)

print(ans)