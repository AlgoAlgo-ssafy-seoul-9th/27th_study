N = int(input())
arr = list(map(int, input().split()))

D = [0]*(N+1)
for i in range(1, N+1):
    max_jmp = 0
    for j in range(max(0, i-100), i):
        if i-j<=arr[j] and max_jmp < D[j] + 1:
            max_jmp = D[j] + 1
    D[i] = max_jmp
 
print(max(D[:N]))
