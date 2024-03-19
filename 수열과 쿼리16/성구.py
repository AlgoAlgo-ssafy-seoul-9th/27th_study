# 14428 수열과 쿼리 16
import sys
input = sys.stdin.readline

# 반복문 구현
 
class SegmentTree:

    def __init__(self, arr:list) -> None:
        self.N = len(arr)
        self.tree = [[0,0] for _ in range(self.N << 1)]
        self.build(arr)
        return
    
    def build(self, arr:list) -> None:
        for i in range(self.N):
            self.tree[i+self.N] = (arr[i], i+1)

        for i in range(self.N-1, 0, -1):
            self.tree[i] = min(self.tree[i<<1], self.tree[(i<<1)+1], key=lambda x:(x[0],x[1]))

        return

    def update(self, idx:int, val:int) -> None:
        index = idx-1+self.N
        self.tree[index] = (val, idx)

        index >>= 1

        while index:
            self.tree[index] = min(self.tree[index<<1], self.tree[(index<<1)+1], key=lambda x:(x[0],x[1]))
            index >>= 1
        return
    

    def find(self, start:int, end:int) -> int:
        start += self.N-1
        end += self.N

        result = (10**9+1, 0)
        
        while start < end:
            if start & 1:
                result = min(result, self.tree[start], key=lambda x:(x[0],x[1]))
                start += 1
            
            if end & 1:
                end -= 1
                result = min(result, self.tree[end], key=lambda x:(x[0],x[1]))

            start >>= 1
            end >>= 1
        
        return result[1]


def main():
    N = int(input())
    arr = list(map(int, input().split()))
    ST = SegmentTree(arr)
    for _ in range(int(input())):
        order, i, v = map(int, input().split())
        if order & 1:
            ST.update(i, v)
        else:
            print(ST.find(i, v))



if __name__ == "__main__":
    main()



''' 재귀 구현
class SegmentTree:
    def __init__(self, arr:list) -> None:
        self.segment_tree = [10**9] + [10**9] * (4*len(arr))
        self.arr = arr
        self.build(0, len(arr)-1, 1)
        return

    def build(self, left, right, parent) -> tuple:
        if left == right:
            self.segment_tree[parent] = (left, self.arr[left]) 
            return self.segment_tree[parent]
        mid = (left + right) // 2
        left_idx, left_val = self.build(left, mid, parent*2)
        right_idx, right_val = self.build(mid+1, right, parent*2+1)
        if left_val < right_val:
            self.segment_tree[parent] = (left_idx, left_val)
        elif left_val == right_val:
            if left_idx <= right_idx:
                self.segment_tree[parent] = (left_idx, left_val)
            else:
                self.segment_tree[parent] = (right_idx, right_val)
        else:
            self.segment_tree[parent] = (right_idx, right_val)
        return self.segment_tree[parent]

    def query(self, s:int, e:int, now_idx:int, left:int, right:int) -> int:
        # s, e: now_idx가 포함하는 범위 
        # left, right: 구해야하는 범위
        # now_idx: 현재 segment_tree 위치
        if left > e or right < s:
            return (10**9,10**9)
        if left <= s and right >= e:
            return self.segment_tree[now_idx]
        mid = (s+e)//2
        l_idx, l_val = self.query(s, mid, now_idx*2, left, right)
        r_idx, r_val = self.query(mid+1, e, now_idx*2+1, left, right)
        if l_val < r_val:
            return (l_idx, l_val)
        elif l_val == r_val:
            if l_idx <= r_idx:
                return (l_idx, l_val)
            else:
                return (r_idx, r_val)
        else:
            return (r_idx, r_val)
    
    def update(self, s:int, e:int, now_idx:int, idx:int, val:int) -> None:
        # s,e: now_idx가 포함하는 범위
        # 현재 노드: now_idx
        # 변경할 인덱스와 값: idx, val
        if s == e:
            self.segment_tree[now_idx] = (idx, val)
            return
        
        mid = (s+e)//2
        if s <= idx <= mid:
            self.update(s, mid, now_idx*2, idx, val)
        elif mid+1 <= idx <= e:
            self.update(mid+1, e, now_idx*2+1, idx, val)
        
        left, right = now_idx*2, now_idx*2+1

        if self.segment_tree[left][1] == self.segment_tree[right][1]:
            if self.segment_tree[left][0] <= self.segment_tree[right][0]:
                self.segment_tree[now_idx] = self.segment_tree[left]
            else:
                self.segment_tree[now_idx] = self.segment_tree[right]
        else:
            self.segment_tree[now_idx] = min(self.segment_tree[left], self.segment_tree[right], key=lambda x:x[1])
        
        return 

def main():
    N = int(input())
    arr = list(map(int, input().split()))
    ST = SegmentTree(arr)
    M = int(input())
    for _ in range(M):
        order, i, v = map(int, input().split())
        if order == 1:
            ST.update(0, N-1, 1, i-1, v)
        else:
            print(ST.query(0, N-1, 1, i-1, v-1)[0]+1)



if __name__ == "__main__":
    main()


'''