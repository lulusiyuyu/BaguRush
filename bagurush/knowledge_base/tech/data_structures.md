# 数据结构与算法核心知识点

## 1. 数组与链表

### 数组（Array）
- **底层**：连续内存块，通过下标 O(1) 随机访问。
- **优点**：缓存友好（空间局部性），随机访问快。
- **缺点**：插入/删除 O(n)（需移动元素），大小固定（静态数组）。

### 链表（Linked List）
- **单链表**：每个节点存储数据 + 指向下一节点的指针。
- **双链表**：额外存储前驱指针，`collections.deque` 底层实现。
- **优点**：插入/删除 O(1)（已知节点位置），无需连续内存。
- **缺点**：随机访问 O(n)，额外指针内存开销，不缓存友好。

| 操作 | 数组 | 链表 |
|---|---|---|
| 随机访问 | O(1) | O(n) |
| 头部插入/删除 | O(n) | O(1) |
| 尾部插入/删除 | O(1) | O(1)（有尾指针） |
| 查找 | O(n) | O(n) |

---

## 2. 栈与队列

### 栈（Stack）—— LIFO
```python
stack = []
stack.append(1)   # push
stack.pop()       # pop，O(1)
```
应用：函数调用栈、括号匹配、DFS、表达式求值。

### 队列（Queue）—— FIFO
```python
from collections import deque
queue = deque()
queue.append(1)    # enqueue，O(1)
queue.popleft()    # dequeue，O(1)
```
应用：BFS、任务调度、消息队列。

### 单调栈（Monotonic Stack）
维护一个单调递增/递减的栈，用于求「下一个更大/小元素」：
```python
def next_greater(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # 存储下标
    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            result[stack.pop()] = num
        stack.append(i)
    return result
```

---

## 3. 哈希表（Hash Table）

### 核心原理
将 key 通过哈希函数映射到数组下标，实现 O(1) 平均查找。

### 哈希冲突解决
1. **链地址法（Chaining）**：桶存储链表，Java HashMap 使用。
2. **开放寻址法（Open Addressing）**：Python dict 使用，遇冲突向后探测空位。
   - 线性探测：`h(k, i) = (h(k) + i) % m`
   - 二次探测：`h(k, i) = (h(k) + i²) % m`

### 负载因子（Load Factor）
`α = n/m`（元素数/桶数），超过阈值（Java 0.75，Python 2/3）时扩容。

---

## 4. 二叉树与搜索树

### 二叉搜索树（BST）
左子树所有节点 < 根 < 右子树所有节点。中序遍历结果有序。
- 查找/插入/删除：平均 O(log n)，最坏 O(n)（退化成链表）。

### AVL 树
**自平衡 BST**，保证任意节点左右子树高度差 ≤ 1。
- 通过**旋转**（左旋/右旋/左右旋/右左旋）维护平衡。
- 查找/插入/删除：严格 O(log n)。
- 缺点：频繁旋转，写操作较慢。

### 红黑树（Red-Black Tree）
**近似平衡 BST**，五条性质保证最长路径 ≤ 最短路径 2 倍：
1. 节点是红色或黑色。
2. 根节点是黑色。
3. 叶子节点（NIL）是黑色。
4. 红色节点的子节点必须是黑色（不能连续红色）。
5. 从根到任意叶子路径上的黑色节点数相同。

操作复杂度：O(log n)。C++ `std::map`、Java `TreeMap` 使用红黑树。

### B/B+ 树
为**磁盘 I/O 优化**的多路平衡搜索树，一个节点存储多个 key，减少树高（减少磁盘读次数）。

**B 树 vs B+ 树**：
| 特性 | B 树 | B+ 树 |
|---|---|---|
| 数据存储位置 | 所有节点都存数据 | 只有叶子节点存数据 |
| 叶子节点链接 | 无 | 双向链表（支持范围查询） |
| 查找效率 | O(log n)，不稳定 | O(log n)，叶子层统一 |
| 范围查询 | 需回溯 | 直接遍历叶子链表 |

**MySQL InnoDB 使用 B+ 树**原因：
- 叶子节点链表支持高效范围查询（`BETWEEN`、`ORDER BY`）。
- 内部节点只存 key，单节点存更多 key，树更矮，I/O 更少。

---

## 5. 堆（Heap）

**最大堆/最小堆**：完全二叉树，父节点总是 ≥（最大堆）或 ≤（最小堆）子节点。

```python
import heapq

# Python 默认最小堆
heap = [3, 1, 4, 1, 5]
heapq.heapify(heap)           # O(n)
heapq.heappush(heap, 2)       # O(log n)
heapq.heappop(heap)           # O(log n)，返回最小值

# 最大堆：存负数
max_heap = [-x for x in [3, 1, 4]]
heapq.heapify(max_heap)
max_val = -heapq.heappop(max_heap)
```

**应用**：
- Top-K 问题（维护 K 大/小堆）
- 优先队列（Dijkstra、Prim）
- 堆排序

---

## 6. 图（Graph）

### 表示方式
- **邻接矩阵**：`O(V²)` 空间，`O(1)` 查边，适合稠密图。
- **邻接表**：`O(V+E)` 空间，适合稀疏图。

### BFS（广度优先搜索）
使用队列，**层序遍历**，求最短路径（无权图）：
```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### DFS（深度优先搜索）
使用栈（递归或显式栈），用于拓扑排序、连通分量：
```python
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
```

---

## 7. 排序算法

### 快速排序（Quick Sort）
- **平均 O(n log n)**，最坏 O(n²)（已排序数组 + 选最后元素为 pivot）。
- 原地排序，不稳定。
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)
```

### 归并排序（Merge Sort）
- 稳定，**O(n log n)** 稳定，额外 O(n) 空间。
- Python `sorted()` / `list.sort()` 使用 Timsort（归并 + 插入排序）。

### 堆排序（Heap Sort）
- **O(n log n)**，原地排序，不稳定，建堆 O(n)，n 次 pop O(n log n)。

### 算法对比
| 算法 | 平均 | 最坏 | 空间 | 稳定 |
|---|---|---|---|---|
| 快速排序 | O(n log n) | O(n²) | O(log n) | 否 |
| 归并排序 | O(n log n) | O(n log n) | O(n) | 是 |
| 堆排序 | O(n log n) | O(n log n) | O(1) | 否 |
| 插入排序 | O(n²) | O(n²) | O(1) | 是 |
| 冒泡排序 | O(n²) | O(n²) | O(1) | 是 |

---

## 8. 动态规划（DP）

**核心思路**：将问题分解为子问题，通过记忆化避免重复计算。

**四步法**：
1. **定义状态** `dp[i]` 的含义
2. **状态转移方程**
3. **初始化边界**
4. **计算顺序**

```python
# 经典例子：最长递增子序列（LIS）
def lis(nums):
    n = len(nums)
    dp = [1] * n  # dp[i] = 以 nums[i] 结尾的 LIS 长度
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)  # O(n²)

# O(n log n) 解法：二分查找维护 tails 数组
import bisect
def lis_fast(nums):
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

---

## 9. 贪心算法

**核心**：每步选取局部最优解，期望得到全局最优解（需证明贪心选择性质）。

```python
# 区间调度最大化（选最多不重叠区间）
def max_intervals(intervals):
    intervals.sort(key=lambda x: x[1])  # 按结束时间排序
    count, end = 0, float('-inf')
    for start, finish in intervals:
        if start >= end:
            count += 1
            end = finish
    return count
```

---

## 10. 二分查找

```python
import bisect

def binary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = l + (r - l) // 2  # 避免溢出
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1

# 标准库：bisect_left 找第一个 >= target 的位置
pos = bisect.bisect_left(arr, target)
```

**变体**：
- 找第一个满足条件的位置（左边界）：`bisect_left`
- 找最后一个满足条件的位置（右边界）：`bisect_right - 1`
- 旋转数组二分：判断哪半边有序，针对性收缩范围
