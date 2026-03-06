# Python 核心知识点

## 1. GIL（全局解释器锁）

### 什么是 GIL
GIL（Global Interpreter Lock）是 CPython 解释器中的一把互斥锁，保证同一时刻只有一个线程执行 Python 字节码。它的存在是因为 CPython 的内存管理（引用计数）不是线程安全的。

### GIL 的影响
- **CPU 密集型任务**：多线程无法利用多核，性能不如单线程（线程切换有开销）。
- **I/O 密集型任务**：线程在等待 I/O 时会释放 GIL，多线程仍然有效。
- **多进程**：使用 `multiprocessing` 模块可绕过 GIL，每个进程有独立的 GIL。

### GIL 的释放时机
1. I/O 操作（网络、磁盘）
2. 每执行约 100 条字节码（可通过 `sys.setswitchinterval` 调整，默认 5ms）
3. 调用 C 扩展时（如 NumPy 的底层操作）

### 常见面试追问
- **为什么不去掉 GIL？** 去掉会破坏大量依赖 GIL 的 C 扩展库，且引用计数需要改为原子操作，性能损耗大。
- **Python 3.12 计划**：分阶段移除 GIL（PEP 703，Per-Interpreter GIL）。

---

## 2. 内存管理

### 引用计数（Reference Counting）
CPython 为每个对象维护一个 `ob_refcnt` 计数器：
- 对象被引用时加 1，引用消失时减 1。
- 计数归 0 时立即回收内存。

```python
import sys
a = []
print(sys.getrefcount(a))  # 2（a 本身 + getrefcount 的参数）
b = a
print(sys.getrefcount(a))  # 3
del b
print(sys.getrefcount(a))  # 2
```

**优点**：回收及时，无需暂停（Stop-the-World）。  
**缺点**：无法处理循环引用。

### 循环垃圾回收（Cyclic GC）
用于解决循环引用问题：
```python
a = []
b = []
a.append(b)
b.append(a)
del a, del b  # 引用计数不归 0，循环 GC 负责回收
```

Python GC 使用**分代回收**（Generational GC）：
- **0 代**：新创建的对象，回收最频繁。
- **1 代**：经过 1 次 GC 存活的对象。
- **2 代**：经过 2 次 GC 存活的对象，回收最少。

触发条件：某代对象数量超过阈值（默认 700/10/10）。

### 内存池（pymalloc）
CPython 对小对象（≤512 字节）使用内存池管理，避免频繁向 OS 申请和释放内存：
- **Arena**（256KB）→ **Pool**（4KB）→ **Block**（8~512 字节，8 字节对齐）。

### 常见陷阱
```python
# 小整数缓存：-5 ~ 256 的整数对象是单例
a = 256; b = 256; print(a is b)  # True
a = 257; b = 257; print(a is b)  # False（CPython 交互模式可能 True，编译后 False）

# 字符串驻留（interning）：符合标识符规则的字符串会被缓存
```

---

## 3. 装饰器（Decorator）

本质是**高阶函数**：接受函数作为参数，返回新函数。

### 基本实现
```python
import functools

def timer(func):
    @functools.wraps(func)  # 保留原函数的 __name__、__doc__ 等元信息
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时: {time.time() - start:.3f}s")
        return result
    return wrapper

@timer
def slow_func():
    import time; time.sleep(1)
```

### 带参数的装饰器
```python
def retry(max_times=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_times - 1:
                        raise
            return None
        return wrapper
    return decorator

@retry(max_times=5)
def unstable_api():
    ...
```

### 类装饰器
```python
class Singleton:
    _instances = {}
    def __call__(self, cls):
        @functools.wraps(cls)
        def get_instance(*args, **kwargs):
            if cls not in self._instances:
                self._instances[cls] = cls(*args, **kwargs)
            return self._instances[cls]
        return get_instance
```

---

## 4. 生成器与迭代器

### 迭代器协议
实现了 `__iter__()` 和 `__next__()` 方法的对象：
```python
class CountUp:
    def __init__(self, start, end):
        self.current = start
        self.end = end
    def __iter__(self):
        return self
    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        val = self.current
        self.current += 1
        return val
```

### 生成器（Generator）
用 `yield` 关键字实现的惰性迭代器，**按需生成值**，内存高效：

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 生成器表达式（比列表推导式节省内存）
gen = (x**2 for x in range(1000000))
```

### yield from
简化嵌套生成器：
```python
def chain(*iterables):
    for it in iterables:
        yield from it  # 等价于 for item in it: yield item
```

### send() 与双向通信
```python
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is None:
            break
        total += value

acc = accumulator()
next(acc)          # 启动生成器
acc.send(10)       # → 10
acc.send(20)       # → 30
```

---

## 5. 上下文管理器（Context Manager）

### __enter__ 和 __exit__
```python
class ManagedResource:
    def __enter__(self):
        print("获取资源")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("释放资源")
        return False  # False 表示不抑制异常，True 则抑制

with ManagedResource() as r:
    pass  # 离开 with 块时自动调用 __exit__
```

### contextlib.contextmanager
```python
from contextlib import contextmanager

@contextmanager
def timer_ctx(label):
    import time
    start = time.time()
    try:
        yield  # with 块执行位置
    finally:
        print(f"{label}: {time.time()-start:.3f}s")

with timer_ctx("查询"):
    run_query()
```

---

## 6. 异步编程（asyncio）

### 核心概念
- **协程（Coroutine）**：`async def` 定义，用 `await` 挂起。
- **事件循环（Event Loop）**：调度协程执行，单线程。
- **任务（Task）**：协程的包装，可并发执行。

```python
import asyncio

async def fetch(url):
    await asyncio.sleep(1)  # 模拟 I/O
    return f"data from {url}"

async def main():
    # 并发执行多个协程
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
    )
    print(results)

asyncio.run(main())
```

### async/await vs 多线程
| 维度 | asyncio | threading |
|---|---|---|
| 并发单位 | 协程 | 线程 |
| 切换方式 | 主动 yield（await） | 操作系统抢占 |
| 内存开销 | 极小（KB 级） | 较大（MB 级栈） |
| 适用场景 | I/O 密集型 | I/O 密集型 |
| 共享状态 | 无竞争（单线程） | 需要锁保护 |

---

## 7. 多线程 vs 多进程

### threading（多线程）
- 共享内存，通信简单。
- 受 GIL 限制，CPU 密集型无优势。
- 适用：I/O 密集型（网络请求、文件读写）。

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_url, url) for url in urls]
    results = [f.result() for f in futures]
```

### multiprocessing（多进程）
- 独立内存空间，通信需要 Queue/Pipe/共享内存。
- 绕过 GIL，适合 CPU 密集型任务。
- 进程启动开销大（约 10~100ms）。

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    results = list(executor.map(cpu_heavy_task, data_chunks))
```

---

## 8. 常见数据类型底层实现

### list（动态数组）
- 底层是 C 数组，存储对象指针。
- **动态扩容**：容量不足时按 `(size * 1.125 + 6)` 扩容（Python 3.x 近似），不是翻倍。
- `append` 均摊 O(1)，`insert(0, x)` O(n)，`pop()` O(1)，`pop(0)` O(n)。

### dict（哈希表）
- Python 3.7+ 保证**插入有序**（使用紧凑哈希表）。
- 底层：哈希数组（indices）+ 紧凑条目数组（entries）。
- 哈希冲突用**开放寻址法**（探测）。
- 负载因子超 2/3 时扩容（翻倍）。
- 查找/插入/删除均摊 O(1)。

```python
# dict 的键必须是可哈希对象（实现了 __hash__ 和 __eq__）
d = {(1, 2): "tuple", "str": "string"}  # 合法
# d = {[1,2]: "list"}  # TypeError: unhashable type: 'list'
```

### set（哈希集合）
- 基于哈希表实现，不存储值，只存储键。
- 成员检测 O(1)，集合运算（交/并/差）高效。
- `frozenset` 是不可变版本，可作为 dict 键。

### 总结对比
| 类型 | 底层结构 | 有序 | 可变 | 查找复杂度 |
|---|---|---|---|---|
| list | 动态数组 | ✅ | ✅ | O(n) |
| tuple | 静态数组 | ✅ | ❌ | O(n) |
| dict | 哈希表 | ✅(3.7+) | ✅ | O(1) 均摊 |
| set | 哈希表 | ❌ | ✅ | O(1) 均摊 |
| frozenset | 哈希表 | ❌ | ❌ | O(1) 均摊 |
