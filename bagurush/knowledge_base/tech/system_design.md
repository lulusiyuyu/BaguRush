# 系统设计核心知识点

## 1. 负载均衡（Load Balancing）

### 什么是负载均衡
将网络请求/流量分发到多台服务器，避免单点过载，提升可用性和吞吐量。

### 负载均衡算法
| 算法 | 原理 | 适用场景 |
|---|---|---|
| 轮询（Round Robin） | 依次轮流分发 | 请求耗时相近 |
| 加权轮询 | 按权重比例分发 | 服务器性能不均 |
| 最少连接 | 分发给当前连接数最少的 | 长连接/耗时不均 |
| IP Hash | 根据客户端 IP 哈希，固定路由 | Session 亲和性 |
| 一致性哈希 | 环形哈希，节点增减影响范围小 | 分布式缓存 |

### 四层 vs 七层负载均衡
- **L4（传输层）**：基于 IP + 端口，性能高（如 LVS、AWS NLB）。
- **L7（应用层）**：基于 HTTP 内容（URL、Header），功能丰富（如 Nginx、AWS ALB）。

### 高可用：主备 + 心跳
Keepalived 实现 VRRP：主 LB 故障时，备用 LB 接管 VIP，秒级切换。

---

## 2. 缓存策略（Redis）

### 缓存读写策略
| 模式 | 读流程 | 写流程 | 一致性 |
|---|---|---|---|
| **Cache-Aside（旁路缓存）** | 先读缓存，Miss 则读 DB 回填 | 写 DB 后删缓存 | 最终一致 |
| **Read/Write-Through** | 缓存代理 DB 读写 | 写缓存，缓存同步写 DB | 强一致 |
| **Write-Behind（Write-Back）** | — | 先写缓存，异步批量写 DB | 弱一致 |

**Cache-Aside 常见问题**：
- **缓存击穿**：热点 key 过期瞬间大量请求打到 DB → 互斥锁 or 提前续期。
- **缓存穿透**：查询不存在的 key（DB 也没有）→ 布隆过滤器 or 空值缓存（短 TTL）。
- **缓存雪崩**：大量 key 同时过期 → 随机 TTL、预热、多级缓存。

### Redis 数据结构
| 类型 | 底层实现 | 典型用途 |
|---|---|---|
| String | SDS（简单动态字符串） | 计数器、Session、分布式锁 |
| Hash | listpack / hashtable | 对象存储（用户信息） |
| List | listpack / quicklist | 消息队列、时间线 |
| Set | listpack / hashtable | 标签、去重、共同好友 |
| ZSet | listpack / skiplist+hashtable | 排行榜、延迟队列 |

### Redis 持久化
- **RDB**：快照，定期全量保存 `.rdb`，恢复快，有数据丢失风险。
- **AOF**：追加写操作日志，可配置 always/everysec/no，更安全但文件大。
- **混合持久化（Redis 4+）**：RDB + AOF 增量，兼顾速度与安全。

### Redis 分布式锁
```python
# SET NX PX 原子操作
SET key value NX PX 30000
# 释放：使用 Lua 脚本确保原子性（避免释放他人的锁）
if redis.get(key) == value:
    redis.delete(key)
```

---

## 3. 消息队列（Message Queue）

### 核心作用
- **解耦**：生产者/消费者独立部署，互不影响。
- **削峰**：流量高峰时缓冲请求，保护下游服务。
- **异步**：非核心流程异步处理，加速主链路响应。

### Kafka vs RabbitMQ
| 特性 | Kafka | RabbitMQ |
|---|---|---|
| 模型 | 发布/订阅，日志追加 | 消息队列（AMQP） |
| 吞吐量 | 极高（百万 TPS） | 较高（万级 TPS） |
| 消息保留 | 可持久化，可重复消费 | 消费后删除 |
| 延迟 | 毫秒级 | 微秒级 |
| 适用场景 | 日志收集、事件流、大数据 | 任务队列、RPC、低延迟 |

### Kafka 核心概念
- **Topic**：消息分类，分若干 **Partition**（物理分片）。
- **Partition**：有序、不可变的消息日志，每条消息有 offset。
- **Consumer Group**：同一组内每个 Partition 只有一个 Consumer，实现负载均衡。
- **Replication**：每个 Partition 有多副本（Leader + Follower），保证高可用。

### 消息可靠性保障
1. **生产者确认（acks）**：`acks=all` 等待所有副本确认。
2. **消费者手动 ACK**：处理成功后才提交 offset。
3. **幂等性**：消费逻辑支持重复消费（数据库唯一键/状态机）。

---

## 4. 数据库选型（SQL vs NoSQL）

### 关系型数据库（SQL）
**代表**：MySQL、PostgreSQL  
**特点**：ACID 事务、结构化 Schema、SQL 查询、JOIN 支持  
**适用**：事务型业务（订单、支付、用户）、数据一致性要求高

### NoSQL 数据库分类
| 类型 | 代表 | 特点 | 适用 |
|---|---|---|---|
| 文档型 | MongoDB | 灵活 Schema，JSON 存储 | 内容管理、日志 |
| 键值型 | Redis | 极高读写性能 | 缓存、会话 |
| 列族型 | HBase, Cassandra | 海量数据，高写入 | 时序数据、日志分析 |
| 图数据库 | Neo4j | 图遍历、关系查询 | 社交网络、知识图谱 |

### MySQL 索引优化
- **B+ 树索引**：InnoDB 默认，支持范围查询。
- **联合索引最左前缀**：`(a, b, c)` 索引，查询须从最左列开始。
- **覆盖索引**：查询列全部在索引中，无需回表（Using index）。
- **EXPLAIN 分析**：关注 `type`（ref > range > index > ALL）和 `Extra`。

---

## 5. 分布式系统

### CAP 理论
分布式系统无法同时满足：
- **C（Consistency）**：所有节点同一时刻数据一致。
- **A（Availability）**：每次请求都能收到响应（不保证最新）。
- **P（Partition Tolerance）**：网络分区时系统继续运行。

**实际选择**：网络分区在分布式系统中不可避免，因此只能在 CP 或 AP 间权衡：
- CP：ZooKeeper、HBase（牺牲可用性保强一致）
- AP：Cassandra、DynamoDB（牺牲强一致保可用性）

### BASE 理论
对 ACID 的弱化，针对大规模分布式：
- **Basically Available（基本可用）**：允许部分降级。
- **Soft State（软状态）**：数据可以有中间状态。
- **Eventual Consistency（最终一致）**：不要求实时一致，但最终趋于一致。

### 分布式事务
| 方案 | 原理 | 优缺点 |
|---|---|---|
| 2PC | 协调者 + 参与者两阶段提交 | 强一致，协调者单点，性能差 |
| SAGA | 本地事务链 + 补偿事务 | 最终一致，复杂度高 |
| TCC | Try-Confirm-Cancel | 高性能，业务侵入性强 |
| 消息事务 | 本地事务 + 可靠消息 | 简单，最终一致 |

---

## 6. 微服务架构

### 核心优点
- 独立部署、独立扩展。
- 技术栈多元（多语言）。
- 故障隔离。

### 核心挑战
- 服务发现（Consul、Nacos、Kubernetes DNS）。
- 负载均衡（客户端 Ribbon，服务端 Nginx）。
- 链路追踪（Jaeger、Zipkin、SkyWalking）。
- 分布式配置（Apollo、Nacos）。

### 服务网格（Service Mesh）
Istio + Envoy：将服务治理逻辑从应用代码中抽离到 Sidecar Proxy，实现流量控制、安全、可观测性。

---

## 7. API 设计规范

### RESTful API 设计
```
资源名用名词复数：GET /users，POST /users，GET /users/{id}
动作用 HTTP Method：GET(查) POST(创) PUT(全量改) PATCH(部分改) DELETE(删)
状态码语义：200 OK, 201 Created, 204 No Content, 400 Bad Request, 401 Unauthorized, 404 Not Found, 500 Internal Server Error
版本化：URL 前缀 /v1/users 或 Header Accept: application/vnd.api+json;version=1
```

### 幂等性
- GET、PUT、DELETE 天然幂等。
- POST 通常不幂等（需客户端传唯一 `idempotency-key`）。

---

## 8. 限流与熔断

### 限流算法
| 算法 | 原理 | 特点 |
|---|---|---|
| 固定窗口 | 时间窗口内计数 | 简单，窗口边界有突刺 |
| 滑动窗口 | 滑动时间统计 | 更平滑，Redis ZSet 实现 |
| 漏桶 | 固定速率流出，多余丢弃 | 平滑输出，无法应对突发 |
| 令牌桶 | 固定速率生产令牌，有令牌才能请求 | 允许一定突发（Guava RateLimiter） |

### 熔断器（Circuit Breaker）
状态机：**Closed → Open → Half-Open**
- `Closed`：正常请求，错误率超阈值 → `Open`。
- `Open`：所有请求直接拒绝（快速失败），超时后 → `Half-Open`。
- `Half-Open`：放行少量请求试探，成功 → `Closed`，失败 → `Open`。

框架：Hystrix（Netflix，已停维）、Resilience4j（Java）、`tenacity`（Python）。
