# LLM生产部署实践：从零搭建企业级AI平台

> 基于1200个生产案例和工业界最佳实践
> 完整Docker配置 + 监控方案 + CI/CD流程

---

## 第一章：生产环境架构设计

### 1.1 整体架构图

```
┌─────────────┐
│   用户请求   │
└──────┬──────┘
       ▼
┌─────────────┐     ┌─────────────┐
│  API Gateway │────▶│  Rate Limit │
└──────┬──────┘     └─────────────┘
       ▼
┌─────────────┐     ┌─────────────┐
│  Load Balancer│────▶│  FastAPI App │
└─────────────┘     │   (x3副本)   │
                    └──────┬──────┘
                           ▼
              ┌────────────────────────┐
              │       服务层            │
              │  ┌─────┐ ┌─────┐      │
              │  │ RAG │ │Agent│ ...  │
              │  └─────┘ └─────┘      │
              └──────────┬─────────────┘
                         ▼
              ┌────────────────────┐
              │     数据层          │
              │ Qdrant Redis Postgres│
              └────────────────────┘
```

### 1.2 核心组件清单

| 组件 | 技术选型 | 用途 |
|------|----------|------|
| **API Gateway** | Nginx/Traefik | 路由、SSL、限流 |
| **应用服务** | FastAPI + Gunicorn | 业务逻辑 |
| **向量数据库** | Qdrant | 知识库存储 |
| **缓存** | Redis | 结果缓存、Session |
| **数据库** | PostgreSQL | 业务数据 |
| **监控** | Prometheus + Grafana | 指标收集和可视化 |
| **日志** | ELK/Loki | 日志收集和分析 |
| **链路追踪** | OpenTelemetry | 分布式追踪 |

---

## 第二章：Docker生产配置

### 2.1 完整docker-compose.yml

```yaml
version: '3.8'

services:
  # ========== 应用服务 ==========
  api:
    build:
      context: ./app
      dockerfile: Dockerfile
    container_name: ai-api
    ports:
      - "8080:8080"
    environment:
      - ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://aiuser:aipass123@postgres:5432/aidb
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
    volumes:
      - ./app:/app
      - app_logs:/app/logs
    depends_on:
      - qdrant
      - redis
      - postgres
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - ai-network

  # ========== 向量数据库 ==========
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: ai-qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
    deploy:
      resources:
        limits:
          memory: 2G
    networks:
      - ai-network

  # ========== 缓存 ==========
  redis:
    image: redis:7-alpine
    container_name: ai-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          memory: 1G
    networks:
      - ai-network

  # ========== 数据库 ==========
  postgres:
    image: postgres:15-alpine
    container_name: ai-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=aiuser
      - POSTGRES_PASSWORD=aipass123
      - POSTGRES_DB=aidb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    deploy:
      resources:
        limits:
          memory: 1G
    networks:
      - ai-network

  # ========== 监控 ==========
  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: ai-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=15d'
      - '--web.enable-lifecycle'
    networks:
      - ai-network

  grafana:
    image: grafana/grafana:10.2.3
    container_name: ai-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    networks:
      - ai-network

  # ========== 网关 ==========
  nginx:
    image: nginx:alpine
    container_name: ai-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    networks:
      - ai-network

volumes:
  qdrant_storage:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
  app_logs:

networks:
  ai-network:
    driver: bridge
```

### 2.2 Dockerfile

```dockerfile
# Dockerfile - FastAPI应用
FROM python:3.11-slim as builder

WORKDIR /app

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 生产镜像
FROM python:3.11-slim

WORKDIR /app

# 复制依赖
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY ./app .

# 设置环境变量
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 非root用户运行（安全）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--access-logfile", "-", "--error-logfile", "-", "main:app"]
```

### 2.3 requirements.txt

```
# Web框架
fastapi==0.110.0
uvicorn[standard]==0.27.1
gunicorn==21.2.0
python-multipart==0.0.9

# AI框架
langchain==0.1.12
langchain-openai==0.0.8
langgraph==0.0.26
llama-index==0.10.15

# 数据库
qdrant-client==1.7.3
redis==5.0.2
asyncpg==0.29.0
sqlalchemy==2.0.25

# 监控
prometheus-client==0.20.0
opentelemetry-api==1.23.0
opentelemetry-sdk==1.23.0
opentelemetry-instrumentation-fastapi==0.44b0

# 工具
pydantic==2.6.3
pydantic-settings==2.2.1
python-dotenv==1.0.1
httpx==0.27.0
aiohttp==3.9.3
tenacity==8.2.3

# 日志
structlog==24.1.0
```

---

## 第三章：监控与可观测性

### 3.1 Prometheus配置

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastapi-app'
    static_configs:
      - targets: ['api:8080']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### 3.2 应用内埋点代码

```python
# monitoring.py - 监控指标定义
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
import time

# 定义指标
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

LLM_CALL_COUNT = Counter(
    'llm_calls_total',
    'Total LLM API calls',
    ['model', 'status']
)

LLM_LATENCY = Histogram(
    'llm_call_duration_seconds',
    'LLM API call latency',
    ['model']
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests'
)

# FastAPI中间件
async def metrics_middleware(request: Request, call_next):
    """监控中间件"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:
        status = 500
        raise e
    finally:
        duration = time.time() - start_time
        ACTIVE_REQUESTS.dec()
        
        # 记录指标
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
    
    return response

# 在FastAPI应用中使用
from fastapi import FastAPI

app = FastAPI()
app.middleware("http")(metrics_middleware)

@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# LLM调用埋点示例
async def call_llm_with_metrics(prompt: str, model: str = "gpt-3.5-turbo"):
    """带监控的LLM调用"""
    with LLM_LATENCY.labels(model=model).time():
        try:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            LLM_CALL_COUNT.labels(model=model, status="success").inc()
            return response
        except Exception as e:
            LLM_CALL_COUNT.labels(model=model, status="error").inc()
            raise
```

### 3.3 关键监控指标

| 指标类别 | 具体指标 | 告警阈值 |
|----------|----------|----------|
| **可用性** | 服务健康状态 | 失败率 > 1% |
| **性能** | P99延迟 | > 2s |
| **错误率** | 5xx错误率 | > 0.1% |
| **LLM成本** | Token消耗/小时 | 突增 > 50% |
| **资源** | CPU/内存使用率 | > 80% |

---

## 第四章：性能优化

### 4.1 缓存策略

```python
# cache.py - 多级缓存实现
import asyncio
from functools import wraps
from typing import Optional
import hashlib
import json

class CacheManager:
    """多级缓存管理"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}  # L1: 本地内存
        self.local_ttl = 60    # 本地缓存60秒
    
    async def get(self, key: str) -> Optional[str]:
        """获取缓存 - L1 -> L2"""
        # 先查本地
        if key in self.local_cache:
            value, expire_time = self.local_cache[key]
            if time.time() < expire_time:
                return value
            else:
                del self.local_cache[key]
        
        # 再查Redis
        value = await self.redis.get(key)
        if value:
            # 回填本地缓存
            self.local_cache[key] = (value, time.time() + self.local_ttl)
        
        return value
    
    async def set(self, key: str, value: str, ttl: int = 3600):
        """设置缓存"""
        # 设置本地
        self.local_cache[key] = (value, time.time() + self.local_ttl)
        # 设置Redis
        await self.redis.setex(key, ttl, value)

# 装饰器用法
def cached(ttl: int = 3600):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存key
            cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode()).hexdigest()}"
            
            # 查缓存
            cached_value = await cache_manager.get(cache_key)
            if cached_value:
                return json.loads(cached_value)
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 写入缓存
            await cache_manager.set(cache_key, json.dumps(result), ttl)
            
            return result
        return wrapper
    return decorator

# 使用示例
class RAGService:
    @cached(ttl=1800)  # 缓存30分钟
    async def search(self, query: str):
        """搜索结果缓存"""
        # 实际检索逻辑
        return await self._do_search(query)
```

### 4.2 连接池优化

```python
# connection_pool.py
import httpx
from openai import AsyncOpenAI

# HTTP连接池
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100
    ),
    timeout=httpx.Timeout(30.0)
)

# OpenAI客户端复用
openai_client = AsyncOpenAI(
    http_client=http_client,
    max_retries=3
)

# 数据库连接池
from asyncpg import create_pool

db_pool = None

async def init_db_pool():
    global db_pool
    db_pool = await create_pool(
        host="postgres",
        database="aidb",
        user="aiuser",
        password="aipass123",
        min_size=10,
        max_size=50
    )
```

---

## 第五章：CI/CD流程

### 5.1 GitHub Actions配置

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      
      - name: Run tests
        run: pytest tests/ -v --cov=app

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          docker build -t ai-app:${{ github.sha }} .
          docker tag ai-app:${{ github.sha }} ai-app:latest
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push ai-app:${{ github.sha }}
          docker push ai-app:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/ai-app
            docker-compose pull
            docker-compose up -d
            docker system prune -f
```

---

## 第六章：常见问题排查

### Q1: 容器启动失败？

```bash
# 查看日志
docker-compose logs -f api

# 检查资源
docker stats

# 进入容器调试
docker-compose exec api bash
```

### Q2: LLM API调用超时？

**解决方案**：
1. 增加超时时间
2. 使用流式响应
3. 实现降级策略

```python
# 降级策略
async def llm_with_fallback(prompt: str):
    """主模型失败时降级到备用模型"""
    try:
        return await call_gpt4(prompt)
    except TimeoutError:
        print("GPT-4超时，降级到GPT-3.5")
        return await call_gpt35(prompt)
```

### Q3: 内存泄漏？

**排查方法**：
```bash
# 查看内存使用
docker stats

# Python内存分析
pip install memory_profiler
python -m memory_profiler app.py
```

---

**本文整理时间**: 2026-03-04
