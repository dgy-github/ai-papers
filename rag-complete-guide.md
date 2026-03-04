# RAG技术完全指南：从零实现企业级知识库系统

> 基于《Retrieval-Augmented Generation for Large Language Models: A Survey》深度整理
> 不需要看其他书，只读这一篇就能动手实现

---

## 第一章：RAG到底是什么？（概念篇）

### 1.1 一句话理解RAG

**RAG = 搜索引擎 + ChatGPT**

传统ChatGPT的问题：
- 不知道你的私有数据（公司文档、个人笔记）
- 会产生"幻觉"（瞎编答案）
- 知识截止到训练时间

RAG的解决方式：
1. 把你的文档**向量化**存入数据库
2. 用户提问时，先**检索**相关文档
3. 把检索结果**塞给**ChatGPT作为参考
4. ChatGPT基于参考生成答案

### 1.2 RAG vs 微调（Fine-tuning）

| 对比项 | RAG | 微调 |
|--------|-----|------|
| **成本** | 低（只需API调用） | 高（需要GPU训练） |
| **时效性** | 实时（数据随时更新） | 滞后（需要重新训练） |
| **数据量** | 无限制 | 受限于模型容量 |
| **幻觉** | 较少（有参考依据） | 仍有 |
| **适用场景** | 知识库问答 | 特定风格/格式 |

**结论**：做知识库问答，优先选RAG，不是微调。

### 1.3 RAG的三大发展阶段

#### 阶段1：Naive RAG（基础版）

流程：
```
用户提问 → 向量化 → 向量检索 → 拼接Prompt → LLM生成
```

**代码实现**：
```python
# 最简RAG实现（50行代码）
from openai import OpenAI
import numpy as np

client = OpenAI()

# 1. 准备知识库（模拟）
documents = [
    "Python是一种高级编程语言，由Guido van Rossum于1991年创建",
    "Java是一种面向对象的编程语言，由Sun Microsystems于1995年发布",
    "JavaScript是网页开发的主要语言，可以在浏览器中运行"
]

# 2. 生成文档向量（实际生产用Embedding模型）
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

doc_embeddings = [get_embedding(doc) for doc in documents]

# 3. 相似度计算（余弦相似度）
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 4. 检索函数
def retrieve(query, top_k=2):
    query_emb = get_embedding(query)
    # 计算与所有文档的相似度
    similarities = [cosine_similarity(query_emb, doc_emb) 
                   for doc_emb in doc_embeddings]
    # 获取Top-K
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# 5. 生成答案
def rag_answer(query):
    # 检索相关文档
    relevant_docs = retrieve(query)
    context = "\n".join(relevant_docs)
    
    # 构建Prompt
    prompt = f"""基于以下参考资料回答问题：

参考资料：
{context}

问题：{query}

请基于参考资料回答，如果资料中没有相关信息，请明确说明。"""
    
    # 调用LLM
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 测试
print(rag_answer("Python是谁创建的？"))
```

**Naive RAG的问题**：
1. **检索不准**：用户问"Python性能如何"，可能召回"Java性能"的文档
2. **上下文太长**：召回10个文档，可能超出LLM上下文限制
3. **信息冲突**：多个文档答案矛盾，LLM不知道该信谁

#### 阶段2：Advanced RAG（进阶版）

针对Naive RAG的问题，逐个解决：

**问题1：检索不准 → 查询优化**

用户问："Python怎么样？"
这个查询太模糊，可能指：
- Python语言特性
- Python性能
- Python就业前景

**解决方案：查询重写（Query Rewriting）**

```python
# 查询重写实现
def rewrite_query(original_query):
    """用LLM重写查询，使其更具体"""
    prompt = f"""将用户的模糊查询重写为3个具体的搜索查询。

用户查询：{original_query}

请生成3个更具体的查询，涵盖不同方面："""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 解析返回的3个查询
    rewritten = response.choices[0].message.content
    # 实际实现需要解析文本
    return [original_query] + parse_queries(rewritten)

# 示例：
# 输入："Python怎么样？"
# 输出：["Python性能如何？", "Python适合做什么？", "Python学习难度？"]
```

**HyDE（Hypothetical Document Embeddings）技术**：

核心思想：先让LLM生成一个假设答案，然后用这个答案去检索

```python
defhyde_retrieval(query):
    """HyDE检索"""
    # Step 1: 生成假设答案
    hypothetical_prompt = f"""回答以下问题，提供一个简短的答案：
{query}"""
    
    hypo_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": hypothetical_prompt}]
    )
    hypothetical_answer = hypo_response.choices[0].message.content
    
    # Step 2: 用假设答案做检索（而不是原始查询）
    results = retrieve(hypothetical_answer, top_k=5)
    return results
```

**为什么HyDE有效？**
- 用户查询："Python性能"
- 生成的假设答案："Python是一种解释型语言，执行速度比C++慢，但开发效率高..."
- 假设答案包含更多关键词，更容易匹配到相关文档

**问题2：上下文太长 → 重排序（Reranking）**

不是召回越多越好，而是**越准越好**。

**两阶段检索**：
```
第一阶段（召回）：快速但粗略，召回100个候选
第二阶段（精排）：慢但精确，选出Top-5
```

```python
# 两阶段检索实现
from sentence_transformers import CrossEncoder

# 加载重排序模型（交叉编码器）
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def two_stage_retrieve(query, top_k=5):
    # 第一阶段：向量检索，召回100个
    candidates = vector_search(query, k=100)
    
    # 第二阶段：重排序，选出Top-5
    # 交叉编码器同时看查询和文档，更准确
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    
    # 按分数排序
    ranked = sorted(zip(candidates, scores), 
                   key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
```

**为什么交叉编码器比双编码器准？**

- **双编码器**（第一阶段）：查询和文档分别编码，然后算相似度
  - 优点：快，可以预先计算文档向量
  - 缺点：查询和文档没有交互，不准

- **交叉编码器**（第二阶段）：查询和文档一起输入模型
  - 优点：模型可以看到两者交互，更准
  - 缺点：慢，不能预先计算

**问题3：信息冲突 → 上下文压缩**

不是把所有检索结果都塞给LLM，而是**只保留最相关的部分**。

```python
# 上下文压缩实现
def compress_context(query, documents):
    """只保留与查询最相关的段落"""
    compressed = []
    for doc in documents:
        # 分割成段落
        paragraphs = doc.split('\n\n')
        
        # 计算每个段落与查询的相关性
        for para in paragraphs:
            relevance = calculate_relevance(query, para)
            if relevance > 0.7:  # 只保留高相关的
                compressed.append(para)
    
    return compressed
```

#### 阶段3：Modular RAG（模块化）

把RAG拆成独立模块，按需组合：

```
查询模块 → 检索模块 → 重排序模块 → 生成模块
   ↓           ↓           ↓           ↓
可替换      可替换      可替换      可替换
```

**模块示例**：
- **查询模块**：重写、扩展、HyDE
- **检索模块**：向量检索、关键词检索、图谱检索
- **重排序模块**：交叉编码器、LLM重排
- **生成模块**：不同Prompt策略

---

## 第二章：向量数据库选型（工具篇）

### 2.1 主流向量数据库对比

| 数据库 | 优点 | 缺点 | 适用场景 | 推荐指数 |
|--------|------|------|----------|----------|
| **Chroma** | 极简API，4个函数搞定 | 扩展性有限 | 原型开发、学习 | ⭐⭐⭐ |
| **Qdrant** | 性能好，Rust编写 | 相对较新 | 生产环境 | ⭐⭐⭐⭐⭐ |
| **Milvus** | 功能最全 | 部署复杂 | 大规模企业 | ⭐⭐⭐⭐ |
| **Pinecone** | 托管服务，零运维 | 贵 | 不想运维 | ⭐⭐⭐ |

**我的建议**：
- **学习阶段**：Chroma（最简单）
- **生产环境**：Qdrant（性能+易用平衡）

### 2.2 Qdrant完整使用指南

**安装启动**：
```bash
# Docker方式（推荐）
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Python客户端
pip install qdrant-client
```

**完整CRUD代码**：
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# 连接客户端
client = QdrantClient(url="http://localhost:6333")

# 1. 创建集合（类似MySQL的表）
collection_name = "knowledge_base"

# 如果集合已存在，先删除
client.delete_collection(collection_name)

# 创建新集合
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=1536,  # OpenAI embedding维度
        distance=Distance.COSINE  # 余弦相似度
    )
)
print(f"✅ 集合 '{collection_name}' 创建成功")

# 2. 插入数据
documents = [
    {"id": str(uuid.uuid4()), "text": "Python由Guido van Rossum于1991年创建", "category": "历史"},
    {"id": str(uuid.uuid4()), "text": "Python的设计哲学强调代码可读性", "category": "设计"},
    {"id": str(uuid.uuid4()), "text": "Python广泛用于数据分析、AI、Web开发", "category": "应用"},
]

# 生成向量（实际用Embedding模型）
from openai import OpenAI
openai_client = OpenAI()

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 批量插入
points = []
for doc in documents:
    embedding = get_embedding(doc["text"])
    points.append(PointStruct(
        id=doc["id"],
        vector=embedding,
        payload={  # 元数据
            "text": doc["text"],
            "category": doc["category"]
        }
    ))

client.upsert(collection_name=collection_name, points=points)
print(f"✅ 插入 {len(points)} 条数据")

# 3. 检索
query = "Python是谁发明的？"
query_vector = get_embedding(query)

results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3,
    with_payload=True  # 返回完整数据
)

print(f"\n🔍 查询: {query}")
print("=" * 50)
for i, result in enumerate(results, 1):
    print(f"{i}. [相似度: {result.score:.4f}]")
    print(f"   内容: {result.payload['text']}")
    print(f"   分类: {result.payload['category']}")
    print()

# 4. 条件过滤检索（类似SQL的WHERE）
print("🎯 只搜索'历史'分类的文档:")
filtered_results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "历史"}}
        ]
    },
    limit=3
)
for result in filtered_results:
    print(f"   - {result.payload['text']} (相似度: {result.score:.4f})")

# 5. 删除数据
# client.delete(collection_name=collection_name, points_selector=["id1", "id2"])

# 6. 查看集合信息
collection_info = client.get_collection(collection_name)
print(f"\n📊 集合统计:")
print(f"   文档数: {collection_info.points_count}")
print(f"   向量维度: {collection_info.config.params.vectors.size}")
```

### 2.3 混合检索实现（向量 + 关键词）

为什么需要混合检索？
- **向量检索**：擅长语义匹配（"Python性能" → "Python执行速度"）
- **关键词检索**：擅长精确匹配（"Python 3.11" → 必须包含"3.11"）

```python
from qdrant_client.models import Fusion, FusionQuery

def hybrid_search(query, top_k=5):
    """混合检索：向量 + 关键词"""
    
    # 生成查询向量
    query_vector = get_embedding(query)
    
    # 执行混合检索
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        # RRF (Reciprocal Rank Fusion) 融合算法
        with_vectors=False
    )
    
    return results

# 更简单的实现：分别检索，然后融合
def simple_hybrid_search(query, top_k=5):
    """简单混合检索实现"""
    
    # 1. 向量检索
    vector_results = client.search(
        collection_name=collection_name,
        query_vector=get_embedding(query),
        limit=top_k * 2
    )
    
    # 2. 关键词检索（Qdrant支持全文搜索）
    keyword_results = client.search(
        collection_name=collection_name,
        query_vector=get_embedding(query),  # 仍需向量，但用filter过滤
        query_filter={
            "must": [
                {"key": "text", "match": {"text": query}}
            ]
        },
        limit=top_k * 2
    )
    
    # 3. RRF融合
    return reciprocal_rank_fusion(vector_results, keyword_results, top_k)

def reciprocal_rank_fusion(vec_results, key_results, top_k):
    """RRF融合算法"""
    k = 60  # 常数，防止排名靠后的得分过低
    scores = {}
    
    # 向量检索结果打分
    for rank, result in enumerate(vec_results):
        doc_id = result.id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # 关键词检索结果打分
    for rank, result in enumerate(key_results):
        doc_id = result.id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # 按分数排序
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
```

---

## 第三章：生产级RAG系统实现（实战篇）

### 3.1 完整项目结构

```
rag-system/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI入口
│   ├── config.py            # 配置管理
│   ├── models.py            # 数据模型
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py   # 文档处理
│   │   ├── embedding_service.py    # 嵌入服务
│   │   ├── retrieval_service.py    # 检索服务
│   │   └── rag_service.py          # RAG核心
│   └── api/
│       ├── __init__.py
│       └── v1/
│           ├── documents.py        # 文档上传API
│           └── chat.py             # 对话API
├── data/
│   └── uploads/             # 上传的文档
├── requirements.txt
└── docker-compose.yml
```

### 3.2 文档处理服务（实现代码）

```python
# app/services/document_processor.py
from pathlib import Path
from typing import List
import re

class DocumentProcessor:
    """文档处理服务 - 类似Java的DocumentService"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, file_path: Path) -> List[str]:
        """处理文件，返回文本块列表"""
        # 1. 读取文件
        text = self._read_file(file_path)
        
        # 2. 清洗文本
        text = self._clean_text(text)
        
        # 3. 分块
        chunks = self._chunk_text(text)
        
        return chunks
    
    def _read_file(self, file_path: Path) -> str:
        """读取不同格式文件"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return file_path.read_text(encoding='utf-8')
        
        elif suffix == '.pdf':
            # 使用PyPDF2或pymupdf
            try:
                import fitz  # pymupdf
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
            except ImportError:
                raise ImportError("请安装pymupdf: pip install pymupdf")
        
        elif suffix == '.md':
            return file_path.read_text(encoding='utf-8')
        
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        return text.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        递归字符文本分割
        策略：优先按段落分割，段落太长则按句子，句子太长则按字符
        """
        chunks = []
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
            else:
                # 段落太长，按句子分割
                sentences = re.split(r'(?<=[。！？.!?])', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks

# 使用示例
processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
chunks = processor.process_file(Path("data/document.pdf"))
print(f"分成了 {len(chunks)} 个块")
```

### 3.3 RAG服务完整实现

```python
# app/services/rag_service.py
from typing import List, Dict
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
import numpy as np

class RAGService:
    """RAG核心服务"""
    
    def __init__(self):
        self.llm = AsyncOpenAI()
        self.vector_db = QdrantClient(url="http://localhost:6333")
        self.collection_name = "documents"
    
    async def add_document(self, text: str, metadata: Dict = None):
        """添加文档到知识库"""
        # 1. 生成向量
        embedding = await self._get_embedding(text)
        
        # 2. 存入向量数据库
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": text,
                **(metadata or {})
            }
        )
        
        self.vector_db.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
    
    async def query(self, question: str, top_k: int = 5) -> Dict:
        """RAG查询"""
        # 1. 检索相关文档
        contexts = await self._retrieve(question, top_k)
        
        # 2. 构建Prompt
        prompt = self._build_prompt(question, contexts)
        
        # 3. 生成答案
        answer = await self._generate(prompt)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "sources": [ctx["text"] for ctx in contexts]
        }
    
    async def _retrieve(self, query: str, top_k: int) -> List[Dict]:
        """检索相关文档"""
        # 生成查询向量
        query_vector = await self._get_embedding(query)
        
        # 向量检索
        results = self.vector_db.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        return [
            {
                "text": result.payload["text"],
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            }
            for result in results
        ]
    
    def _build_prompt(self, question: str, contexts: List[Dict]) -> str:
        """构建Prompt"""
        context_text = "\n\n".join([
            f"[相关度: {ctx['score']:.2f}] {ctx['text']}"
            for ctx in contexts
        ])
        
        return f"""你是一个专业的问答助手。请基于以下参考资料回答问题。

参考资料：
{context_text}

问题：{question}

要求：
1. 基于参考资料回答，不要编造信息
2. 如果参考资料中没有答案，明确说明"根据现有资料无法回答"
3. 回答要简洁准确

回答："""
    
    async def _generate(self, prompt: str) -> str:
        """调用LLM生成答案"""
        response = await self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本向量"""
        response = await self.llm.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

# 使用示例
async def main():
    rag = RAGService()
    
    # 添加文档
    await rag.add_document(
        "Python是一种解释型、面向对象、动态数据类型的高级程序设计语言",
        metadata={"source": "Python官方文档", "category": "介绍"}
    )
    
    # 查询
    result = await rag.query("Python是什么类型的语言？")
    print(f"答案：{result['answer']}")
    print(f"参考：{result['sources']}")

# 运行
import asyncio
asyncio.run(main())
```

### 3.4 FastAPI接口实现

```python
# app/api/v1/chat.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from app.services.rag_service import RAGService
from app.services.document_processor import DocumentProcessor

router = APIRouter(prefix="/api/v1")
rag_service = RAGService()
doc_processor = DocumentProcessor()

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG问答接口"""
    try:
        result = await rag_service.query(
            question=request.question,
            top_k=request.top_k
        )
        
        # 计算置信度（基于检索分数）
        avg_score = sum(ctx["score"] for ctx in result["contexts"]) / len(result["contexts"])
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=avg_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """文档上传接口"""
    try:
        # 保存文件
        file_path = f"data/uploads/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 处理文档
        chunks = doc_processor.process_file(file_path)
        
        # 添加到知识库
        for chunk in chunks:
            await rag_service.add_document(
                text=chunk,
                metadata={"source": file.filename}
            )
        
        return {
            "message": "文档上传成功",
            "filename": file.filename,
            "chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 第四章：常见问题排查（FAQ）

### Q1: 检索不到相关文档？

**可能原因**：
1. **分块太大**：500字的文档块，查询只有5个字，匹配度低
   - **解决**：减小chunk_size到200-300

2. **Embedding模型不对**：用的模型和领域不匹配
   - **解决**：专业领域用专用Embedding（如BGE-M3）

3. **查询太短**：用户只输入"Python"
   - **解决**：使用查询重写扩展查询

### Q2: LLM回答幻觉严重？

**可能原因**：
1. **Prompt不够明确**：没有明确约束LLM只能基于参考资料
   - **解决**：Prompt里加"如果资料中没有，请明确说明"

2. **检索结果太少**：只给LLM 1个参考文档
   - **解决**：增加top_k到5-10

3. **检索结果不相关**：给了LLM不相关的文档
   - **解决**：设置相似度阈值，低于0.7的丢弃

### Q3: 响应速度太慢？

**优化方案**：
```python
# 1. 缓存Embedding（避免重复计算）
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding_cached(text: str):
    return get_embedding(text)

# 2. 异步并发
async def batch_query(questions: List[str]):
    tasks = [rag_service.query(q) for q in questions]
    return await asyncio.gather(*tasks)

# 3. 向量检索优化（使用HNSW索引）
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
        hnsw_config={  # HNSW索引，检索更快
            "m": 16,
            "ef_construct": 100
        }
    )
)
```

---

## 第五章：实战练习

### 练习1：搭建最简RAG（30分钟）

目标：实现一个能回答Python相关问题的RAG系统

步骤：
1. 准备5条Python知识
2. 存入Chroma
3. 实现查询功能
4. 测试3个问题

### 练习2：改进检索质量（1小时）

目标：实现查询重写 + 重排序

步骤：
1. 添加查询重写功能
2. 实现两阶段检索
3. 对比改进前后的效果

### 练习3：完整项目（2小时）

目标：实现支持PDF上传的知识库系统

步骤：
1. 搭建FastAPI服务
2. 实现PDF解析
3. 实现文档上传API
4. 实现问答API
5. 添加前端界面（Streamlit）

---

## 总结

读完本文，你应该掌握：
- ✅ RAG的核心原理和三种发展阶段
- ✅ 向量数据库的选型和使用
- ✅ 混合检索、查询重写、重排序等高级技术
- ✅ 完整的生产级RAG系统实现
- ✅ 常见问题的排查方法

**下一步**：继续学习Agent技术（ReAct框架）

---

**本文整理自**: RAG Survey论文 + 生产实践 + 作者实战经验  
**整理时间**: 2026-03-04  
**更新频率**: 持续更新优化
