# LangGraph工作流实战：从零实现复杂Agent编排

> 基于LangGraph官方文档和生产实践深度整理
> 不需要看其他书，只读这一篇就能动手实现工作流

---

## 第一章：为什么需要LangGraph？

### 1.1 简单Agent的局限

用ReAct实现的Agent：
```
用户提问 → Thought → Action → Observation → Final Answer
```

**问题**：
- 只能线性执行，不能分支判断
- 不能循环（失败后重试）
- 不能并行执行多个任务
- 状态管理混乱

### 1.2 LangGraph是什么？

**LangGraph = 状态机 + 图计算 + Agent编排**

核心概念：
- **State（状态）**：整个工作流的共享数据
- **Node（节点）**：执行具体任务的函数
- **Edge（边）**：节点之间的流转关系
- **Graph（图）**：节点和边的组合

**类比理解**：
- Java的Spring State Machine
- Camunda BPMN工作流引擎
- Airflow DAG

### 1.3 LangGraph vs 其他方案

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **原生ReAct** | 简单 | 不能处理复杂流程 | 简单问答 |
| **LangGraph** | 灵活、可循环、可分支 | 学习成本 | 复杂工作流 |
| **AutoGen** | 多Agent对话 | 不易控制 | 多Agent协作 |
| ** CrewAI** | 高层抽象 | 不够灵活 | 快速原型 |

---

## 第二章：核心概念详解

### 2.1 State（状态）

**作用**：在整个工作流中共享数据

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# 定义状态类型
class AgentState(TypedDict):
    """工作流状态 - 类似Java的DTO"""
    messages: Annotated[list, add_messages]  # 对话历史
    query: str                               # 用户查询
    documents: list                          # 检索到的文档
    answer: str                              # 最终答案
    iteration_count: int                     # 迭代次数
    
# 解释：
# TypedDict = 类似Java的Map<String, Object>
# Annotated = 添加元数据，add_messages表示合并消息
```

**对应Java概念**：
```java
// Java版本
public class AgentState {
    private List<Message> messages;
    private String query;
    private List<Document> documents;
    private String answer;
    private int iterationCount;
    // getter/setter...
}
```

### 2.2 Node（节点）

**作用**：执行具体任务的函数

```python
from langgraph.graph import StateGraph

# 节点1：检索
def retrieve(state: AgentState) -> AgentState:
    """检索节点 - 类似Java的Service方法"""
    query = state["query"]
    # 执行检索...
    documents = search(query)
    return {
        **state,
        "documents": documents
    }

# 节点2：生成
async def generate(state: AgentState) -> AgentState:
    """生成节点 - 可以是异步的"""
    documents = state["documents"]
    # 调用LLM生成...
    answer = await llm.generate(documents)
    return {
        **state,
        "answer": answer
    }

# 节点3：检查
def check_answer(state: AgentState) -> str:
    """条件节点 - 返回下一个节点名称"""
    if len(state["answer"]) < 10:
        return "regenerate"  # 太短，重新生成
    return "end"  # 完成
```

### 2.3 Edge（边）

**作用**：定义节点之间的流转

```python
# 普通边：顺序执行
builder.add_edge("retrieve", "generate")

# 条件边：根据返回值决定下一步
builder.add_conditional_edges(
    "check_answer",      # 源节点
    check_answer,        # 条件函数
    {
        "regenerate": "generate",  # 返回"regenerate" -> 回到generate
        "end": END                 # 返回"end" -> 结束
    }
)

# 循环边：实现重试逻辑
builder.add_edge("generate", "check_answer")
```

### 2.4 完整图构建

```python
from langgraph.graph import StateGraph, END

# 创建工作流构建器
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("check", check_answer)

# 添加边
builder.set_entry_point("retrieve")  # 入口
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "check")
builder.add_conditional_edges(
    "check",
    lambda state: "ok" if len(state["answer"]) > 10 else "retry",
    {"ok": END, "retry": "generate"}
)

# 编译
graph = builder.compile()
```

---

## 第三章：完整实战 - 研报生成工作流

### 3.1 场景描述

**任务**：自动生成行业研报
```
用户输入主题 → 研究 → 写作 → 审核 → 发布/修改
```

**流程图**：
```
[开始] → [Research Agent] → [Writer Agent] → [Reviewer Agent]
                              ↓                    ↓
                         [质量检查]          [人工审批]
                              ↓                    ↓
                         合格→[发布]         通过→[发布]
                         不合格→[重写]        不通过→[修改]
```

### 3.2 完整代码实现

```python
# research_workflow.py - 研报生成工作流
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import asyncio

# ========== 1. 定义状态 ==========
class ReportState(TypedDict):
    topic: str                           # 研报主题
    research_notes: str                  # 研究笔记
    draft: str                           # 草稿
    review_comments: str                 # 审核意见
    final_report: str                    # 最终报告
    quality_score: float                 # 质量评分
    iteration_count: int                 # 迭代次数
    requires_human_approval: bool        # 是否需要人工审批

# ========== 2. 定义节点 ==========
async def research_node(state: ReportState) -> ReportState:
    """研究节点：收集信息"""
    print(f"🔍 正在研究主题: {state['topic']}")
    
    # 模拟研究过程（实际调用搜索工具）
    await asyncio.sleep(1)
    
    notes = f"""
主题: {state['topic']}

研究要点:
1. 市场规模：2024年达到XXX亿元
2. 主要玩家：公司A、公司B、公司C
3. 发展趋势：AI驱动、政策支持
4. 风险提示：竞争激烈、技术迭代快
"""
    
    return {
        **state,
        "research_notes": notes
    }

async def writing_node(state: ReportState) -> ReportState:
    """写作节点：生成研报草稿"""
    print(f"✍️ 正在撰写研报...")
    
    notes = state["research_notes"]
    
    # 模拟写作（实际调用LLM）
    await asyncio.sleep(1)
    
    draft = f"""
# {state['topic']}行业研究报告

## 执行摘要
本文分析了{state['topic']}的市场现状和发展趋势...

## 市场分析
{notes}

## 投资建议
建议关注头部企业，谨慎评估风险。

（以上为示例内容，实际应调用LLM生成）
"""
    
    return {
        **state,
        "draft": draft,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

async def review_node(state: ReportState) -> ReportState:
    """审核节点：评估研报质量"""
    print(f"📋 正在审核研报...")
    
    draft = state["draft"]
    
    # 模拟审核（实际调用LLM评估）
    await asyncio.sleep(0.5)
    
    # 简单评分逻辑
    score = min(0.9, 0.5 + len(draft) / 2000)  # 越长分越高（示例）
    
    comments = f"质量评分: {score:.2f}\n"
    if score < 0.7:
        comments += "需要补充更多数据和分析"
    else:
        comments += "内容完整，建议发布"
    
    return {
        **state,
        "quality_score": score,
        "review_comments": comments,
        "requires_human_approval": score >= 0.8  # 高分需要人工确认
    }

def quality_check(state: ReportState) -> Literal["rewrite", "approve", "human_review"]:
    """质量检查：决定下一步"""
    score = state["quality_score"]
    iteration = state.get("iteration_count", 0)
    
    if score < 0.6 and iteration < 3:
        print(f"⚠️ 质量不足({score:.2f})，需要重写")
        return "rewrite"
    elif state["requires_human_approval"]:
        print(f"⏸️ 质量优秀({score:.2f})，等待人工审批")
        return "human_review"
    else:
        print(f"✅ 审核通过({score:.2f})")
        return "approve"

async def human_approval_node(state: ReportState) -> ReportState:
    """人工审批节点"""
    print(f"\n{'='*50}")
    print(f"📄 研报草稿:\n{state['draft']}")
    print(f"\n💬 审核意见: {state['review_comments']}")
    print(f"{'='*50}\n")
    
    # 实际生产环境：发送到审批系统，等待人工确认
    # 这里模拟自动通过
    print("模拟：人工审批通过")
    
    return {
        **state,
        "final_report": state["draft"]
    }

def final_publish(state: ReportState) -> ReportState:
    """发布节点"""
    print(f"📢 研报已发布！")
    print(f"主题: {state['topic']}")
    print(f"质量分: {state['quality_score']:.2f}")
    
    return {
        **state,
        "final_report": state["draft"]
    }

# ========== 3. 构建工作流 ==========
def create_workflow():
    """创建研报生成工作流"""
    builder = StateGraph(ReportState)
    
    # 添加节点
    builder.add_node("research", research_node)
    builder.add_node("writing", writing_node)
    builder.add_node("review", review_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("publish", final_publish)
    
    # 添加边
    builder.set_entry_point("research")
    builder.add_edge("research", "writing")
    builder.add_edge("writing", "review")
    
    # 条件边：根据质量决定下一步
    builder.add_conditional_edges(
        "review",
        quality_check,
        {
            "rewrite": "writing",       # 重写
            "human_review": "human_approval",  # 人工审批
            "approve": "publish"        # 直接发布
        }
    )
    
    builder.add_edge("human_approval", "publish")
    builder.add_edge("publish", END)
    
    return builder.compile()

# ========== 4. 运行 ==========
async def main():
    # 创建工作流
    workflow = create_workflow()
    
    # 初始状态
    initial_state = {
        "topic": "人工智能在医疗领域的应用",
        "research_notes": "",
        "draft": "",
        "review_comments": "",
        "final_report": "",
        "quality_score": 0.0,
        "iteration_count": 0,
        "requires_human_approval": False
    }
    
    # 执行工作流
    print("🚀 启动研报生成工作流\n")
    result = await workflow.ainvoke(initial_state)
    
    print(f"\n✅ 工作流完成！")
    print(f"最终报告:\n{result['final_report'][:500]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3.3 代码解析

**状态流转图**：
```
research → writing → review
              ↑      ↓
              └──────┘ (rewrite)
              
review → human_approval → publish → END
  ↓
approve → publish → END
```

**关键特性**：
1. **循环支持**：质量不好可以回到writing重写
2. **条件分支**：根据质量分数走不同路径
3. **人工介入**：human_approval节点可以暂停等人
4. **状态持久**：每次节点的输出自动合并到state

---

## 第四章：高级特性

### 4.1 状态持久化（Checkpointer）

**作用**：工作流可以暂停、恢复、重试

```python
from langgraph.checkpoint import MemorySaver

# 内存检查点（开发用）
memory = MemorySaver()

# 创建带检查点的工作流
workflow = builder.compile(checkpointer=memory)

# 执行时指定thread_id
config = {"configurable": {"thread_id": "report_001"}}
result = workflow.invoke(initial_state, config)

# 稍后恢复
# workflow.invoke(None, config)  # 从上次状态继续
```

**对应Java概念**：
- Saga模式的事务补偿
- 工作流引擎的状态持久化

### 4.2 流式输出（Streaming）

```python
# 流式执行，实时看到每个节点的输出
async for event in workflow.astream(initial_state):
    print(f"节点: {event['node']}")
    print(f"输出: {event['state']}")
```

### 4.3 与Java工作流引擎对比

| 特性 | Java (Camunda) | LangGraph |
|------|----------------|-----------|
| 定义方式 | BPMN XML | Python代码 |
| 学习曲线 | 高 | 低 |
| 灵活性 | 中 | 高 |
| LLM集成 | 需开发 | 原生支持 |
| 部署方式 | 独立服务 | 库函数 |

---

## 第五章：常见问题

### Q1: 如何调试工作流？

```python
# 查看执行轨迹
from langgraph.graph import draw_ascii

print(draw_ascii(workflow))

# 或可视化（需要graphviz）
workflow.get_graph().draw_png("workflow.png")
```

### Q2: 如何处理错误？

```python
def error_handler(state):
    """错误处理节点"""
    print(f"错误: {state.get('error')}")
    # 发送告警、记录日志等
    return state

builder.add_node("error_handler", error_handler)
# 在可能出错的地方添加错误边
```

### Q3: 如何并行执行？

```python
# 使用Send并行多个任务
from langgraph.constants import Send

def parallel_map(state):
    """将任务分发给多个worker"""
    queries = state["queries"]
    return [Send("worker", {"query": q}) for q in queries]

builder.add_conditional_edges("start", parallel_map, ["worker"])
```

---

## 实战练习

### 练习1：实现带重试的RAG（30分钟）

**要求**：
- 检索失败时重试3次
- 重试间隔指数退避
- 超过3次转人工处理

### 练习2：实现并行研究（1小时）

**要求**：
- 同时从3个数据源收集信息
- 合并结果后写作
- 使用Send实现并行

### 练习3：完整客服工作流（2小时）

**场景**：
```
用户问题 → 意图识别 → [查询订单|退换货|技术支持]
    ↓
各分支处理 → 满意度检查 → 结束/人工接管
```

---

**本文整理自**: LangGraph官方文档 + 生产实践  
**整理时间**: 2026-03-04
