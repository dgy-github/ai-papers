# 多Agent系统架构实战：从零搭建智能团队协作

> 基于多智能体论文和生产实践深度整理
> 完整可运行代码 + 企业级架构设计

---

## 第一章：为什么需要多Agent？

### 1.1 单Agent的局限

单Agent的问题：
- 复杂任务需要多个领域知识
- 单点故障风险
- 处理时间长
- 难以维护

**类比**：
- 单Agent = 全栈工程师做所有事
- 多Agent = 前端、后端、测试分工协作

### 1.2 多Agent的优势

| 优势 | 说明 |
|------|------|
| **专业化** | 每个Agent专注一个领域 |
| **并行化** | 多个任务同时处理 |
| **容错性** | 单个失败不影响整体 |
| **可扩展** | 随时添加新Agent |

---

## 第二章：核心架构模式

### 2.1 Supervisor-Worker模式

**架构图**：
```
          Supervisor (协调者)
               │
    ┌──────────┼──────────┐
    │          │          │
 Worker A   Worker B   Worker C
```

**Python实现**：
```python
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Task:
    """任务定义"""
    id: str
    type: str
    data: Dict[str, Any]
    priority: int = 1

@dataclass
class Agent:
    """Agent定义"""
    name: str
    capabilities: List[str]
    
    async def execute(self, task: Task) -> Dict:
        """执行任务"""
        print(f"[{self.name}] 执行任务: {task.id}")
        await asyncio.sleep(1)  # 模拟执行
        return {"status": "success", "agent": self.name}

class Supervisor:
    """Supervisor - 任务分配器"""
    
    def __init__(self):
        self.workers: Dict[str, Agent] = {}
        self.task_history: List[Dict] = []
    
    def register_worker(self, agent: Agent):
        """注册Worker"""
        self.workers[agent.name] = agent
        print(f"✅ 注册Worker: {agent.name}")
    
    def assign_task(self, task: Task) -> Agent:
        """分配任务 - 基于能力匹配"""
        for name, agent in self.workers.items():
            if task.type in agent.capabilities:
                return agent
        raise ValueError(f"没有Agent能处理任务类型: {task.type}")
    
    async def process_tasks(self, tasks: List[Task]) -> List[Dict]:
        """批量处理任务"""
        # 分组：哪些可以并行
        assignments = {}
        for task in tasks:
            agent = self.assign_task(task)
            if agent.name not in assignments:
                assignments[agent.name] = []
            assignments[agent.name].append(task)
        
        # 并行执行
        results = []
        for agent_name, agent_tasks in assignments.items():
            agent = self.workers[agent_name]
            # 同一Agent的任务串行，不同Agent并行
            for task in agent_tasks:
                result = await agent.execute(task)
                results.append(result)
        
        return results

# 使用示例
async def main():
    supervisor = Supervisor()
    
    # 注册Worker
    supervisor.register_worker(Agent(
        name="Researcher",
        capabilities=["research", "search"]
    ))
    supervisor.register_worker(Agent(
        name="Writer",
        capabilities=["write", "edit"]
    ))
    supervisor.register_worker(Agent(
        name="Reviewer",
        capabilities=["review", "check"]
    ))
    
    # 创建任务
    tasks = [
        Task(id="t1", type="research", data={"topic": "AI"}),
        Task(id="t2", type="write", data={"topic": "AI"}),
        Task(id="t3", type="review", data={"doc": "draft"})
    ]
    
    # 处理
    results = await supervisor.process_tasks(tasks)
    print(f"\n✅ 完成 {len(results)} 个任务")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.2 消息总线模式

```python
import asyncio
from typing import Callable, Dict, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Message:
    """消息定义"""
    topic: str
    payload: Dict
    sender: str
    timestamp: datetime = field(default_factory=datetime.now)

class MessageBus:
    """消息总线 - 类似Kafka"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue = asyncio.Queue()
        self.running = False
    
    def subscribe(self, topic: str, callback: Callable):
        """订阅主题"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        print(f"📩 {callback.__name__} 订阅了 {topic}")
    
    async def publish(self, topic: str, payload: Dict, sender: str):
        """发布消息"""
        msg = Message(topic=topic, payload=payload, sender=sender)
        await self.message_queue.put(msg)
    
    async def run(self):
        """启动消息循环"""
        self.running = True
        while self.running:
            msg = await self.message_queue.get()
            
            if msg.topic in self.subscribers:
                for callback in self.subscribers[msg.topic]:
                    asyncio.create_task(callback(msg))
    
    async def stop(self):
        self.running = False

# Agent使用消息总线
class BusAgent:
    """基于消息总线的Agent"""
    
    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus = bus
    
    async def handle_research(self, msg: Message):
        """处理研究任务"""
        print(f"[{self.name}] 收到研究任务: {msg.payload}")
        # 执行研究...
        
        # 发布结果
        await self.bus.publish(
            topic="research.completed",
            payload={"result": "研究完成"},
            sender=self.name
        )

# 使用示例
async def demo_bus():
    bus = MessageBus()
    
    # 创建Agent
    researcher = BusAgent("Researcher", bus)
    writer = BusAgent("Writer", bus)
    
    # 订阅消息
    bus.subscribe("research.task", researcher.handle_research)
    bus.subscribe("research.completed", 
                  lambda m: print(f"Writer收到研究结果: {m.payload}"))
    
    # 启动消息总线
    asyncio.create_task(bus.run())
    
    # 发布任务
    await bus.publish("research.task", {"topic": "AI"}, "User")
    
    await asyncio.sleep(2)
    await bus.stop()
```

---

## 第三章：完整实战 - 智能投研系统

```python
# investment_system.py - 智能投研系统
import asyncio
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Company:
    name: str
    ticker: str
    sector: str

@dataclass
class Report:
    company: str
    analysis: str
    risk_level: str
    recommendation: str
    created_at: datetime

# ========== Agent定义 ==========
class DataCollectionAgent:
    """数据收集Agent"""
    
    async def collect(self, company: Company) -> Dict:
        """收集公司数据"""
        print(f"📊 [{self.__class__.__name__}] 收集 {company.name} 数据")
        await asyncio.sleep(1)
        
        return {
            "company": company,
            "financials": {"revenue": 100, "profit": 20},
            "news": ["Q3财报超预期", "新产品发布"]
        }

class AnalysisAgent:
    """分析Agent"""
    
    async def analyze(self, data: Dict) -> Dict:
        """财务分析"""
        print(f"📈 [{self.__class__.__name__}] 分析 {data['company'].name}")
        await asyncio.sleep(1)
        
        financials = data["financials"]
        pe_ratio = financials["revenue"] / financials["profit"]
        
        return {
            "company": data["company"],
            "pe_ratio": pe_ratio,
            "growth_potential": "高" if pe_ratio < 10 else "中",
            "analysis": f"市盈率{pe_ratio:.1f}，增长潜力{'高' if pe_ratio < 10 else '中'}"
        }

class ReportGenerationAgent:
    """报告生成Agent"""
    
    async def generate(self, analysis: Dict) -> Report:
        """生成研报"""
        print(f"📄 [{self.__class__.__name__}] 生成 {analysis['company'].name} 研报")
        await asyncio.sleep(1)
        
        return Report(
            company=analysis["company"].name,
            analysis=analysis["analysis"],
            risk_level="中",
            recommendation="买入" if analysis["growth_potential"] == "高" else "持有",
            created_at=datetime.now()
        )

class RiskAssessmentAgent:
    """风险评估Agent"""
    
    async def assess(self, report: Report) -> Report:
        """风险评估"""
        print(f"⚠️ [{self.__class__.__name__}] 评估 {report.company} 风险")
        await asyncio.sleep(0.5)
        
        # 实际应调用风险模型
        report.risk_level = "中低风险"
        return report

# ========== 系统编排 ==========
class InvestmentResearchSystem:
    """智能投研系统"""
    
    def __init__(self):
        self.data_collector = DataCollectionAgent()
        self.analyst = AnalysisAgent()
        self.report_generator = ReportGenerationAgent()
        self.risk_assessor = RiskAssessmentAgent()
    
    async def research_company(self, company: Company) -> Report:
        """研究单个公司"""
        # 顺序执行：数据收集 -> 分析 -> 报告生成 -> 风险评估
        data = await self.data_collector.collect(company)
        analysis = await self.analyst.analyze(data)
        report = await self.report_generator.generate(analysis)
        final_report = await self.risk_assessor.assess(report)
        
        return final_report
    
    async def batch_research(self, companies: List[Company]) -> List[Report]:
        """批量研究（并行）"""
        tasks = [self.research_company(c) for c in companies]
        return await asyncio.gather(*tasks)

# ========== 使用 ==========
async def main():
    system = InvestmentResearchSystem()
    
    companies = [
        Company("Tech Corp", "TECH", "科技"),
        Company("Finance Inc", "FIN", "金融"),
        Company("Energy Ltd", "ENG", "能源")
    ]
    
    print("🚀 启动智能投研系统\n")
    
    # 批量研究
    reports = await system.batch_research(companies)
    
    print("\n📋 研究报告汇总:")
    print("=" * 60)
    for report in reports:
        print(f"\n公司: {report.company}")
        print(f"分析: {report.analysis}")
        print(f"风险: {report.risk_level}")
        print(f"建议: {report.recommendation}")
        print(f"时间: {report.created_at}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 第四章：生产级多Agent系统

### 4.1 错误处理与恢复

```python
class RobustMultiAgentSystem:
    """健壮的多Agent系统"""
    
    async def execute_with_retry(self, agent, task, max_retries=3):
        """带重试的执行"""
        for attempt in range(max_retries):
            try:
                return await agent.execute(task)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"⚠️ 第{attempt+1}次失败，重试: {e}")
                await asyncio.sleep(2 ** attempt)  # 指数退避
    
    async def execute_parallel_with_timeout(self, agents, tasks, timeout=30):
        """并行执行带超时"""
        async def execute(agent, task):
            try:
                return await asyncio.wait_for(
                    agent.execute(task),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return {"error": "执行超时", "agent": agent.name}
        
        tasks_list = [execute(a, t) for a, t in zip(agents, tasks)]
        return await asyncio.gather(*tasks_list, return_exceptions=True)
```

### 4.2 状态管理

```python
from typing import TypedDict

class MultiAgentState(TypedDict):
    """多Agent共享状态"""
    tasks: List[Task]
    results: Dict[str, Any]
    errors: List[Dict]
    metadata: Dict
    
class StateManager:
    """状态管理器"""
    
    def __init__(self):
        self.states: Dict[str, MultiAgentState] = {}
    
    def create_session(self, session_id: str):
        self.states[session_id] = {
            "tasks": [],
            "results": {},
            "errors": [],
            "metadata": {}
        }
    
    def update_state(self, session_id: str, update: Dict):
        if session_id in self.states:
            self.states[session_id].update(update)
```

---

## 实战练习

1. **实现任务队列系统**（30分钟）
   - 使用Redis做任务队列
   - Worker从队列取任务执行
   - 支持任务优先级

2. **实现Agent健康检查**（1小时）
   - 定期检查Agent状态
   - 故障自动重启
   - 服务发现机制

3. **完整智能客服系统**（2小时）
   - 意图识别Agent
   - 订单查询Agent
   - 技术支持Agent
   - Supervisor分配任务

---

**本文整理时间**: 2026-03-04
