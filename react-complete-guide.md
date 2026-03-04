# ReAct Agent完全指南：从零实现智能助手

> 基于论文《ReAct: Synergizing Reasoning and Acting in Language Models》深度整理
> 不需要看其他书，只读这一篇就能动手实现Agent

---

## 第一章：Agent到底是什么？（概念篇）

### 1.1 一句话理解Agent

**Agent = LLM + 工具 + 记忆 + 规划能力**

普通ChatGPT只能对话，Agent可以：
- 🔧 **使用工具**（搜索、计算、查数据库）
- 🧠 **自主规划**（分解任务、决定下一步做什么）
- 💾 **记住上下文**（多轮对话保持连贯）
- 🔄 **自我纠错**（失败时调整策略）

**类比理解**：
- 普通LLM = 只会背书的学生
- Agent = 会查资料、会思考、会行动的研究员

### 1.2 Agent vs RAG的区别

| 对比项 | RAG | Agent |
|--------|-----|-------|
| **核心能力** | 知识检索 | 任务执行 |
| **交互方式** | 一问一答 | 多轮自主执行 |
| **工具使用** | 无 | 有（搜索、计算等） |
| **适用场景** | 知识库问答 | 复杂任务自动化 |
| **示例** | "公司年假规定是什么？" | "帮我订明天去北京的机票" |

**一句话总结**：RAG让LLM有知识，Agent让LLM有能力。

### 1.3 为什么需要ReAct模式？

**没有ReAct的传统方式**：
```
用户：北京今天天气怎么样？
LLM：我不知道实时天气信息。
```

**有ReAct的方式**：
```
Thought: 用户问天气，我需要查询实时天气数据。
Action: 调用天气API查询北京天气
Observation: 北京今天晴，25°C，微风
Thought: 已获得天气信息，可以回答用户。
Final Answer: 北京今天晴天，25°C，微风，适合出行。
```

**ReAct的核心优势**：
1. **可解释性**：每一步都有思考过程，知道Agent为什么这么做
2. **可控性**：可以人工介入任何一步
3. **容错性**：工具调用失败可以重试或换策略

---

## 第二章：ReAct模式详解（原理篇）

### 2.1 ReAct循环流程

```
        ┌─────────────┐
        │    Start    │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │   Thought   │  ← 思考：现在该做什么？
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │   Action    │  ← 行动：调用工具
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ Observation │  ← 观察：工具返回什么？
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  完成任务？  │──Yes──▶ Final Answer
        └──────┬──────┘
               │ No
               └───────┐
                       ▼
                ┌─────────────┐
                │   Error？    │──Yes──▶ 处理错误
                └──────┬──────┘
                       │ No
                       └───────┐
                               ▼
                        回到 Thought
```

### 2.2 三个核心组件

#### Thought（思考）

**作用**：分析当前情况，决定下一步行动

**示例**：
```
Thought: 用户问的是"Python的创始人是谁"，这是一个事实性问题，
我需要使用搜索工具来查找这个信息。
```

**关键要素**：
- 理解用户意图
- 评估当前进度
- 规划下一步行动
- 处理异常情况

#### Action（行动）

**作用**：执行具体操作，通常是调用工具

**示例**：
```
Action: search[Python创始人]
```

**常见Action类型**：
| Action | 格式 | 示例 |
|--------|------|------|
| 搜索 | `search[query]` | `search[Python创始人]` |
| 计算 | `calculate[expression]` | `calculate[2+2]` |
| 查询数据库 | `query_db[sql]` | `query_db[SELECT * FROM users]` |
| 完成 | `finish[answer]` | `finish[Python创始人是Guido]` |

#### Observation（观察）

**作用**：获取Action的执行结果

**示例**：
```
Observation: Python由Guido van Rossum于1991年创建。
```

**来源**：
- 工具返回结果
- API响应数据
- 数据库查询结果
- 错误信息

---

## 第三章：最简Agent实现（实战篇）

### 3.1 50行代码实现ReAct Agent

```python
# react_agent_minimal.py - 最简ReAct Agent实现
import re
from openai import OpenAI

client = OpenAI()

# ========== 1. 定义工具 ==========
def search(query: str) -> str:
    """模拟搜索工具"""
    knowledge_base = {
        "Python创始人": "Python由Guido van Rossum于1991年创建",
        "Java创始人": "Java由James Gosling于1995年在Sun Microsystems创建",
        "Python最新版本": "Python 3.12于2023年10月发布",
    }
    return knowledge_base.get(query, "未找到相关信息")

def calculate(expression: str) -> str:
    """计算工具"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "计算错误"

# 工具注册表
TOOLS = {
    "search": search,
    "calculate": calculate
}

# ========== 2. 定义Prompt模板 ==========
REACT_PROMPT = """你是一个智能助手，可以使用工具解决问题。

可用工具：
- search[query]: 搜索信息
- calculate[expression]: 计算数学表达式

请按照以下格式回答：
Thought: 思考你需要做什么
Action: 工具名称[参数]
Observation: 工具返回的结果
... (可以重复Thought/Action/Observation)
Final Answer: 最终答案

开始！

问题: {question}
"""

# ========== 3. ReAct Agent核心 ==========
class ReActAgent:
    def __init__(self, max_iterations=5):
        self.max_iterations = max_iterations
    
    def run(self, question: str) -> str:
        # 构建对话历史
        history = REACT_PROMPT.format(question=question)
        
        for i in range(self.max_iterations):
            print(f"\n=== 第{i+1}轮 ===")
            print(f"当前历史:\n{history}\n")
            
            # 调用LLM
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": history}],
                stop=["Observation:"]  # 在Observation前停止，等待工具返回
            )
            
            content = response.choices[0].message.content
            print(f"LLM输出:\n{content}\n")
            
            # 检查是否完成
            if "Final Answer:" in content:
                return content.split("Final Answer:")[-1].strip()
            
            # 解析Thought和Action
            thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", content, re.DOTALL)
            action_match = re.search(r"Action:\s*(\w+)\[(.+?)\]", content)
            
            if not action_match:
                print("未找到Action，结束")
                return "无法完成任务"
            
            thought = thought_match.group(1).strip() if thought_match else ""
            tool_name = action_match.group(1)
            tool_input = action_match.group(2)
            
            print(f"🤔 Thought: {thought}")
            print(f"🔧 Action: {tool_name}[{tool_input}]")
            
            # 执行工具
            if tool_name in TOOLS:
                observation = TOOLS[tool_name](tool_input)
            else:
                observation = f"错误: 未知工具 '{tool_name}'"
            
            print(f"👀 Observation: {observation}")
            
            # 更新历史
            history += f"\n{content}\nObservation: {observation}"
        
        return "达到最大迭代次数，未完成"

# ========== 4. 测试 ==========
if __name__ == "__main__":
    agent = ReActAgent()
    
    # 测试1: 搜索问题
    print("=" * 50)
    print("测试1: Python创始人是谁？")
    result = agent.run("Python的创始人是谁？")
    print(f"\n✅ 最终答案: {result}")
    
    # 测试2: 计算问题
    print("\n" + "=" * 50)
    print("测试2: 25乘以4等于多少？")
    result = agent.run("25乘以4等于多少？")
    print(f"\n✅ 最终答案: {result}")
```

**运行结果**：
```
==================================================
测试1: Python创始人是谁？

=== 第1轮 ===
🤔 Thought: 用户询问Python的创始人，这是一个事实性问题，我需要搜索相关信息
🔧 Action: search[Python创始人]
👀 Observation: Python由Guido van Rossum于1991年创建

=== 第2轮 ===
✅ 最终答案: Python的创始人是Guido van Rossum，他在1991年创建了Python语言。
```

### 3.2 代码逐行解析

**第1-20行：工具定义**
- 定义了两个工具：`search` 和 `calculate`
- 注册到 `TOOLS` 字典，方便动态调用

**第23-38行：Prompt模板**
- 告诉LLM有哪些工具可用
- 规定了输出格式（Thought/Action/Observation）
- `{question}` 是占位符，会被替换为实际问题

**第41-85行：Agent核心类**
- `max_iterations`: 防止无限循环
- `run()`: 主循环，交替执行Thought和Action
- `re.DOTALL`: 让 `.` 匹配换行符
- `stop=["Observation:"]`: 控制LLM在Observation前停止，等待工具返回

**第88-95行：测试**
- 创建Agent实例
- 运行两个测试用例

---

## 第四章：生产级Agent封装（进阶篇）

### 4.1 完整Agent类实现

```python
# react_agent_production.py - 生产级Agent实现
import asyncio
import json
import re
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from enum import Enum

# ========== 数据模型 ==========
class ActionType(Enum):
    SEARCH = "search"
    CALCULATE = "calculate"
    QUERY_DB = "query_db"
    API_CALL = "api_call"
    FINISH = "finish"

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    async def execute(self, **kwargs) -> str:
        """执行工具"""
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

@dataclass
class AgentStep:
    """执行步骤记录"""
    step_number: int
    thought: str
    action: str
    action_input: Dict
    observation: str
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

# ========== 工具实现 ==========
async def async_search(query: str) -> str:
    """异步搜索工具"""
    # 实际实现调用搜索API
    await asyncio.sleep(0.5)  # 模拟网络延迟
    return f"搜索结果: {query}的相关信息..."

def calculate(expression: str) -> float:
    """计算工具"""
    # 安全计算：只允许基本运算符
    allowed_chars = set('0123456789+-*/.() ')
    if not all(c in allowed_chars for c in expression):
        raise ValueError("表达式包含非法字符")
    return eval(expression)

async def query_database(sql: str) -> List[Dict]:
    """数据库查询工具"""
    # 实际实现连接数据库
    await asyncio.sleep(0.3)
    return [{"id": 1, "name": "示例数据"}]

# ========== 生产级Agent ==========
class ProductionAgent:
    """生产级ReAct Agent"""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_iterations: int = 10,
        temperature: float = 0.7,
        timeout: float = 30.0
    ):
        self.client = AsyncOpenAI()
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.timeout = timeout
        self.tools: Dict[str, Tool] = {}
        self.history: List[AgentStep] = []
        self.system_prompt = self._build_system_prompt()
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        # 更新system prompt
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建System Prompt"""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return f"""你是一个智能助手，可以使用工具解决问题。

可用工具：
{tool_descriptions}

请按照以下格式回答（严格遵循）：
Thought: 你的思考过程
Action: 工具名称{{"参数名": "参数值"}}

工具会在你输出Action后自动执行，并返回Observation。

规则：
1. 每次只输出一个Thought和一个Action
2. 如果工具返回错误，分析原因并尝试其他方法
3. 完成任务后，输出: Action: finish{{"answer": "最终答案"}}
4. 如果无法完成，输出: Action: finish{{"answer": "无法完成，原因：..."}}
"""
    
    async def run(self, question: str) -> Dict[str, Any]:
        """
        运行Agent
        
        Returns:
            {
                "success": bool,
                "answer": str,
                "steps": List[AgentStep],
                "iterations": int,
                "duration": float
            }
        """
        start_time = asyncio.get_event_loop().time()
        self.history = []
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"问题: {question}"}
        ]
        
        try:
            for iteration in range(self.max_iterations):
                # 调用LLM
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature
                    ),
                    timeout=self.timeout
                )
                
                content = response.choices[0].message.content
                
                # 解析Thought和Action
                parsed = self._parse_response(content)
                
                if not parsed:
                    return self._build_result(
                        False, "无法解析LLM响应", iteration + 1, start_time
                    )
                
                thought, action_name, action_input = parsed
                
                # 记录步骤
                step = AgentStep(
                    step_number=iteration + 1,
                    thought=thought,
                    action=action_name,
                    action_input=action_input,
                    observation=""
                )
                
                # 检查是否完成
                if action_name == "finish":
                    step.observation = "任务完成"
                    self.history.append(step)
                    return self._build_result(
                        True,
                        action_input.get("answer", "完成"),
                        iteration + 1,
                        start_time
                    )
                
                # 执行工具
                if action_name in self.tools:
                    observation = await self.tools[action_name].execute(**action_input)
                else:
                    observation = f"错误: 未知工具 '{action_name}'"
                
                step.observation = observation
                self.history.append(step)
                
                # 更新对话历史
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            
            # 达到最大迭代次数
            return self._build_result(
                False, "达到最大迭代次数", self.max_iterations, start_time
            )
            
        except asyncio.TimeoutError:
            return self._build_result(
                False, "执行超时", len(self.history), start_time
            )
        except Exception as e:
            return self._build_result(
                False, f"执行错误: {str(e)}", len(self.history), start_time
            )
    
    def _parse_response(self, content: str) -> Optional[tuple]:
        """解析LLM响应"""
        try:
            # 提取Thought
            thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", content, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""
            
            # 提取Action
            action_match = re.search(r"Action:\s*(\w+)\s*\{\{(.+?)\}\}", content, re.DOTALL)
            if not action_match:
                # 尝试其他格式
                action_match = re.search(r"Action:\s*(\w+)\[(.+?)\]", content)
                if action_match:
                    action_name = action_match.group(1)
                    action_input = {"query": action_match.group(2)}
                    return thought, action_name, action_input
                return None
            
            action_name = action_match.group(1)
            action_json = "{" + action_match.group(2) + "}"
            action_input = json.loads(action_json)
            
            return thought, action_name, action_input
            
        except Exception as e:
            print(f"解析错误: {e}")
            return None
    
    def _build_result(
        self,
        success: bool,
        answer: str,
        iterations: int,
        start_time: float
    ) -> Dict[str, Any]:
        """构建结果"""
        duration = asyncio.get_event_loop().time() - start_time
        return {
            "success": success,
            "answer": answer,
            "steps": self.history,
            "iterations": iterations,
            "duration": round(duration, 2)
        }
    
    def get_execution_trace(self) -> str:
        """获取执行轨迹（用于调试）"""
        trace = []
        for step in self.history:
            trace.append(f"""
Step {step.step_number}:
  Thought: {step.thought}
  Action: {step.action}({step.action_input})
  Observation: {step.observation}
""")
        return "\n".join(trace)

# ========== 使用示例 ==========
async def main():
    # 创建Agent
    agent = ProductionAgent(max_iterations=5)
    
    # 注册工具
    agent.register_tool(Tool(
        name="search",
        description="搜索信息，参数: query (str)",
        func=async_search
    ))
    agent.register_tool(Tool(
        name="calculate",
        description="计算数学表达式，参数: expression (str)",
        func=calculate
    ))
    agent.register_tool(Tool(
        name="finish",
        description="完成任务，参数: answer (str)",
        func=lambda answer: answer
    ))
    
    # 运行测试
    result = await agent.run("搜索Python的相关信息，然后计算2+3")
    
    print(f"成功: {result['success']}")
    print(f"答案: {result['answer']}")
    print(f"迭代次数: {result['iterations']}")
    print(f"耗时: {result['duration']}秒")
    print(f"\n执行轨迹:\n{agent.get_execution_trace()}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.2 关键特性说明

| 特性 | 说明 | 对应Java经验 |
|------|------|-------------|
| **异步设计** | 所有工具调用都是async | 类似CompletableFuture |
| **类型安全** | 使用dataclass和类型提示 | 类似Java泛型 |
| **错误处理** | 完整的异常捕获和超时控制 | 类似try-catch |
| **可观测性** | 记录完整执行轨迹 | 类似链路追踪 |
| **工具注册** | 动态工具注册机制 | 类似SPI |

---

## 第五章：记忆管理实现（进阶篇）

### 5.1 为什么需要记忆？

**没有记忆的问题**：
```
用户: 我叫张三
Agent: 你好张三！
用户: 我叫什么？
Agent: 我不知道你的名字。
```

**有记忆的效果**：
```
用户: 我叫张三
Agent: 你好张三！
用户: 我叫什么？
Agent: 你叫张三。
```

### 5.2 记忆类型

| 记忆类型 | 存储位置 | 用途 | 示例 |
|----------|----------|------|------|
| **短期记忆** | 上下文窗口 | 当前对话 | 之前的问答 |
| **长期记忆** | 向量数据库 | 跨会话 | 用户偏好、历史记录 |
| **工作记忆** | 变量/状态 | 任务执行中 | 当前任务进度 |

### 5.3 实现代码

```python
# memory.py - 记忆管理实现
from typing import List, Dict
from dataclasses import dataclass
import json

@dataclass
class Message:
    """对话消息"""
    role: str  # "user" 或 "assistant"
    content: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()

class ConversationMemory:
    """对话记忆管理"""
    
    def __init__(self, max_messages: int = 10):
        self.messages: List[Message] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append(Message(role=role, content=content))
        # 保持最近N条
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> str:
        """获取对话上下文"""
        context = []
        for msg in self.messages:
            prefix = "用户" if msg.role == "user" else "助手"
            context.append(f"{prefix}: {msg.content}")
        return "\n".join(context)
    
    def clear(self):
        """清空记忆"""
        self.messages = []
    
    def to_dict(self) -> List[Dict]:
        """序列化"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]
    
    @classmethod
    def from_dict(cls, data: List[Dict]) -> "ConversationMemory":
        """反序列化"""
        memory = cls()
        for item in data:
            memory.add_message(item["role"], item["content"])
        return memory

# 在Agent中使用记忆
class AgentWithMemory(ProductionAgent):
    """带记忆功能的Agent"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = ConversationMemory(max_messages=10)
    
    async def chat(self, user_input: str) -> str:
        """对话模式（带记忆）"""
        # 添加用户输入到记忆
        self.memory.add_message("user", user_input)
        
        # 构建带上下文的prompt
        context = self.memory.get_context()
        full_prompt = f"""之前的对话：
{context}

请基于以上对话上下文回答问题。"""
        
        # 调用Agent
        result = await self.run(full_prompt)
        
        # 添加助手回复到记忆
        if result["success"]:
            self.memory.add_message("assistant", result["answer"])
        
        return result["answer"]
```

---

## 第六章：常见问题排查（FAQ）

### Q1: LLM不按照格式输出Action？

**现象**：
```
LLM输出: 我需要搜索一下
而不是: Thought: 我需要搜索
         Action: search[xxx]
```

**解决方案**：
1. **强化Prompt**：明确说明输出格式
2. **Few-shot示例**：给1-2个示例
3. **后处理**：用正则强制提取

```python
# Few-shot Prompt示例
FEW_SHOT_PROMPT = """
示例1:
问题: 北京天气怎么样？
Thought: 用户问天气，我需要查询天气
Action: search{"query": "北京天气"}

示例2:
问题: 25+33等于多少？
Thought: 用户问数学计算，我需要计算
Action: calculate{"expression": "25+33"}

现在请回答：
问题: {question}
"""
```

### Q2: 工具调用失败，Agent无限循环？

**现象**：同一个Action重复执行

**解决方案**：
1. **错误信息反馈**：告诉LLM工具为什么失败
2. **最大迭代限制**：设置max_iterations
3. **去重检测**：记录已执行过的Action

```python
def is_duplicate_action(self, action_name: str, action_input: dict) -> bool:
    """检测重复Action"""
    for step in self.history:
        if (step.action == action_name and 
            step.action_input == action_input):
            return True
    return False
```

### Q3: 响应时间太长？

**优化方案**：
```python
# 1. 工具调用并行化
async def parallel_tool_calls(tasks: List[dict]):
    """并行执行多个工具调用"""
    async def execute(task):
        tool = self.tools.get(task["tool"])
        return await tool.execute(**task["params"])
    
    results = await asyncio.gather(*[execute(t) for t in tasks])
    return results

# 2. 缓存常用查询
from functools import lru_cache

@lru_cache(maxsize=100)
async def cached_search(query: str):
    return await search(query)

# 3. 使用更快的模型
# gpt-3.5-turbo 比 gpt-4 快且便宜
```

---

## 第七章：实战练习

### 练习1：实现一个计算器Agent（30分钟）

**目标**：能执行多步数学计算

**要求**：
1. 支持加减乘除
2. 能处理多步计算（如：25+33，结果再乘以2）
3. 记录计算历史

**提示代码**：
```python
class CalculatorAgent(ReActAgent):
    def __init__(self):
        super().__init__()
        self.memory = {"last_result": 0}
    
    def calculate(self, expression: str) -> str:
        # 支持使用上次结果
        expression = expression.replace("上次结果", str(self.memory["last_result"]))
        result = eval(expression)
        self.memory["last_result"] = result
        return str(result)
```

### 练习2：实现带记忆的问答Agent（1小时）

**目标**：记住用户信息，支持多轮对话

**要求**：
1. 记住用户姓名
2. 记住用户偏好
3. 跨轮次使用这些信息

### 练习3：实现多工具组合Agent（2小时）

**目标**：组合多个工具解决复杂问题

**场景**：
```
用户: 帮我查一下Python的最新版本，然后计算一下从1991年到现在多少年了？

Agent需要：
1. 搜索Python最新版本
2. 提取年份信息
3. 计算2024-1991
4. 整合答案
```

---

## 总结

读完本文，你应该掌握：
- ✅ Agent的核心概念和与RAG的区别
- ✅ ReAct模式（Thought-Action-Observation）
- ✅ 50行代码实现最简Agent
- ✅ 生产级Agent的完整封装
- ✅ 记忆管理实现
- ✅ 常见问题排查

**下一步**：学习LangGraph，实现更复杂的工作流编排

---

**本文整理自**: ReAct论文 + 生产实践 + 作者实战经验  
**整理时间**: 2026-03-04
