# ReAct: 推理与行动协同框架（中文版）

**原文标题**: ReAct: Synergizing Reasoning and Acting in Language Models  
**作者**: Shunyu Yao et al. (Google Research)  
**原文链接**: https://arxiv.org/abs/2210.03629  
**发表时间**: 2022年10月

---

## 中文摘要

虽然大语言模型在语言理解和交互式决策任务上展现了令人印象深刻的能力，但其推理能力（如思维链提示）和行动能力（如行动计划生成）主要被作为独立主题研究。本文探索使用LLM以交错方式生成推理轨迹和任务特定行动，使两者产生更大协同效应：推理轨迹帮助模型诱导、跟踪和更新行动计划以及处理异常，而行动使其能够与外部资源（如知识库或环境）交互以收集额外信息。

我们将该方法应用于语言理解和决策制定两大类任务，证明了其相比最先进基线的有效性，以及相比无推理或行动组件的方法在人类可解释性和可信度方面的改进。

---

## 1. 核心概念

### 1.1 什么是ReAct

ReAct = **Re**asoning（推理）+ **Act**ing（行动）

**核心思想**: 让LLM交替进行"思考"和"行动"，形成推理-行动链。

### 1.2 对比其他方法

| 方法 | 特点 | 局限 |
|------|------|------|
| 标准Prompt | 直接回答 | 无推理过程，易出错 |
| Chain-of-Thought | 只推理 | 无法获取外部信息 |
| 行动-only | 只调用工具 | 缺乏系统性思考 |
| **ReAct** | **推理+行动交替** | **结合两者优势** |

---

## 2. ReAct工作流程

### 2.1 基本模式

```
Thought（思考）→ Action（行动）→ Observation（观察）
    ↓
Thought → Action → Observation
    ↓
...（循环直到完成任务）
```

### 2.2 具体示例

**任务**: "《奥本海默》电影的导演还导演过哪些其他电影？"

**Step 1 - Thought**:  
我需要先找到《奥本海默》电影的导演是谁。

**Step 1 - Action**:  
搜索[《奥本海默》电影导演]

**Step 1 - Observation**:  
《奥本海默》由克里斯托弗·诺兰执导。

**Step 2 - Thought**:  
现在我需要查找克里斯托弗·诺兰导演的其他作品。

**Step 2 - Action**:  
搜索[克里斯托弗·诺兰导演作品]

**Step 2 - Observation**:  
诺兰的代表作包括：《盗梦空间》《星际穿越》《黑暗骑士三部曲》《记忆碎片》《敦刻尔克》等。

**Step 3 - Thought**:  
我已经获得了足够信息，可以回答用户问题了。

**Step 3 - Action**:  
完成[克里斯托弗·诺兰的其他作品包括：《盗梦空间》《星际穿越》《蝙蝠侠：黑暗骑士三部曲》《记忆碎片》《敦刻尔克》等。]

---

## 3. 核心组件设计

### 3.1 Thought（思考）

**作用**:
- 分解复杂任务为子任务
- 决定下一步行动
- 跟踪任务进度
- 处理错误或异常情况

**示例模板**:
```
Thought: 用户问的是X，我需要先获取Y信息才能回答。
我需要使用搜索工具来查找Y。
```

### 3.2 Action（行动）

**常见行动类型**:
- **Search[query]**: 搜索信息
- **Lookup[entity]**: 查找特定实体
- **Calculate[expression]**: 计算
- **Finish[answer]**: 完成任务

**行动格式**:
```
Action: Search[《奥本海默》导演]
```

### 3.3 Observation（观察）

**作用**: 行动执行后返回的结果，作为下一步思考的输入。

**特点**:
- 来自外部工具/环境
- 可能包含错误信息
- 需要解析和整合

---

## 4. 工程实现

### 4.1 Prompt模板设计

```python
REACT_PROMPT = """
解决一个问题，通过交替进行Thought（思考）、Action（行动）和Observation（观察）。

可用工具:
- Search[query]: 搜索信息
- Calculator[expression]: 计算表达式
- Finish[answer]: 给出最终答案

示例:
问题: 2023年诺贝尔物理学奖得主是谁？
Thought: 我需要搜索2023年诺贝尔物理学奖的信息。
Action: Search[2023年诺贝尔物理学奖得主]
Observation: 2023年诺贝尔物理学奖授予Anne L'Huillier、Ferenc Krausz和Pierre Agostini。
Thought: 我已获得完整信息，可以回答。
Action: Finish[2023年诺贝尔物理学奖得主是Anne L'Huillier、Ferenc Krausz和Pierre Agostini。]

现在开始:
问题: {question}
"""
```

### 4.2 Python实现

```python
class ReActAgent:
    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
    
    def run(self, question: str) -> str:
        context = f"Question: {question}\n"
        
        for i in range(self.max_iterations):
            # 生成Thought和Action
            response = self.llm.generate(REACT_PROMPT + context)
            
            # 解析Thought
            thought = self._extract_thought(response)
            context += f"Thought: {thought}\n"
            
            # 解析Action
            action = self._extract_action(response)
            action_type, action_input = self._parse_action(action)
            context += f"Action: {action}\n"
            
            # 执行Action
            if action_type == "Finish":
                return action_input
            
            observation = self.tools[action_type].run(action_input)
            context += f"Observation: {observation}\n"
        
        return "达到最大迭代次数，未找到答案"
    
    def _extract_thought(self, text: str) -> str:
        # 提取Thought内容
        pass
    
    def _extract_action(self, text: str) -> str:
        # 提取Action内容
        pass
    
    def _parse_action(self, action: str) -> tuple:
        # 解析Action类型和输入
        pass
```

### 4.3 LangChain实现

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="用于搜索信息"
    ),
    Tool(
        name="Calculator", 
        func=calculator_function,
        description="用于数学计算"
    )
]

# 获取ReAct Prompt
prompt = hub.pull("hwchase17/react")

# 创建Agent
agent = create_react_agent(llm, tools, prompt)

# 执行
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "你的问题"})
```

---

## 5. 实验结果

### 5.1 问答任务（HotpotQA）

| 方法 | EM分数 | 相对提升 |
|------|--------|----------|
| 标准Prompt | 28.7% | - |
| CoT | 29.4% | +2.4% |
| Action-only | 20.7% | -27.9% |
| **ReAct** | **29.8%** | **+3.8%** |

### 5.2 事实验证（Fever）

ReAct在Fever数据集上达到**91.2%**准确率，有效减少了幻觉。

### 5.3 决策任务（ALFWorld, WebShop）

- **ALFWorld**: ReAct比模仿学习方法提升**34%**
- **WebShop**: ReAct比模仿学习方法提升**10%**

---

## 6. 优势分析

### 6.1 可解释性

ReAct产生人类可读的推理轨迹，便于：
- 调试Agent行为
- 理解模型决策过程
- 人工审核和纠错

### 6.2 可信度

通过显式推理，用户可以验证：
- 信息来源
- 推理逻辑
- 结论依据

### 6.3 容错性

当行动失败时，模型可以：
- 识别错误
- 调整策略
- 尝试替代方案

---

## 7. 局限与改进

### 7.1 主要局限

1. **上下文长度限制**: 长轨迹可能超出模型上下文
2. **错误累积**: 早期错误可能影响后续推理
3. **工具依赖**: 工具质量直接影响Agent性能

### 7.2 改进方向

1. **Reflexion**: 让Agent自我反思和纠错
2. **多Agent协作**: 多个ReAct Agent分工协作
3. **学习优化**: 从历史轨迹学习更好的推理策略

---

## 8. 生产实践建议

### 8.1 工具设计

**原则**:
- 工具功能单一明确
- 错误信息清晰
- 返回格式统一

**示例**:
```python
class SearchTool:
    def run(self, query: str) -> str:
        try:
            results = search_engine.search(query)
            return format_results(results)
        except Exception as e:
            return f"搜索失败: {str(e)}"
```

### 8.2 安全防护

1. **工具权限控制**: 限制工具可调用的范围
2. **输入验证**: 验证工具输入的合法性
3. **超时控制**: 防止长时间运行的工具
4. **最大迭代限制**: 防止无限循环

### 8.3 监控优化

**关键指标**:
- 平均迭代次数
- 工具调用成功率
- 任务完成率
- 平均响应时间

**优化方向**:
- 缓存常见查询结果
- 优化Prompt减少迭代次数
- 使用更强大的模型提升推理质量

---

## 9. 与其他Agent架构的关系

```
ReAct (推理+行动)
    ↓
Reflexion (增加自我反思)
    ↓
Multi-Agent (多个ReAct协作)
    ↓
AutoGPT (全自动目标驱动)
```

ReAct是许多现代Agent架构的基础组件。

---

## 10. 总结

ReAct是一种简单但强大的Agent设计模式，通过交替进行推理和行动，使LLM能够：
1. 系统性解决复杂问题
2. 与外部工具和环境交互
3. 提供可解释的决策过程

对于工程实践，ReAct提供了一个易于实现、可扩展的基础框架，是构建AI Agent的推荐起点。

---

**翻译整理**: AI Agent学习资料  
**原文引用**: Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.
