# 健康管理长期跟踪 Agent 工作台

一个面向医疗健康场景的 Agent 项目，支持多会话健康咨询、体检报告上传解析、自动报告解读、异常指标追问、健康小结生成与 PDF 导出。

这个项目的重点不是做“医疗诊断”，而是实现一套更完整、可解释、可追踪的健康管理 Agent 工作流：

- 上传体检报告
- 异步解析并抽取结构化指标
- 自动生成报告解读
- 围绕异常指标持续追问
- 生成健康小结并导出 PDF
- 通过轨迹面板查看 Agent 运行过程

## 项目亮点

- 基于 `LangGraph` 构建任务型 Agent 主链，已接入 `goal / task run / trace / planner / replanner / memory`
- 支持体检报告上传、异步解析、自动解读、持续追问、小结导出的一体化闭环
- 支持多层记忆，包括短期上下文、会话摘要记忆、报告洞察记忆和历史报告关联
- 支持报告工具链，包括指标标准化、异常项解释、趋势比较和风险标记
- 补充了 Agent 评测、报告主链回归测试和 Agent 轨迹调试能力

## 技术栈

- 后端：Python / FastAPI / SQLModel / LangGraph
- 前端：React / TypeScript / Vite
- 数据层：SQLite（默认） / MySQL / Redis
- 模型与增强：Qwen / OCR / WHO ICD-11
- 交互：SSE 流式输出
- 工程化：Alembic / Docker / pytest

## 核心能力

### 1. 多会话健康咨询工作台

- 左侧支持新建、切换、重命名、删除会话
- 中间支持流式聊天
- 支持围绕当前报告持续追问
- 支持查看 Agent 轨迹与运行调试信息

### 2. 体检报告解析链路

- 支持 PDF / JPG / PNG 上传
- 支持文本型 PDF 提取与图片 OCR
- 自动抽取结构化指标
- 识别异常项与参考范围
- 报告解析完成后自动生成一条报告解读

### 3. 健康小结

- 基于当前报告和对话上下文生成健康小结
- 支持历史版本查看
- 支持 PDF 下载

### 4. Agent 工程能力

- 基于 `LangGraph` 的任务型 Agent 主链
- 显式拆分安全规则、意图路由、计划生成、工具执行和结构化生成
- 支持 `goal / task run / trace / planner / replanner / memory`
- 支持 Agent 轨迹查看与运行回放

### 5. 长期跟踪能力

- 会话摘要记忆 `session memory`
- 报告洞察记忆 `report insight`
- 会话与历史报告关联
- 报告趋势比较与规则风险标记

### 6. 回归测试与评测

- Agent 行为回归测试
- Agent eval 数据集
- 报告主链回归测试
- 覆盖路由、安全降级、报告追问、自动报告解读、多轮 memory 等关键场景

## 项目结构

```text
backend/
  app/
    agent_graph/        # LangGraph 主链
    api/                # FastAPI 路由
    core/               # 配置、数据库、schema
    models/             # SQLModel 实体
    services/           # Agent / 报告 / 小结 / 缓存 / memory / eval
  migrations/           # Alembic 迁移
  tests/                # 后端测试与评测

frontend/
  src/
    App.tsx             # 主工作台
    api.ts              # 前端接口
    styles.css          # 样式

data/
  uploads/              # 上传文件
  output/               # 导出结果
  sqlite/               # 本地 SQLite 数据
```

## 快速开始

### 1. 启动后端

```powershell
cd D:\35174\Desktop\agent2\backend
python -m pip install -e .[dev]
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

### 2. 启动前端

```powershell
cd D:\35174\Desktop\agent2\frontend
npm install
npm run dev
```

前端默认地址：

- [http://localhost:5173](http://localhost:5173)

后端文档地址：

- [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. 可选：单独运行报告解析 Worker

如果你不想使用内嵌 worker，也可以单独启动：

```powershell
cd D:\35174\Desktop\agent2\backend
python -m app.worker
```

默认情况下，后端启动时会自动拉起一个内嵌 worker，因此本地开发通常只需要启动前后端。

## 环境变量

后端主要配置在 `backend/.env`。

常用配置示例：

```env
QWEN_API_KEY=your-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen3.5-plus
QWEN_FAST_MODEL=qwen3.5-flash
QWEN_MAX_MODEL=qwen3-max
QWEN_VL_MODEL=qwen3-vl-flash

SHORT_TERM_CONTEXT_TURNS=4

WHO_ENABLED=true
WHO_TIMEOUT_SECONDS=6
WHO_CONNECT_TIMEOUT_SECONDS=3
WHO_TRUST_ENV=false

DATABASE_URL=sqlite:///../data/sqlite/medical_agent.db
REDIS_URL=redis://127.0.0.1:6379/0
```

说明：

- `FAST_MODEL` 用于输入分析、意图路由、轻量结构化任务
- `MAX_MODEL` 用于更高质量的最终回答生成
- `VL_MODEL` 用于图片 OCR
- `WHO_*` 用于术语标准来源增强，可按需关闭
- 默认可用 SQLite，本项目也支持切换到 MySQL + Redis

## 数据库与迁移

项目已经接入 `Alembic`。

首次迁移：

```powershell
cd D:\35174\Desktop\agent2\backend
python -m alembic -c .\alembic.ini upgrade head
```

如果使用 MySQL，先在 `.env` 里修改 `DATABASE_URL`，再执行迁移。

## 测试

运行 Agent 评测与主链回归：

```powershell
cd D:\35174\Desktop\agent2
python -m pytest .\backend\tests\test_agent_eval_runner.py
python -m pytest .\backend\tests\test_report_pipeline_eval.py
```

运行核心 Agent 测试：

```powershell
cd D:\35174\Desktop\agent2
python -m pytest .\backend\tests\test_react_agent.py
```

## 当前定位

这是一个高完成度、工程化的健康管理 Agent 原型系统，重点在：

- Agent 架构设计
- 医疗场景下的受控问答
- 报告解析与长期跟踪
- 多层 memory
- 可观测性与回归评测

它不是医疗诊断系统，也不提供处方、剂量或替代医生决策。

## 后续可继续增强的方向

- 更完整的安全硬规则
- 用户级长期画像 memory
- 健康时间线视图
- 更强的报告趋势分析
- 更完整的端到端评测与部署方案
