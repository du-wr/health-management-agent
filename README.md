# 体检报告解读与健康咨询 Agent Demo

这是一个面向中文场景的全栈演示项目，目标是做一个“能读体检报告、能回答健康相关问题、能导出健康小结”的 Demo。

项目重点不是做医疗诊断，而是演示一条相对完整的产品链路：

- 用户上传体检报告
- 后端解析报告并提取结构化指标
- Agent 围绕报告、医学名词、本地知识库和 WHO ICD-11 做解释
- 前端用流式方式展示回答与解析进度
- 最后生成一份 Markdown / PDF 形式的健康小结

## 项目结构

- `backend/`
  - FastAPI + SQLModel + SQLite
  - 负责报告解析、Agent、知识库、WHO 查询、小结导出
- `frontend/`
  - React + TypeScript + Vite
  - 负责上传、对话、进度展示、小结展示
- `data/`
  - 数据库存储、上传文件、导出 PDF

## 启动方式

### 后端

```powershell
cd backend
copy .env.example .env
python -m pip install -e .[dev]
uvicorn app.main:app --reload --port 8000
```

### 前端

```powershell
cd frontend
npm install
npm run dev
```

默认访问地址：

- 前端：[http://localhost:5173](http://localhost:5173)
- 后端接口文档：[http://localhost:8000/docs](http://localhost:8000/docs)

## 关键环境变量

后端最重要的配置在 `backend/.env`：

```env
QWEN_API_KEY=your-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_FAST_MODEL=qwen3.5-flash
QWEN_MAX_MODEL=qwen3-max
QWEN_VL_MODEL=qwen3-vl-flash
SHORT_TERM_CONTEXT_TURNS=4
WHO_CLIENT_ID=your-who-client-id
WHO_CLIENT_SECRET=your-who-client-secret
```

说明：

- `FAST_MODEL` 用于输入分析、报告拆解、轻量结构化任务
- `MAX_MODEL` 用于最终高质量回答润色
- `VL_MODEL` 用于图片类报告 OCR
- WHO 凭据只影响“医学名词解释”这条链，不影响报告解析主流程

## 当前版本的能力边界

- 这是演示项目，不提供诊断、处方、剂量建议
- 本地知识库是预置种子，不联网抓取公开网页
- WHO ICD-11 主要用于术语标准化和增强术语解释的权威性
- 报告解析对文本型 PDF 体验较好，扫描件仍依赖视觉模型
- 后台任务目前使用 FastAPI `BackgroundTasks`，适合 Demo，不适合高并发生产环境

## 学习文档

如果你想系统理解这个 Demo，建议按下面顺序阅读：

1. [项目流程解析.md](./项目流程解析.md)
2. [后端流程解析.md](./后端流程解析.md)
3. [前端流程解析.md](./前端流程解析.md)
4. [主体Agent流程解析.md](./主体Agent流程解析.md)

这四份文档是按“总览 -> 后端 -> 前端 -> Agent 核心逻辑”的顺序写的，适合从零开始理解整个项目。
