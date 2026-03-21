# 体检报告解读与健康咨询 Agent

一个面向中国医疗语境的最小可演示产品，包含：

- 体检/检验报告上传与结构化解析
- 自定义 ReAct 健康咨询 Agent
- 联网抓取知识库初始化与受控引用
- 健康小结 Markdown/PDF 导出

## 目录

- `backend/`: FastAPI + SQLite 后端
- `frontend/`: React + TypeScript 前端
- `data/`: SQLite、上传文件和导出结果

## 后端启动

1. 安装 Python 3.11+
2. 在 `backend/.env` 中配置千问与 WHO 凭据
3. 安装依赖：

```powershell
cd backend
python -m pip install -e .
```

4. 启动：

```powershell
uvicorn app.main:app --reload --port 8000
```

## 前端启动

```powershell
cd frontend
npm install
npm run dev
```

前端默认请求 `http://localhost:8000`。

## 说明

- 这是 v1 最小实现，未包含鉴权、多用户和定时任务。
- 原始文档中的密钥不能继续使用，部署前应全部轮换并写入环境变量。

