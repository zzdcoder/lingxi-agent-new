# 灵犀智能助手

## 项目简介

基于 LangChain 构建的企业级检索增强生成系统，支持智能文档处理与知识库问答。

## 目录结构

```
enterprise_rag/
├── app/              # 应用入口
├── api/              # RESTful API 层
│   └── routes/       # 路由定义
├── core/             # 核心配置与异常
├── ingestion/        # 文档摄入管道
├── embeddings/       # 文本嵌入服务
├── vectorstore/      # 向量数据库
├── rag/              # RAG 引擎
├── llm/              # 大语言模型集成
├── tools/            # Agent 工具
├── utils/            # 公共工具
├── models/           # 数据模型
├── tests/            # 测试套件
│   ├── unit/         # 单元测试
│   └── integration/  # 集成测试
├── scripts/          # 运维脚本
└── docs/             # 项目文档
```

## 快速开始

1. 复制环境变量模板
2. 安装依赖
3. 启动服务

## 开发规范

- 所有包须包含 `__init__.py` 并写明职责
- 配置统一收口至 `core` 包
- 异常按模块定义，继承自 `RAGException`
