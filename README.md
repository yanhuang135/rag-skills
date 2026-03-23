# 本地文档问答（doc-qa-assistant）

基于 `.agent/skills/doc-qa-assistant` 的本地 RAG 文档问答示例：

- 扫描 `knowledge/` 下的 PDF
- 生成 Embedding 并落盘到 FAISS（`vector_db/`）
- 使用 OpenAI 兼容接口模型进行问答
- 返回答案及引用来源页码

---

## 目录结构

```text
rag-skills/
├── .agent/
│   └── skills/
│       └── doc-qa-assistant/
│           ├── run.sh
│           ├── requirements.txt
│           ├── .env.example
│           ├── .env
│           ├── vector_db/
│           │   ├── index.faiss
│           │   └── index.pkl
│           └── scripts/
│               ├── build_vector_store.py
│               └── doc_qa_assistant.py
└── knowledge/
    └── ... PDF 文档 ...
```

---

## 快速开始

### 1) 安装依赖

```bash
cd .agent/skills/doc-qa-assistant
python -m pip install -r requirements.txt
```

### 2) 配置环境变量

复制 `.env.example` 为 `.env`，至少配置：

- `LLM_API_BASE`
- `LLM_API_KEY`
- `LLM_MODEL`

可选：

- `EMBEDDING_MODEL`（默认 `BAAI/bge-base-zh`）
- `EMBEDDING_MODEL_FALLBACK`
- `HF_ENDPOINT=https://hf-mirror.com`（国内网络建议）
- `EMBEDDING_LOCAL_FILES_ONLY=true`（离线模型）

### 3) 首次构建向量库

```bash
cd .agent/skills/doc-qa-assistant
python scripts/build_vector_store.py
```

构建完成后会生成：

- `.agent/skills/doc-qa-assistant/vector_db/index.faiss`
- `.agent/skills/doc-qa-assistant/vector_db/index.pkl`

### 4) 开始问答

#### 单次问答

```bash
cd .agent/skills/doc-qa-assistant
bash run.sh "人工智能在营销领域的应用趋势是什么？"
```

或直接：

```bash
python scripts/doc_qa_assistant.py -q "人工智能在营销领域的应用趋势是什么？"
```

#### 交互问答

```bash
cd .agent/skills/doc-qa-assistant
bash run.sh
```

输入 `exit` 或 `quit` 退出。

---

## 工作机制

1. `doc_qa_assistant.py` 启动后加载本地 FAISS 向量库  
2. 将问题向量化并做相似度检索（默认 Top-K=3）  
3. 将检索上下文填充到 `PromptTemplate`  
4. 调用大模型生成回答，并返回来源 `source/page`  

---

## 常见问题

### 1) `错误：knowledge 目录不存在`

请确认仓库根目录存在 `knowledge/`，并从正确目录运行脚本。

### 2) 模型下载超时（HuggingFace）

建议在 `.env` 中设置：

```env
HF_ENDPOINT=https://hf-mirror.com
EMBEDDING_MODEL_FALLBACK=BAAI/bge-small-zh-v1.5
```

### 3) 结果乱码（Windows 控制台）

建议使用 UTF-8 终端或在 PowerShell 设置 UTF-8 编码后运行。

### 4) 向量库不存在

先执行：

```bash
python scripts/build_vector_store.py
```

---

## 后续迭代（非 PDF）

当前 `doc-qa-assistant` 以 PDF 向量化为主，后续建议按以下顺序扩展：

1. **Markdown / TXT**
   - 处理方式：直接读取文本并分块（不需要 OCR/版面还原）
   - 落地建议：在 `build_vector_store.py` 增加 `*.md`、`*.txt` 文件扫描和加载器分支
   - 价值：实现“安全知识文档、规范文档”与 PDF 同库检索

2. **Excel / CSV**
   - 处理方式：按工作表和关键列转为文本片段（保留 `sheet_name`、行号、主键列）
   - 落地建议：将每行或每组聚合结果构造成 `Document`，写入 `metadata`（如 `source/sheet/row`）
   - 价值：支持“库存、订单、统计表”类问题的语义检索与溯源

3. **Word（DOCX）**
   - 处理方式：提取段落、标题、表格后分块入库
   - 落地建议：保留章节层级信息到 `metadata`（如 `heading_path`）
   - 价值：覆盖制度文档、方案文档等常见企业资料

4. **统一检索质量策略**
   - 按文件类型设置不同 `chunk_size/chunk_overlap`
   - 引入重排（rerank）提升召回质量
   - 根据问题类型做目录级过滤（如优先 `Safety Knowledge`）

5. **增量更新与多库管理**
   - 为每个文件记录 hash 与更新时间，只重建变更文件向量
   - 支持多索引（按领域拆分）与路由检索，降低噪音召回

建议优先级：**Markdown/TXT -> Excel/CSV -> DOCX -> 增量更新**。

---

## 说明

- 当前技能主要面向 **PDF 文档问答**。  
