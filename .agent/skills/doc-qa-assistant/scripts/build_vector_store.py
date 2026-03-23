#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建本地向量库（FAISS）
- 扫描 knowledge 目录下的 PDF
- 提取/分块
- 生成 embeddings
- 保存到 vector_db 目录
"""

import os
import sys
import shutil
import requests
from typing import List

from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", ".env"))

# 允许从同目录导入
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    sys.exit(1)


def _compute_paths():
    script_dir = os.path.dirname(__file__)
    skill_dir = os.path.dirname(script_dir)
    trae_dir = os.path.dirname(skill_dir)
    project_root = os.path.dirname(trae_dir)
    project_root = os.path.dirname(project_root)

    knowledge_dir = os.path.join(project_root, "knowledge")
    vector_db_path = os.path.join(skill_dir, "vector_db")
    return knowledge_dir, vector_db_path


def _is_model_endpoint_reachable(model_name: str, endpoint: str = None) -> bool:
    if os.path.isdir(model_name):
        return True

    base_url = endpoint.rstrip("/") if endpoint else "https://huggingface.co"
    check_url = f"{base_url}/{model_name}/resolve/main/modules.json"
    try:
        r = requests.head(check_url, timeout=5)
        return r.status_code < 500
    except requests.RequestException:
        return False


def _init_embeddings():
    embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-zh")
    embedding_model_fallback = os.environ.get("EMBEDDING_MODEL_FALLBACK", "BAAI/bge-small-zh-v1.5")

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_MAX_RETRIES", "1")

    hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    endpoints = [hf_endpoint] if hf_endpoint else [None, "https://hf-mirror.com"]
    model_candidates = [embedding_model]
    if embedding_model_fallback != embedding_model:
        model_candidates.append(embedding_model_fallback)

    local_files_only = os.environ.get("EMBEDDING_LOCAL_FILES_ONLY", "false").lower() == "true"

    print(f"Embedding 模型候选: {model_candidates}")
    last_error = None
    for endpoint in endpoints:
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
            print(f"使用 HuggingFace 端点: {endpoint}")
        for model_name in model_candidates:
            try:
                if not _is_model_endpoint_reachable(model_name, endpoint):
                    print(f"跳过不可达模型源: model={model_name}, endpoint={endpoint or 'https://huggingface.co'}")
                    continue
                emb = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu", "local_files_only": local_files_only},
                    encode_kwargs={"normalize_embeddings": True},
                )
                if model_name != embedding_model:
                    print(f"主模型不可用，已回退到: {model_name}")
                return emb
            except Exception as e:
                last_error = e
                print(f"加载失败: model={model_name}, endpoint={endpoint or 'default'}, error={e}")

    raise RuntimeError(f"无法加载Embedding模型: {last_error}")


def _get_pdf_files(knowledge_dir: str) -> List[str]:
    pdf_files = []
    for root, dirs, files in os.walk(knowledge_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def _process_pdf(pdf_path: str) -> List[Document]:
    print(f"处理PDF文件: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"  分割为 {len(split_docs)} 个文本块")
    return split_docs


def main():
    knowledge_dir, vector_db_path = _compute_paths()

    print(f"知识目录路径: {knowledge_dir}")
    if not os.path.exists(knowledge_dir):
        print("错误：knowledge 目录不存在")
        sys.exit(1)

    vector_force_rebuild = os.environ.get("VECTOR_DB_FORCE_REBUILD", "true").lower() == "true"
    if vector_force_rebuild and os.path.exists(vector_db_path):
        print(f"强制重建向量库：删除 {vector_db_path}")
        shutil.rmtree(vector_db_path)

    embeddings = _init_embeddings()

    pdf_files = _get_pdf_files(knowledge_dir)
    print(f"找到 {len(pdf_files)} 个PDF文件")
    if not pdf_files:
        print("没有找到PDF文件，向量数据库构建失败")
        sys.exit(1)

    all_docs: List[Document] = []
    for idx, pdf_file in enumerate(pdf_files, 1):
        docs = _process_pdf(pdf_file)
        
        all_docs.extend(docs)
        print(f"文档进度: {idx}/{len(pdf_files)}，累计文本块: {len(all_docs)}")

    if not all_docs:
        print("没有成功处理任何PDF文件，向量数据库构建失败")
        sys.exit(1)

    total_docs = len(all_docs)
    batch_size = max(1, int(os.environ.get("VECTOR_BUILD_BATCH_SIZE", "64")))
    print(f"创建向量存储，共 {total_docs} 个文本块，批大小 {batch_size}")

    first_batch = all_docs[:batch_size]
    vector_store = FAISS.from_documents(first_batch, embeddings)
    print(f"向量化进度: {len(first_batch)}/{total_docs}")

    for start in range(batch_size, total_docs, batch_size):
        end = min(start + batch_size, total_docs)
        batch_docs = all_docs[start:end]
        vector_store.add_documents(batch_docs)
        print(f"向量化进度: {end}/{total_docs}")

    os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
    vector_store.save_local(vector_db_path)
    print(f"向量数据库保存到: {vector_db_path}")


if __name__ == "__main__":
    main()

