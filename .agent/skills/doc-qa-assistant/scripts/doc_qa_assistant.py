#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档问答助手
结合LangChain将knowledge目录的PDF文档存入FAISS数据库
使用主流免费开源Embedding模型和大模型
"""

import os
import sys
import argparse
from typing import Dict, Any

from dotenv import load_dotenv

# 优先从 skill 目录下的 .env 加载配置，避免 shell 环境差异导致变量缺失
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# LangChain imports - 兼容 langchain 新旧版本
try:
    from langchain_community.vectorstores import FAISS
    try:
        from langchain_classic.chains import RetrievalQA
    except ImportError:
        from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    sys.exit(1)

from build_vector_store import main as build_vector_store_main
from build_vector_store import _init_embeddings as build_init_embeddings

class DocQAAssistant:
    """文档问答助手类"""
    
    def __init__(self):
        """初始化文档问答助手"""
        # 配置参数
        # 确保knowledge目录路径相对于项目根目录
        script_dir = os.path.dirname(__file__)
        skill_dir = os.path.dirname(script_dir)
        trae_dir = os.path.dirname(skill_dir)
        project_root = os.path.dirname(trae_dir)  # 现在是 .trae 目录
        project_root = os.path.dirname(project_root)  # 再上一级到项目根目录
        self.knowledge_dir = os.path.join(project_root, "knowledge")
        self.vector_db_path = os.path.join(skill_dir, "vector_db")
        self.llm_api_base = os.environ.get("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.llm_api_key = os.environ.get("LLM_API_KEY", "")
        self.llm_model = os.environ.get("LLM_MODEL", "qwen-turbo")
        
        # 初始化组件
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        
        # 显示知识目录路径
        print(f"知识目录路径: {self.knowledge_dir}")
        print(f"知识目录是否存在: {os.path.exists(self.knowledge_dir)}")
        
        # 初始化系统
        self.initialize()
    
    def initialize(self):
        """初始化系统"""
        print("正在初始化文档问答助手...")
        
        # 初始化Embedding模型
        self._init_embeddings()
        
        # 初始化向量存储
        if os.path.exists(self.vector_db_path):
            print("加载已存在的向量数据库...")
            self.vector_store = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("构建新的向量数据库...")
            self._build_vector_store()
        
        # 初始化QA链
        self._init_qa_chain()
        
        print("文档问答助手初始化完成！")
    
    def _init_embeddings(self):
        """初始化Embedding模型（复用 build_vector_store.py）"""
        self.embeddings = build_init_embeddings()

    def _init_qa_chain(self):
        """初始化QA链"""
        print(f"配置大模型: {self.llm_model}")
        
        # 配置大模型
        llm = ChatOpenAI(
            model=self.llm_model,
            base_url=self.llm_api_base,
            api_key=self.llm_api_key,
            temperature=0.1
        )
        
        # 定义提示模板
        prompt_template = """
        你是一个基于文档的智能问答助手。请根据以下提供的文档内容，回答用户的问题。
        
        文档内容：
        {context}
        
        用户问题：
        {question}
        
        请基于文档内容给出准确、简洁的回答。如果文档中没有相关信息，请明确说明。
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 创建QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def _build_vector_store(self):
        """构建向量存储（复用 build_vector_store.py）"""
        print("调用 build_vector_store.py 构建向量库...")

        build_vector_store_main()

        # 构建完成后重新加载向量库到当前实例
        self.vector_store = FAISS.load_local(
            self.vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("向量数据库构建并加载完成")
    
    def ask(self, question: str) -> Dict[str, Any]:
        """回答用户问题"""
        print(f"处理问题: {question}")
        
        if not self.qa_chain:
            return {"answer": "系统未初始化，请稍后再试", "sources": []}
        
        try:
            # 执行问答
            if hasattr(self.qa_chain, "invoke"):
                result = self.qa_chain.invoke({"query": question})
            else:
                result = self.qa_chain({"query": question})
            
            # 处理结果
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])
            
            # 提取来源信息
            sources = []
            for doc in source_documents:
                source_info = {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0)
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            print(f"问答处理失败: {e}")
            return {"answer": f"处理问题时出错: {e}", "sources": []}
    
    def update_vector_store(self):
        """更新向量存储"""
        print("更新向量存储...")
        self._build_vector_store()
        print("向量存储更新完成！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档问答助手")
    parser.add_argument("-q", "--question", type=str, default="", help="用户问题")
    args = parser.parse_args()

    # 初始化助手
    assistant = DocQAAssistant()

    # 优先使用 run.sh 传入的命令行参数
    single_question = (args.question or "").strip()
    if not single_question:
        single_question = input("请输入问题: ").strip()

    if not single_question:
        print("错误：问题不能为空")
        return

    result = assistant.ask(single_question)
    print(f"问: {single_question}")
    print(f"答: {result['answer']}")
    if result["sources"]:
        print("来源:")
        for source in result["sources"]:
            print(f"  - {source['source']} (第{source['page']}页)")

if __name__ == "__main__":
    main()
