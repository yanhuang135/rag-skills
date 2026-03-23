#!/bin/bash

# 文档问答助手启动脚本

set -e

echo "=== 文档问答助手启动脚本 ==="

# 切换到脚本目录
cd "$(dirname "$0")"

# 选择 Python 命令（兼容 python3/python）
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "错误：未找到 python3 或 python"
    exit 1
fi

# 检查Python版本
echo "检查Python版本..."
$PYTHON_CMD --version

# 安装依赖
echo "安装依赖包..."
$PYTHON_CMD -m pip install -r requirements.txt

# 加载环境变量
if [ -f ".env" ]; then
    echo "加载环境变量..."
    set -a
    . ./.env
    set +a
else
    echo "警告：未找到.env文件，使用默认配置"
fi

# 运行文档问答助手
if [ "$#" -gt 0 ]; then
    QUESTION="$*"
    echo "启动单次问答模式..."
    echo "问题: $QUESTION"
    $PYTHON_CMD scripts/doc_qa_assistant.py -q "$QUESTION"
else
    echo "启动交互问答模式（输入 exit 退出）..."
    while true; do
        printf "请输入问题: "
        IFS= read -r QUESTION
        if [ -z "$QUESTION" ]; then
            continue
        fi
        if [ "$QUESTION" = "exit" ] || [ "$QUESTION" = "quit" ]; then
            echo "已退出问答模式"
            break
        fi
        $PYTHON_CMD scripts/doc_qa_assistant.py -q "$QUESTION"
    done
fi
