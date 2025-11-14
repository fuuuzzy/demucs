#!/bin/bash
# Demucs UV 安装脚本
# 支持 CPU 和 CUDA 环境

set -e

echo "=================================="
echo "Demucs UV 安装脚本"
echo "=================================="
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 uv 是否已安装
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}UV 未安装，正在安装...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}✓ UV 安装完成${NC}"
else
    echo -e "${GREEN}✓ UV 已安装${NC}"
fi

echo ""
echo "请选择安装模式:"
echo "1) CPU 模式 (仅使用 CPU)"
echo "2) CUDA 11.8 模式 (推荐，支持大多数显卡)"
echo "3) CUDA 12.1 模式 (最新显卡)"
echo "4) 仅安装基础依赖 (手动安装 PyTorch)"
echo ""
read -p "请输入选项 [1-4]: " choice

# 创建虚拟环境
echo ""
echo -e "${YELLOW}创建虚拟环境...${NC}"
uv venv
source .venv/bin/activate
echo -e "${GREEN}✓ 虚拟环境已创建${NC}"

# 根据选择安装依赖
case $choice in
    1)
        echo ""
        echo -e "${YELLOW}安装 CPU 版本...${NC}"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        uv pip install -e ".[cpu]"
        echo -e "${GREEN}✓ CPU 版本安装完成${NC}"
        ;;
    2)
        echo ""
        echo -e "${YELLOW}安装 CUDA 11.8 版本...${NC}"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        uv pip install -e .
        echo -e "${GREEN}✓ CUDA 11.8 版本安装完成${NC}"
        ;;
    3)
        echo ""
        echo -e "${YELLOW}安装 CUDA 12.1 版本...${NC}"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        uv pip install -e .
        echo -e "${GREEN}✓ CUDA 12.1 版本安装完成${NC}"
        ;;
    4)
        echo ""
        echo -e "${YELLOW}安装基础依赖...${NC}"
        uv pip install -e .
        echo -e "${YELLOW}⚠ 请手动安装 PyTorch${NC}"
        echo "CPU: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        echo "CUDA 11.8: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        echo "CUDA 12.1: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

# 验证安装
echo ""
echo -e "${YELLOW}验证安装...${NC}"
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 安装验证成功${NC}"
else
    echo -e "${RED}✗ 安装验证失败${NC}"
    exit 1
fi

# 给运行脚本添加执行权限
chmod +x run_separation.py

echo ""
echo "=================================="
echo -e "${GREEN}安装完成！${NC}"
echo "=================================="
echo ""
echo "使用方法:"
echo "  1. 激活虚拟环境: source .venv/bin/activate"
echo "  2. 运行分离脚本: python run_separation.py <audio_file>"
echo "  3. 查看帮助: python run_separation.py --help"
echo ""
echo "示例:"
echo "  python run_separation.py song.mp3"
echo "  python run_separation.py -b music_folder/"
echo ""

