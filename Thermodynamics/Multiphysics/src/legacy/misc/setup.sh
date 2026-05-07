#!/bin/bash
# ============================================================
# setup.sh — 一键修复并安装 physicsnemo-sym (补丁增强版)
# ============================================================

set -e

# 1. 环境安全性与版本自检 (防止用户误入旧环境)
EXPECTED_ENV="physicsnemo_v3_py310"
CURRENT_ENV=$(basename "$CONDA_PREFIX")
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

echo "==== 环境安全性自检 ===="
echo "当前环境: $CURRENT_ENV (期望: $EXPECTED_ENV)"
echo "Python版本: $PYTHON_VERSION (期望: 3.10)"

if [ "$CURRENT_ENV" != "$EXPECTED_ENV" ] || [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "❌ 错误: 你似乎没有在 physicsnemo_v3_py310 (Python 3.10) 环境中运行。"
    echo "请执行以下命令重置环境:"
    echo "  conda env remove -n physicsnemo_env --all (清理旧环境)"
    echo "  conda env create -f environment.yml (根据新配置创建)"
    echo "  conda activate $EXPECTED_ENV"
    echo "  bash setup.sh"
    exit 1
fi
echo "✅ 环境校验通过，开始应用补丁..."

# 2. 寻找系统 nvcc 并创建补丁 (直接植入 Conda bin 目录，拦截优先级最高)
SYS_NVCC=$(which nvcc || echo "/usr/local/cuda-12.4/bin/nvcc")
NVCC_PATCH="$CONDA_PREFIX/bin/nvcc"
if [ ! -f "$NVCC_PATCH" ] || [ "$(grep -c "allow-unsupported-compiler" "$NVCC_PATCH")" -eq 0 ]; then
    echo "正在 Conda 目录注入 nvcc 补丁..."
    cat <<EOF > "$NVCC_PATCH"
#!/bin/bash
$SYS_NVCC -allow-unsupported-compiler "\$@"
EOF
    chmod +x "$NVCC_PATCH"
fi

# 2. 锁定编译器路径 (优先使用 Conda 12 版本)
export CC=$(which x86_64-conda-linux-gnu-cc || which gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++ || which g++)

# 3. 定义镜像源和超时时间
INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_OPTS="--default-timeout=100 -i $INDEX_URL"

echo "==== 补丁调试状态 ===="
echo "NVCC Wrapper: $(which nvcc)"
echo "GCC Version: $(gcc --version | head -n 1)"
echo "Target Source: $INDEX_URL"
echo "======================"

echo "[1/2] 准备构建依赖 (numpy<2, Cython)..."
pip install "numpy==1.26.4" Cython setuptools wheel $PIP_OPTS --force-reinstall

echo "[2/2] 编译安装 nvidia-physicsnemo-sym (通过补丁绕过检查)..."
pip install nvidia-physicsnemo-sym --no-build-isolation $PIP_OPTS

echo ""
echo "==== 最终验证 ===="
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "from physicsnemo.sym.geometry.primitives_3d import Box; print('PhysicsNeMo-Sym: OK')"
echo "==== 环境搭建大功告成！ ===="
