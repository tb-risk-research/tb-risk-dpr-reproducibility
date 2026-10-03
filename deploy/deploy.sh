#!/usr/bin/env bash
# =============================================================================
# tb_risk 一键部署脚本（Linux/macOS）
#
# 功能：
#   1. 创建/复用虚拟环境
#   2. 安装锁定依赖（requirements.lock）
#   3. 安装项目（-e）
#   4. 打包离线分发（可选，含模型与配置）
#   5. 校验依赖与运行自检
#
# 用法：
#   ./deploy/deploy.sh            # 标准部署
#   ./deploy/deploy.sh --package  # 部署后打包离线分发
#   ./deploy/deploy.sh --offline  # 仅构建离线产物（不联网安装）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PACKAGE=0
OFFLINE=0

for arg in "$@"; do
  case "$arg" in
    --package) PACKAGE=1 ;;
    --offline) OFFLINE=1 ;;
    *) echo "未知参数: $arg"; exit 1 ;;
  esac
done

echo "==> 项目根目录: $ROOT_DIR"

# ---- 1. 虚拟环境 ----
if [ ! -d "$VENV_DIR" ]; then
  echo "==> 创建虚拟环境: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "==> Python: $(python --version)"

# ---- 2. 安装依赖 ----
if [ "$OFFLINE" -eq 1 ]; then
  echo "==> 离线模式：跳过联网依赖安装（请使用预打包 wheel 离线安装）"
else
  echo "==> 安装锁定依赖..."
  pip install --upgrade pip
  pip install -r "$ROOT_DIR/requirements.lock"
fi

# ---- 3. 安装项目 ----
echo "==> 安装项目（开发模式）..."
pip install -e "$ROOT_DIR"

# ---- 4. 可选：打包离线分发 ----
if [ "$PACKAGE" -eq 1 ]; then
  echo "==> 打包离线分发产物..."
  python -c "from tb_risk.ops.deploy import pack_distribution; print('打包完成:', pack_distribution())"
fi

# ---- 5. 校验 ----
echo "==> 依赖状态："
python -m tb_risk.main --check || python "$ROOT_DIR/main.py" --check
echo "==> 部署完成。启动 GUI: tb-risk 或 python main.py"
