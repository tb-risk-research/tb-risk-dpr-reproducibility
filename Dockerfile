# Dockerfile — 可复现运行环境
# 基于 Python 3.9-slim，使用 requirements.lock 安装精确依赖
#
# 用法：
#   docker build -t tb_risk:latest .
#   docker run --rm tb_risk:latest python -m pytest tests/ -v
#
# 文献: Gruning B et al. Practical computational reproducibility.
#       Cell Syst 6(6):631-635, 2018.

FROM python:3.9-slim

LABEL maintainer="tb_risk team"
LABEL description="结核病SEIR传播动力学模型 — 可复现运行环境"
LABEL version="5.0.0"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# - libgomp1: scikit-learn/torch 需要的 OpenMP 运行时
# - python3-tk: tkinter GUI 支持（utils.py 已做 try/except 可选导入，但安装可避免警告）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（使用锁文件确保版本一致）
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r requirements.lock

# 复制项目代码
COPY . /app/

# 安装项目（普通安装，非 editable）
# 2026-09-21：镜像内 editable 安装的 PEP 660 导入钩子未被运行时生效
# （ModuleNotFoundError: No module named 'tb_risk'，55 个测试文件收集失败）；
# CI test job 用升级过的 pip 同样 -e . 却正常，属 pip 版本相关的脆弱机制。
# 镜像本身不可变，editable 毫无收益——普通安装把真实包文件写入 site-packages，
# 对任何 pip 版本都稳健。
RUN pip install --no-cache-dir .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV TB_RISK_ENV=docker
# 离线优先：默认不联网探测，避免容器内超时
ENV TB_RISK_PROBE_TIMEOUT=1.0

# 热更新目录（将新模型/规则放入该目录即可热加载，无需重建镜像）
ENV TB_RISK_HOTUPDATE_DIR=/app/hotupdate
RUN mkdir -p /app/hotupdate /app/models

# 以非 root 用户运行（最小权限原则）
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# 默认入口：运行测试（覆盖为 CMD ["tb-risk"] 可启动 GUI）
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]