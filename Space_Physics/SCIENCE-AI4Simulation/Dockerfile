# 使用官方的 Python 运行时作为父镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 将当前目录的内容复制到容器中的 /app 中
COPY . /app

# 安装依赖包
RUN pip config set global.index-url http://www.scien42.tech:20162 && \
      pip install --no-cache-dir -r requirements.txt --timeout 60 --trusted-host www.scien42.tech

# 设置环境变量
ENV PORT=20160
ENV base_url=
# Expose port
EXPOSE $PORT
# 运行命令
CMD ["python", "main.py"]






# FROM nvcr.io/nvidia/pytorch:23.10-py3

# WORKDIR /workspace/
# COPY . /workspace/

# ENV BASE_URL=
# ENV PORT=8002

# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
#    pip install -r requirements.txt

# # 确保你的shell脚本是可执行模式
# RUN chmod +x ./start.sh

# # 使用CMD来调用sh脚本
# CMD ["./start.sh"]