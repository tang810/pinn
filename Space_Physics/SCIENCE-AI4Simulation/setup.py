from setuptools import setup, find_packages

setup(
    name='AI4PDE', # 包名称
    version='0.1', # 版本号
    packages=find_packages(), # 自动发现包和子包
    install_requires=[
        # 列出所有依赖项，例如 'numpy>=1.10',
    ],
    entry_points={
        # 如果你有命令行工具，可以在这里定义
    },
    author='science42.tech',
    author_email='ximu@science42.tech',
    description='AI4PDE core team',
)
