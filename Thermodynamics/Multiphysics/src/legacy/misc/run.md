# RUN.md  （Docker 运行说明）

## 1. 目录结构约定
将整个交付目录放到本机任意位置，目录名保持为：
PINN4Science/qyb_case_source_bundle



## 2. Docker 镜像
使用 NVIDIA PhysicsNeMo 镜像（与当前环境一致）：
nvcr.io/nvidia/physicsnemo/physicsnemo:25.08



## 3. 启动容器（CPU 运行）
在 Windows PowerShell / Linux / macOS 终端中进入项目根目录后执行：

Windows PowerShell：
docker run --rm -it --shm-size=1g -v ${PWD}:/workspace nvcr.io/nvidia/physicsnemo/physicsnemo:25.08 bash

Linux/macOS：
docker run --rm -it --shm-size=1g -v $(pwd):/workspace nvcr.io/nvidia/physicsnemo/physicsnemo:25.08 bash

进入容器后默认工作目录为 /workspace。

## 4. 容器内安装依赖
如果镜像内已包含依赖，可跳过。建议首次执行一次核对：

cd /workspace
pip install --no-cache-dir -r requirements.txt

依赖检查：
python -c "import torch, numpy, pandas, matplotlib, imageio; import physicsnemo; print('deps ok')"

## 5. 训练（示例）
说明：
- data_path 默认指向 data/3D_data.dat，如果不存在会跳过评估数据。
- 训练输出默认在 results/ 下，模型保存 results/pinnsformer_model.pth

示例（Sym Geometry + box 外轮廓 + CPU 小规模）：
python main.py --mode train --geometry csg --geometry_backend sym --sym_shape box \
  --box_p1 0 0 0 --box_p2 1 1 1 \
  --device cpu --epochs 2 --lr 1.0 \
  --res_points 5000 --bc_points 4000 --ic_points 2000 \
  --time_steps 5 --step_size 0.01 \
  --w_bc 1.0 --bc_outer_type dirichlet --bc_outer_value 300 \
  --bc_hole_type neumann --bc_hole_value 0

如果只想快速验证能跑通（不训练），可以先跑 quick（若 main.py 支持）：
python main.py --mode quick

## 6. 评估（可选）
使用 data_7246.txt 评估/出图（如果 evaluate.py 里实现了 run）：
python evaluate.py --data_path /workspace/data/data_7246.txt --output_dir results

## 7. 生成 GIF（GT vs PINN + Stress）
说明：
- data_7246.txt：格式为 “x y z t T”（空格分隔）
- --plot_on res：用 interior 点云（密度更像体渲染）
- --plot_on bc：用 boundary 点云（更像外表面壳）
- --ckpt：模型权重路径（默认 results/pinnsformer_model.pth）
- GIF 输出目录：--out_dir 指定

### 7.1 生成 res（体点云）版本
python -u viz_sym_geom_from_data.py \
  --device cpu \
  --data_path_gt /workspace/data/data_7246.txt \
  --out_dir results/viz_box_res \
  --plot_on res \
  --geometry csg --geometry_backend sym \
  --sym_shape box --box_p1 0 0 0 --box_p2 1 1 1 \
  --res_points 120000 --bc_points 20000 --ic_points 5000 \
  --time_steps 5 --step_size 0.01 \
  --ckpt results/pinnsformer_model.pth \
  --fps 6 \
  --point_size 22 --point_alpha 0.20 --dpi 180 \
  --cbar_mode shared --t_tol 1e-6 \
  --make_stress --cleanup_frames

输出：
- results/viz_box_res/gt_vs_pred_res.gif
- results/viz_box_res/stress_res.gif

### 7.2 生成 bc（边界点云）版本
python -u viz_sym_geom_from_data.py \
  --device cpu \
  --data_path_gt /workspace/data/data_7246.txt \
  --out_dir results/viz_box_bc \
  --plot_on bc \
  --geometry csg --geometry_backend sym \
  --sym_shape box --box_p1 0 0 0 --box_p2 1 1 1 \
  --bc_points 30000 --res_points 20000 --ic_points 5000 \
  --time_steps 5 --step_size 0.01 \
  --ckpt results/pinnsformer_model.pth \
  --fps 6 \
  --point_size 10 --point_alpha 0.20 --dpi 180 \
  --cbar_mode shared --t_tol 1e-6 \
  --make_stress --cleanup_frames

输出：
- results/viz_box_bc/gt_vs_pred_bc.gif
- results/viz_box_bc/stress_bc.gif

## 8. 常用参数（调图效果）
- 点更大更“实心”：--point_size 增大（例如 10 -> 22）
- 点更透明更“颗粒”：--point_alpha 减小（例如 0.20 -> 0.10）
- 点更密：增大 --res_points / --bc_points
- 动画速度：--fps（越大越快）

## 9. 退出容器
exit