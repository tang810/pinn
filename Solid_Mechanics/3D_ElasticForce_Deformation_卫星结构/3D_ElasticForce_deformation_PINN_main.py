import torch
import argparse
import os
from src.runner import (
    run_numerical_solver,
    run_pinn_training,
    run_quick_inference,
    plot_results_3d,
    export_numerical_results,
    export_training_results,
    export_all_results
)
from src.data_utils import (
    load_user_data,
    generate_virtual_data,
    save_data_info,
    extract_data_parameters,
    validate_data
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main(mode="train", device="cpu", epochs=1000, n_interior=5000, n_boundary=600, 
         n_test_points=50000, E=1.0, mu=0.25, lr=1e-3, output_dir="results/3D_elasticity",
         data_mode="simul", data_path=None):

    print("=== 3D弹性力学问题求解器 ===")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 数据处理逻辑
    user_data = None
    data_source = 'generated'
    use_user_data_only = False

    # 判断数据源
    if data_mode == "load":
        # 尝试加载用户数据
        if data_path and os.path.exists(data_path):
            print(f"📂 尝试加载用户数据文件: {data_path}")
            user_data = load_user_data(data_path)

            if user_data and validate_data(user_data):
                data_source = 'user_file'
                use_user_data_only = True

                # 从用户数据中提取参数，完全覆盖命令行参数
                E, mu, n_interior, n_boundary = extract_data_parameters(user_data, E, mu)
                print(f"✅ 用户数据加载成功")
                print(f"   提取参数: E={E}, μ={mu}, 内部点={n_interior}, 边界点={n_boundary}")
            else:
                print("❌ 用户数据加载失败或验证不通过，切换到虚拟数据模式")
                data_mode = "simul"  # 切换到虚拟数据模式
        else:
            if data_path is None:
                print("❌ 选择了load模式但未提供数据文件路径，切换到虚拟数据模式")
            else:
                print(f"❌ 数据文件不存在: {data_path}，切换到虚拟数据模式")
            data_mode = "simul"  # 切换到虚拟数据模式

    # 如果选择虚拟数据模式或从load模式切换过来
    if data_mode == "simul":
        print("🔄 使用虚拟仿真数据...")
        user_data = generate_virtual_data(n_interior, n_boundary, E, mu)
        data_source = 'simulation'
        use_user_data_only = False
        print(f"   生成参数: E={E}, μ={mu}, 内部点={n_interior}, 边界点={n_boundary}")

    # 根据模式运行相应的求解器
    if mode == "numerical":
        print("\n🔹 数值求解模式")

        numerical_result = run_numerical_solver(
            n_interior=n_interior, 
            n_boundary=n_boundary, 
            E=E, 
            mu=mu
        )
        export_numerical_results(numerical_result, output_dir=f"{output_dir}")
        print(f"📁 数值解结果已保存")

    elif mode == "train":
        print("\n🔹 训练模式")

        # 检查函数是否支持user_data参数
        try:
            training_result = run_pinn_training(
                n_interior=n_interior,
                n_boundary=n_boundary,
                epochs=epochs,
                device=device,
                E=E,
                mu=mu,
                lr=lr,
                save_model=True,
                user_data=user_data if use_user_data_only else None
            )
        except TypeError:
            # 如果不支持user_data参数，则不传递
            print("⚠️  训练函数不支持user_data参数，使用提取的参数")
            training_result = run_pinn_training(
                n_interior=n_interior,
                n_boundary=n_boundary,
                epochs=epochs,
                device=device,
                E=E,
                mu=mu,
                lr=lr,
                save_model=True
            )
        export_training_results(training_result, output_dir=f"{output_dir}")
        print(f"📁 训练结果已保存")

    elif mode == "quick":
        print("\n🔹 快速推理模式")

        # 数值求解部分
        #print("📊 运行数值求解...")
        numerical_result = run_numerical_solver(
            n_interior=n_interior, 
            n_boundary=n_boundary, 
            E=E, 
            mu=mu
        )

        # PINN推理部分
        #print("🧠 运行PINN推理...")
        try:
            pinn_result = run_quick_inference(
                device=device, 
                n_test_points=n_test_points,
                user_data=user_data if use_user_data_only else None
            )
        except TypeError as e:
            pinn_result = run_quick_inference(
                device=device, 
                n_test_points=n_test_points
            )

        if pinn_result:
            export_all_results(numerical_result, pinn_result, output_dir=f"{output_dir}")

            # 可视化主要结果
            #print("🎨 生成可视化结果...")
            coords = pinn_result['coordinates']
            plot_results_3d(pinn_result['u_pred'], coords, 'PINN U-Displacement')
            plot_results_3d(pinn_result['stress_s1'], coords, 'PINN Stress σ11')
            plot_results_3d(pinn_result['stress_s3'], coords, 'PINN Stress σ33')
            print("✅ 快速推理完成")
        else:
            print("❌ 错误: PINN推理失败")

    else:
        raise ValueError(f"不支持的模式: {mode}. 请选择 'numerical', 'train', 或 'quick'")

    #print(f"\n=== 求解完成，数据源: {data_source}，结果保存在: {output_dir} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Elasticity PINN Main Runner")

    # 数据源控制参数 (简化为单一参数)
    parser.add_argument("--data", type=str, default="simul", 
                       choices=["load", "simul"],
                       help="数据模式: load(加载用户数据), simul(使用虚拟数据)")
    parser.add_argument("--data_path", type=str, default=None, 
                       help="用户数据文件路径 (仅在 --data load 时使用，支持 .csv, .txt, .npz 格式)")

    # 基本参数
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["numerical", "train", "quick"], 
                       help="运行模式: numerical(数值解), train(训练), quick(快速推理)")
    parser.add_argument("--device", type=str, default="cpu", help="使用设备: cpu 或 cuda")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")

    # 网格参数 (仅当使用虚拟数据时有效)
    parser.add_argument("--n_interior", type=int, default=5000, help="内部采样点数量 (仅虚拟数据有效)")
    parser.add_argument("--n_boundary", type=int, default=600, help="边界采样点数量 (仅虚拟数据有效)")
    parser.add_argument("--n_test_points", type=int, default=50000, help="测试点数量")

    # 物理参数 (仅当用户数据不包含时作为默认值)
    parser.add_argument("--E", type=float, default=1.0, help="弹性模量 (用户数据优先)")
    parser.add_argument("--mu", type=float, default=0.25, help="泊松比 (用户数据优先)")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")

    args = parser.parse_args()

    # 检查设备可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，自动切换到CPU")
        args.device = "cpu"

    # 参数验证和提示
    if args.data == "load" and args.data_path is None:
        print("⚠️  选择了load模式但未提供--data_path参数")
        print("   程序将自动切换到虚拟数据模式")

    if args.data == "load" and args.data_path and not os.path.exists(args.data_path):
        print(f"⚠️  指定的数据文件不存在: {args.data_path}")
        print("   程序将自动切换到虚拟数据模式")

    main(
        mode=args.mode,
        device=args.device,
        epochs=args.epochs,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_test_points=args.n_test_points,
        E=args.E,
        mu=args.mu,
        lr=args.lr,
        output_dir=args.output_dir,
        data_mode=args.data,
        data_path=args.data_path
    )