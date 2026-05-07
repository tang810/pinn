import numpy as np
import matplotlib.pyplot as plt
import scipy.io 
from scipy.interpolate import griddata

def visualize_results(ref_file, result_file,folder_path):
    print("🧠 初始化 PINN 模型...")
    # ref_file = './data/Re1e5_backStep_L8.mat'
    reference = scipy.io.loadmat(ref_file)
    X_ref = reference['X_ref']
    Y_ref = reference['Y_ref']
    u_ref = reference['U_ref']
    v_ref = reference['V_ref']
    p_ref = reference['P_ref']
    p_ref = p_ref - np.nanmean(p_ref)
    k_ref = reference['K_ref']
    o_ref = reference['O_ref']
    Z_ref = np.zeros_like(Y_ref)
    #-------------------------------------------------
    # load  result
    # -------------------------------------------------
    # Load the .mat file
    # result_file = './result_uvpdata/f1_result_epoch200000.mat'
    RESULT = scipy.io.loadmat(result_file)
    
    # Convert to double precision floating point
    u_pred = RESULT['U_pred'].astype(np.float64)
    v_pred = RESULT['V_pred'].astype(np.float64)
    p_pred = RESULT['P_pred'].astype(np.float64)

    # Adjust p_pred by subtracting the mean
    p_pred = p_pred - np.mean(p_pred)

    # Calculate the errors and flip upside down
    u_err = np.flipud(np.abs(u_pred - u_ref))
    v_err = np.flipud(np.abs(v_pred - v_ref))
    p_err = np.flipud(np.abs(p_pred - p_ref))

    # 计算l2误差
    L2_u = np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref)
    L2_v = np.linalg.norm(v_pred - v_ref) / np.linalg.norm(v_ref)
    L2_p = np.linalg.norm(p_pred[~np.isnan(p_ref)] - p_ref[~np.isnan(p_ref)]) / np.linalg.norm(p_ref[~np.isnan(p_ref)])


    # Define mesh for grid interpolation
    x1 = np.linspace(-4.0, 4.0, 257)
    y1 = np.linspace(0.0, 8.0, 257)
    X_ref1, Y_ref1 = np.meshgrid(x1, y1)

    # Grid interpolation for reference and predicted data
    U_ref1 = griddata((X_ref.ravel(), Y_ref.ravel()), u_ref.ravel(), (X_ref1, Y_ref1), method='linear')
    V_ref1 = griddata((X_ref.ravel(), Y_ref.ravel()), v_ref.ravel(), (X_ref1, Y_ref1), method='linear')
    P_ref1 = griddata((X_ref.ravel(), Y_ref.ravel()), p_ref.ravel(), (X_ref1, Y_ref1), method='linear')

    U_pred1 = griddata((X_ref.ravel(), Y_ref.ravel()), u_pred.ravel(), (X_ref1, Y_ref1), method='linear')
    V_pred1 = griddata((X_ref.ravel(), Y_ref.ravel()), v_pred.ravel(), (X_ref1, Y_ref1), method='linear')
    P_pred1 = griddata((X_ref.ravel(), Y_ref.ravel()), p_pred.ravel(), (X_ref1, Y_ref1), method='linear')

    U_err1 = griddata((X_ref.ravel(), Y_ref.ravel()), u_err.ravel(), (X_ref1, Y_ref1), method='linear')
    V_err1 = griddata((X_ref.ravel(), Y_ref.ravel()), v_err.ravel(), (X_ref1, Y_ref1), method='linear')
    P_err1 = griddata((X_ref.ravel(), Y_ref.ravel()), p_err.ravel(), (X_ref1, Y_ref1), method='linear')

    # Boundary condition adjustments
    for it1 in range(len(x1)):
        for it2 in range(len(y1)):
            if x1[it1] < 0 and y1[it2] < 1.0:
                U_ref1[it2, it1] = np.nan
                V_ref1[it2, it1] = np.nan
                P_ref1[it2, it1] = np.nan
                U_pred1[it2, it1] = np.nan
                V_pred1[it2, it1] = np.nan
                P_pred1[it2, it1] = np.nan
                U_err1[it2, it1] = np.nan
                V_err1[it2, it1] = np.nan
                P_err1[it2, it1] = np.nan
    print("📊 生成预测结果...")
    # Plotting
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # U velocity plots
    axs[0, 0].contourf(X_ref1, Y_ref1, U_ref1, 50, cmap='jet')
    axs[0, 0].set_title('u_{ref}')
    axs[0, 1].contourf(X_ref1, Y_ref1, U_pred1, 50, cmap='jet')
    axs[0, 1].set_title('u_{NN}')
    axs[0, 2].contourf(X_ref1, Y_ref1, U_err1, 50, cmap='jet')
    axs[0, 2].set_title('|u_{ref} - u_{NN}|')

    # V velocity plots
    axs[1, 0].contourf(X_ref1, Y_ref1, V_ref1, 50, cmap='jet')
    axs[1, 0].set_title('v_{ref}')
    axs[1, 1].contourf(X_ref1, Y_ref1, V_pred1, 50, cmap='jet')
    axs[1, 1].set_title('v_{NN}')
    axs[1, 2].contourf(X_ref1, Y_ref1, V_err1, 50, cmap='jet')
    axs[1, 2].set_title('|v_{ref} - v_{NN}|')

    # Pressure plots
    axs[2, 0].contourf(X_ref1, Y_ref1, P_ref1, 50, cmap='jet')
    axs[2, 0].set_title('p_{ref}')
    axs[2, 1].contourf(X_ref1, Y_ref1, P_pred1, 50, cmap='jet')
    axs[2, 1].set_title('p_{NN}')
    axs[2, 2].contourf(X_ref1, Y_ref1, P_err1, 50, cmap='jet')
    axs[2, 2].set_title('|p_{ref} - p_{NN}|')

    # plt.show()
    plt.savefig(f'{folder_path}/uvp_comparison.png', dpi=300)

    # Plotting L2 error
    # print(len(RESULT['L2_u']))
    # epochs_uv = np.arange(1, len(RESULT['L2_u']) + 1)
    
    l2_u = RESULT['L2_u'].ravel()
    l2_v = RESULT['L2_v'].ravel()
    epochs_uv = np.arange(1, len(l2_u) + 1)
    # print(epochs_uv)
    # print(len(l2_u))
    # print(l2_u.shape)
    plt.figure()
    plt.plot(epochs_uv * 100, l2_u, '-', color='b', linewidth=1, label='u')
    plt.plot(epochs_uv * 100, l2_v, '-', color='m', linewidth=1, label='v')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0.01,10])
    plt.xlabel('Epoch')
    plt.ylabel('Relative $L_2$ error')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{folder_path}/l2_error_uv.png', dpi=300)


    # Plotting loss functions
    loss_e1 = RESULT['Loss_eq1'].ravel()
    loss_e2 = RESULT['Loss_eq2'].ravel()
    loss_e3 = RESULT['Loss_eq3'].ravel()
    loss_e4 = RESULT['Loss_eq4'].ravel()
    epochs = np.arange(1, len(loss_e1) + 1)
    

    plt.figure(figsize=(10, 6))
    plt.plot(epochs * 100, loss_e1, '-', color='b', linewidth=1, label='$x$ momentum')
    plt.plot(epochs * 100, loss_e2, '-', color='m', linewidth=1, label='$y$ momentum')
    plt.plot(epochs * 100, loss_e3, '-', color='g', linewidth=1, label='mass conservation')
    plt.plot(epochs * 100, loss_e4, '-', color='k', linewidth=1, label='EVM constraint')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-8,1e-1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{folder_path}/loss_functions.png', dpi=300)
    print("✅ 推理完成")
















































