import os
import time
from datetime import datetime

import numpy as np
import torch

from .data_utils import icbc_data, residual_data
from .models import NN
from .precomp import inputs
from .test import gpt_test, gpt_test_loss, pinn_test, pinn_test_loss
from .train import offline_generation, pinn_train


def _load_nu_candidates(cfg):
    if cfg.data == "real":
        if os.path.exists(cfg.data_path):
            arr = np.load(cfg.data_path)
            arr = np.asarray(arr, dtype=np.float64).reshape(-1)
            arr = arr[(arr > 0.0) & (arr <= 1.0)]
            if arr.size >= 8:
                return np.unique(arr)
            print(f"[warn] real data in {cfg.data_path} is too small. Falling back to simul.")
        else:
            print(f"[warn] real data file not found: {cfg.data_path}. Falling back to simul.")
    return np.linspace(0.005, 1.0, 129)


def run(cfg, device):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"Program Start: {datetime.now()}\n")
    print(f"Device: {device}")

    xi, xf = -1.0, 1.0
    ti, tf = 0.0, 1.0

    xt_resid, f_hat, xt_test = residual_data(xi, xf, ti, tf, cfg.nc, cfg.n_test)
    xt_resid = xt_resid.to(device)
    f_hat = f_hat.to(device)
    xt_test = xt_test.to(device)

    ic_xt, ic_u, bc_xt, bc_u = icbc_data(xi, xf, ti, tf, cfg.bc_pts, cfg.ic_pts)
    ic_xt, ic_u = ic_xt.to(device), ic_u.to(device)
    bc_xt, bc_u = bc_xt.to(device), bc_u.to(device)

    b_train = _load_nu_candidates(cfg)

    layers_pinn = np.array([2, 20, 20, 20, 20, 1])

    neurons = np.zeros(cfg.number_of_neurons)
    neurons[0] = np.median(b_train)

    test_size = xt_test.shape[0]
    xt_size = xt_resid.shape[0]
    ic_size = ic_xt.shape[0]
    bc_size = bc_xt.shape[0]

    xt_resid = xt_resid.requires_grad_()
    out_full = torch.zeros((xt_size, cfg.number_of_neurons), device=device)
    out_bc = torch.zeros((bc_size, cfg.number_of_neurons), device=device)
    out_ic = torch.zeros((ic_size, cfg.number_of_neurons), device=device)
    out_t_full = torch.zeros((xt_size, cfg.number_of_neurons), device=device)
    out_x_full = torch.zeros((xt_size, cfg.number_of_neurons), device=device)
    out_xx_full = torch.zeros((xt_size, cfg.number_of_neurons), device=device)
    out_test = torch.zeros((test_size, cfg.number_of_neurons), device=device)

    num_largest_mag = int(xt_size * 0.2)
    idx_list = torch.zeros((cfg.number_of_neurons, num_largest_mag), dtype=torch.long)
    loss_list = np.zeros(cfg.number_of_neurons)
    generation_time = np.zeros(cfg.number_of_neurons)

    print("GPT-PINN Training Started")
    total_time_1 = time.time()

    train_out = train_out_x = train_out_t = train_out_xx = train_out_ic = train_out_bc = None

    for i, neuron in enumerate(neurons):
        print("*" * 60)
        b_train = np.delete(b_train, np.where(b_train == neuron)[0])

        nu = neuron
        pinn = NN(layers_pinn, nu).to(device)
        final_loss = pinn_train(
            pinn,
            xt_resid,
            ic_xt,
            ic_u,
            bc_xt,
            bc_u,
            f_hat,
            cfg.epochs_pinn,
            cfg.lr_pinn,
            cfg.tol,
        )
        print(f"PINN Final Loss: {final_loss}")

        train_out, train_out_x, train_out_t, train_out_xx, train_out_ic, train_out_bc = inputs(
            pinn,
            xt_resid,
            out_full,
            out_t_full,
            out_x_full,
            out_xx_full,
            out_ic,
            out_bc,
            ic_xt,
            bc_xt,
            i,
            out_test,
            xt_test,
            xt_size,
            num_largest_mag,
            idx_list,
        )

        # save latest PINN weights as reusable checkpoint
        torch.save(pinn.state_dict(), os.path.join(cfg.model_dir, "latest_pinn.pt"))

        if (not cfg.train_final) and (i + 1 == cfg.number_of_neurons):
            end = cfg.number_of_neurons - 1
            break

        t1 = time.time()
        largest_loss, largest_case = offline_generation(
            b_train,
            xt_size,
            ic_size,
            bc_size,
            ic_u,
            bc_u,
            train_out,
            train_out_x,
            train_out_t,
            train_out_xx,
            train_out_ic,
            train_out_bc,
            f_hat,
            cfg.epochs_gpt_train,
            cfg.lr_gpt,
            neurons[: i + 1],
        )
        t2 = time.time()

        generation_time[i] = (t2 - t1) / 60
        loss_list[i] = float(largest_loss)

        if i + 1 < cfg.number_of_neurons:
            neurons[i + 1] = largest_case

        if i + 1 == cfg.number_of_neurons:
            end = cfg.number_of_neurons
            break

    total_time = (time.time() - total_time_1) / 3600

    print("*" * 60)
    print("GPT-PINN Training Ended")
    print(f"Total training time: {total_time} Hours")
    print(f"Activation function parameters: {neurons}")

    b_test = b_train[np.random.choice(len(b_train), cfg.test_cases, replace=False)]

    print("GPT-PINN Testing Started")
    gpt_test_time, gpt_test_soln = gpt_test(
        b_test,
        xt_size,
        ic_size,
        bc_size,
        ic_u,
        bc_u,
        train_out,
        train_out_x,
        train_out_t,
        train_out_xx,
        train_out_ic,
        train_out_bc,
        f_hat,
        cfg.epochs_gpt_test,
        cfg.lr_gpt,
        neurons,
        out_test,
    )

    gpt_test_losses = gpt_test_loss(
        b_test,
        xt_size,
        ic_size,
        bc_size,
        ic_u,
        bc_u,
        train_out,
        train_out_x,
        train_out_t,
        train_out_xx,
        train_out_ic,
        train_out_bc,
        f_hat,
        cfg.epochs_gpt_test,
        cfg.lr_gpt,
        neurons,
    )

    print("PINN Testing Started")
    pinn_test_time, pinn_test_soln = pinn_test(
        b_test,
        layers_pinn,
        xt_resid,
        ic_xt,
        ic_u,
        bc_xt,
        bc_u,
        f_hat,
        cfg.epochs_pinn,
        cfg.lr_pinn,
        xt_test,
        cfg.tol,
    )

    pinn_test_losses = pinn_test_loss(
        b_test,
        layers_pinn,
        xt_resid,
        ic_xt,
        ic_u,
        bc_xt,
        bc_u,
        f_hat,
        cfg.epochs_pinn,
        cfg.lr_pinn,
        cfg.tol,
    )

    np.savetxt(os.path.join(cfg.output_dir, "generation_time.dat"), generation_time[:end])
    np.savetxt(os.path.join(cfg.output_dir, "max_losses.dat"), loss_list[:end])
    np.savetxt(os.path.join(cfg.output_dir, "neurons.dat"), neurons)
    np.savetxt(os.path.join(cfg.output_dir, "total_time.dat"), np.array([total_time]))

    np.savetxt(os.path.join(cfg.output_dir, "xt_resid.dat"), xt_resid.detach().cpu().numpy())
    np.savetxt(os.path.join(cfg.output_dir, "b_test.dat"), b_test)
    np.savetxt(os.path.join(cfg.output_dir, "xt_test.dat"), xt_test.detach().cpu().numpy())

    np.savetxt(os.path.join(cfg.output_dir, "gpt_test_losses.dat"), gpt_test_losses)
    np.savetxt(os.path.join(cfg.output_dir, "gpt_test_soln.dat"), gpt_test_soln)
    np.savetxt(os.path.join(cfg.output_dir, "gpt_test_time.dat"), gpt_test_time + total_time)

    np.savetxt(os.path.join(cfg.output_dir, "pinn_test_losses.dat"), pinn_test_losses)
    np.savetxt(os.path.join(cfg.output_dir, "pinn_test_soln.dat"), pinn_test_soln)
    np.savetxt(os.path.join(cfg.output_dir, "pinn_test_time.dat"), pinn_test_time)

    params = {
        "device": str(device),
        "domain": {"Xi": xi, "Xf": xf, "Ti": ti, "Tf": tf},
        "data_sizes": {"Nc": cfg.nc, "N_test": cfg.n_test, "BC_pts": cfg.bc_pts, "IC_pts": cfg.ic_pts},
        "tol": cfg.tol,
        "layers_pinn": layers_pinn,
        "lr_pinn": cfg.lr_pinn,
        "epochs_pinn": cfg.epochs_pinn,
        "parameter_size": len(b_train) + cfg.number_of_neurons,
        "number_of_neurons": cfg.number_of_neurons,
        "lr_gpt": cfg.lr_gpt,
        "epochs_gpt_train": cfg.epochs_gpt_train,
        "test_cases": cfg.test_cases,
        "epochs_gpt_test": cfg.epochs_gpt_test,
        "num_largest_mag": num_largest_mag,
    }
    np.save(os.path.join(cfg.output_dir, "params.npy"), params)

    torch.save({"neurons": torch.tensor(neurons, dtype=torch.float32)}, os.path.join(cfg.model_dir, "gpt_neurons.pt"))

    print(f"Program End: {datetime.now()}")
