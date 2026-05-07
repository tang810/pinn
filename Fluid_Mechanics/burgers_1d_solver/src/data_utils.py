from torch import sin, pi, linspace, meshgrid, hstack, zeros, vstack


def initial_u(x, t=0):
    return -sin(pi * x)


def boundary_top(t, x=-1):
    return 0 * t


def boundary_bottom(t, x=1):
    return 0 * t


def icbc_data(xi, xf, ti, tf, bc_pts, ic_pts):
    x_bc = linspace(xi, xf, bc_pts)
    t_bc = linspace(ti, tf, bc_pts)
    x_bc_m, t_bc_m = meshgrid(x_bc, t_bc, indexing="ij")

    x_ic = linspace(xi, xf, ic_pts)
    t_ic = linspace(ti, ti, ic_pts)
    x_ic_m, _ = meshgrid(x_ic, t_ic, indexing="ij")

    ic_x = x_ic_m[:, 0].unsqueeze(1)
    ic_t = zeros(ic_x.shape[0], 1)
    ic_u = initial_u(ic_x)
    ic = hstack((ic_x, ic_t))

    bc_bottom_x = x_bc_m[0, :].unsqueeze(1)
    bc_bottom_t = t_bc_m[0, :].unsqueeze(1)
    bc_bottom_u = boundary_bottom(bc_bottom_t)
    bc_bottom = hstack((bc_bottom_x, bc_bottom_t))

    bc_top_x = x_bc_m[-1, :].unsqueeze(1)
    bc_top_t = t_bc_m[-1, :].unsqueeze(1)
    bc_top_u = boundary_top(bc_top_t)
    bc_top = hstack((bc_top_x, bc_top_t))

    xt_train_bc = vstack((bc_top, bc_bottom))
    u_train_bc = vstack((bc_top_u, bc_bottom_u))

    return ic, ic_u, xt_train_bc, u_train_bc


def residual_data(xi, xf, ti, tf, nc, n_test):
    x_resid = linspace(xi, xf, nc)
    t_resid = linspace(ti, tf, nc)

    xx_resid, tt_resid = meshgrid((x_resid, t_resid), indexing="ij")
    x_resid_flat = xx_resid.transpose(1, 0).flatten().unsqueeze(1)
    t_resid_flat = tt_resid.transpose(1, 0).flatten().unsqueeze(1)

    xt_resid = hstack((x_resid_flat, t_resid_flat))
    f_hat_train = zeros((xt_resid.shape[0], 1))

    x_test = linspace(xi, xf, n_test)
    t_test = linspace(ti, tf, n_test)

    xx_test, tt_test = meshgrid((x_test, t_test), indexing="ij")
    x_test_flat = xx_test.transpose(1, 0).flatten().unsqueeze(1)
    t_test_flat = tt_test.transpose(1, 0).flatten().unsqueeze(1)

    xt_test = hstack((x_test_flat, t_test_flat))
    return xt_resid, f_hat_train, xt_test
