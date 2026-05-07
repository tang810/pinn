import time
import numpy as np
import tensorflow as tf


tf1 = tf.compat.v1
tf1.disable_eager_execution()


class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, X_bc, X_ic, settings, lb, ub, u_min, u_scale):
        self.s = settings
        self.lb = lb.astype(np.float32)
        self.ub = ub.astype(np.float32)
        self.u_min = np.float32(u_min)
        self.u_scale = np.float32(u_scale)
        self.t0_norm = np.float32((self.s.t_0 - self.u_min) / self.u_scale)

        self.x_u = X_u[:, 0:1]
        self.y_u = X_u[:, 1:2]
        self.z_u = X_u[:, 2:3]
        self.t_u = X_u[:, 3:4]
        self.u = u

        self.x_f = X_f[:, 0:1]
        self.y_f = X_f[:, 1:2]
        self.z_f = X_f[:, 2:3]
        self.t_f = X_f[:, 3:4]

        self.X_bc = X_bc
        self.X_ic = X_ic

        self.weights, self.biases = self.initialize_nn(self.s.layers)

        self.sess = tf1.Session(
            config=tf1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        )

        self.alpha_x = tf1.Variable([np.log(0.5)], dtype=tf.float32)
        self.alpha_perp = tf1.Variable([np.log(1.0)], dtype=tf.float32)

        self.x_u_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.y_u_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.z_u_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.t_u_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf1.placeholder(tf.float32, shape=[None, 1])

        self.x_f_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.y_f_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.z_f_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf1.placeholder(tf.float32, shape=[None, 1])

        self.X_bc_tf = {k: tf1.placeholder(tf.float32, shape=[None, 4]) for k in ["x0", "x1", "y0", "y1", "z0", "z1"]}
        self.X_ic_tf = tf1.placeholder(tf.float32, shape=[None, 4])

        self.u_pred = self.net_u(self.x_u_tf, self.y_u_tf, self.z_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.y_f_tf, self.z_f_tf, self.t_f_tf)

        self.loss_data = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_pde = tf.reduce_mean(tf.square(self.f_pred))
        self.loss_bc = self.net_bc_loss()
        self.loss_ic = self.net_ic_loss()

        alpha_x = tf.exp(self.alpha_x)
        alpha_perp = tf.exp(self.alpha_perp)
        eps = 1e-8
        self.loss_reg = tf.square((alpha_x - alpha_perp) / (alpha_x + alpha_perp + eps))

        self.loss = (
            self.s.lambda_data * self.loss_data
            + self.s.lambda_pde * self.loss_pde
            + self.s.lambda_bc * self.loss_bc
            + self.s.lambda_ic * self.loss_ic
            + self.s.lambda_reg * self.loss_reg
        )

        self.optimizer_lbfgs = None
        if hasattr(tf1, "contrib") and hasattr(tf1.contrib, "opt"):
            self.optimizer_lbfgs = tf1.contrib.opt.ScipyOptimizerInterface(
                self.loss,
                method="L-BFGS-B",
                options={
                    "maxiter": 50000,
                    "maxfun": 50000,
                    "maxcor": 50,
                    "maxls": 50,
                    "ftol": 1.0 * np.finfo(float).eps,
                },
            )

        global_step = tf1.Variable(0, trainable=False)
        learning_rate = tf1.train.exponential_decay(0.001, global_step, 5000, 0.9, staircase=True)
        self.optimizer_adam = tf1.train.AdamOptimizer(learning_rate)
        self.train_op_adam = self.optimizer_adam.minimize(self.loss, global_step=global_step)

        self.sess.run(tf1.global_variables_initializer())

    def initialize_nn(self, layers):
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = self.xavier_init([layers[l], layers[l + 1]])
            b = tf1.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    @staticmethod
    def xavier_init(size):
        in_dim, out_dim = size
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf1.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(len(weights) - 1):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        return tf.add(tf.matmul(H, weights[-1]), biases[-1])

    def net_u(self, x, y, z, t):
        return self.neural_net(tf.concat([x, y, z, t], 1), self.weights, self.biases)

    def net_f(self, x, y, z, t):
        alpha_x = tf.exp(self.alpha_x)
        alpha_perp = tf.exp(self.alpha_perp)

        u = self.net_u(x, y, z, t)
        u_t = tf.gradients(u, t)[0]

        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]

        u_z = tf.gradients(u, z)[0]
        u_zz = tf.gradients(u_z, z)[0]

        return u_t - (alpha_x * u_xx + alpha_perp * (u_yy + u_zz))

    @staticmethod
    def split_X(X):
        return X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]

    def bc_residual(self, X_face, face_name):
        x, y, z, t = self.split_X(X_face)
        u_norm = self.net_u(x, y, z, t)
        T = u_norm * self.u_scale + self.u_min

        T_x = tf.gradients(T, x)[0]
        T_y = tf.gradients(T, y)[0]
        T_z = tf.gradients(T, z)[0]

        rad_flux = self.s.eps * self.s.sigma * (self.s.t_amb ** 4 - T ** 4)

        if face_name == "x1":
            residual = self.s.k * T_x - self.s.q_sum
        elif face_name == "x0":
            residual = rad_flux + self.s.k * T_x
        elif face_name == "y0":
            residual = rad_flux + self.s.k * T_y
        elif face_name == "y1":
            residual = rad_flux - self.s.k * T_y
        elif face_name == "z0":
            residual = rad_flux + self.s.k * T_z
        elif face_name == "z1":
            residual = rad_flux - self.s.k * T_z
        else:
            raise ValueError("Unknown face name.")

        return residual / self.s.bc_res_scale

    def net_bc_loss(self):
        losses = []
        for face in ["x0", "x1", "y0", "y1", "z0", "z1"]:
            losses.append(tf.reduce_mean(tf.square(self.bc_residual(self.X_bc_tf[face], face))))
        return tf.add_n(losses)

    def net_ic_loss(self):
        x, y, z, t = self.split_X(self.X_ic_tf)
        u_ic = self.net_u(x, y, z, t)
        return tf.reduce_mean(tf.square(u_ic - self.t0_norm))

    def callback(self, loss, ax, aperp):
        print("L-BFGS | Loss: %.3e | a_x: %.8f | a_perp: %.8f" % (loss, np.exp(ax), np.exp(aperp)))

    def get_tf_dict(self):
        tf_dict = {
            self.x_u_tf: self.x_u,
            self.y_u_tf: self.y_u,
            self.z_u_tf: self.z_u,
            self.t_u_tf: self.t_u,
            self.u_tf: self.u,
            self.x_f_tf: self.x_f,
            self.y_f_tf: self.y_f,
            self.z_f_tf: self.z_f,
            self.t_f_tf: self.t_f,
            self.X_ic_tf: self.X_ic,
        }
        for face in ["x0", "x1", "y0", "y1", "z0", "z1"]:
            tf_dict[self.X_bc_tf[face]] = self.X_bc[face]
        return tf_dict

    def train(self, adam_iters, use_lbfgs):
        tf_dict = self.get_tf_dict()
        start_time = time.time()

        for it in range(adam_iters):
            self.sess.run(self.train_op_adam, tf_dict)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                fetches = [self.loss, self.loss_data, self.loss_pde, self.loss_bc, self.loss_ic, self.loss_reg]
                lv, ld, lp, lbc, lic, lr = self.sess.run(fetches, tf_dict)
                ax_val = np.exp(self.sess.run(self.alpha_x)).item()
                aperp_val = np.exp(self.sess.run(self.alpha_perp)).item()
                print(
                    "Adam %d | Loss: %.3e | Lu: %.3e | Lf: %.3e | Lbc: %.3e | Lic: %.3e | Lr: %.3e | ax: %.8f | aperp: %.8f | Time: %.2f"
                    % (it, lv, ld, lp, lbc, lic, lr, ax_val, aperp_val, elapsed)
                )
                start_time = time.time()

        if use_lbfgs:
            if self.optimizer_lbfgs is None:
                print("L-BFGS is unavailable in this TensorFlow build (tf.contrib missing). Skip L-BFGS.")
            else:
                self.optimizer_lbfgs.minimize(
                    self.sess,
                    feed_dict=tf_dict,
                    fetches=[self.loss, self.alpha_x, self.alpha_perp],
                    loss_callback=self.callback,
                )

    def predict(self, X_star):
        tf_dict = {
            self.x_u_tf: X_star[:, 0:1],
            self.y_u_tf: X_star[:, 1:2],
            self.z_u_tf: X_star[:, 2:3],
            self.t_u_tf: X_star[:, 3:4],
            self.x_f_tf: X_star[:, 0:1],
            self.y_f_tf: X_star[:, 1:2],
            self.z_f_tf: X_star[:, 2:3],
            self.t_f_tf: X_star[:, 3:4],
        }
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        return u_star, f_star

    def get_alphas(self):
        return np.exp(self.sess.run(self.alpha_x)).item(), np.exp(self.sess.run(self.alpha_perp)).item()
