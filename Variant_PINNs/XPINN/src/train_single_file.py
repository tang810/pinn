"""
Notebook-friendly train file for XPINN (2D Poisson equation).

Online Jupyter usage:
1. (Optional) Upload XPINN_2D_PoissonEqn.mat as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves the XPINN model as a .pt file to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: XPINN with 3 sub-networks solving Poisson eqn: u_xx + u_yy = exp(x) + exp(y)
         with exact solution u = exp(x) + exp(y).
         Sub-regions: x < 0.4, 0.4 <= x < 0.7, x >= 0.7
"""

import os
import time

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.tri as tri

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import torch



def get_device():
    """Detect device in priority: GPU (cuda) -> NPU (npu/ascend) -> CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
    except ImportError:
        return "cpu"
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return "npu"
    except (ImportError, AttributeError):
        pass
    return "cpu"

# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "e7e12ebd5fa84821a32b63edd24c2c9a"
MODEL_HASH = "e7e12ebd5fa84821a32b63edd24c2c9a"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/xpinn_trained_model.pt")


# =========================
# 2. Data generation
# =========================

def generate_simulated_data():
    N = 50; x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    u_exact = np.exp(X) + np.exp(Y)
    Xf = X.flatten()[:, None]; Yf = Y.flatten()[:, None]
    idx1 = X.flatten() < 0.4; idx2 = (X.flatten() >= 0.4) & (X.flatten() < 0.7)
    idx3 = X.flatten() >= 0.7
    bdy = (X.flatten() == 0) | (X.flatten() == 1) | (Y.flatten() == 0) | (Y.flatten() == 1)
    intf1 = np.abs(X.flatten() - 0.4) < 0.01; intf2 = np.abs(X.flatten() - 0.7) < 0.01

    return {
        'x_f1': Xf[idx1], 'y_f1': Yf[idx1],
        'x_f2': Xf[idx2], 'y_f2': Yf[idx2],
        'x_f3': Xf[idx3], 'y_f3': Yf[idx3],
        'xi1': Xf[intf1], 'yi1': Yf[intf1],
        'xi2': Xf[intf2], 'yi2': Yf[intf2],
        'xb': Xf[bdy], 'yb': Yf[bdy],
        'ub': u_exact.flatten()[bdy][:, None],
        'u_exact': u_exact.flatten()[:, None],
        'u_exact2': u_exact.flatten()[idx2][:, None],
        'u_exact3': u_exact.flatten()[idx3][:, None]
    }


def load_data(data_path, data_source='load'):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_source == 'simul' or not data_path:
        print("Using simulated data...")
        return generate_simulated_data()
    if os.path.isfile(data_path):
        print(f"Loading data: {data_path}")
        return scipy.io.loadmat(data_path)
    elif os.path.isdir(data_path):
        for f in sorted(os.listdir(data_path)):
            if f.endswith('.mat'):
                print(f"Loading data: {os.path.join(data_path, f)}")
                return scipy.io.loadmat(os.path.join(data_path, f))
    print("Warning: no .mat file found, using simulated data.")
    return generate_simulated_data()


def prepare_training_data(data):
    x_f1 = data['x_f1'].flatten()[:, None]; y_f1 = data['y_f1'].flatten()[:, None]
    x_f2 = data['x_f2'].flatten()[:, None]; y_f2 = data['y_f2'].flatten()[:, None]
    x_f3 = data['x_f3'].flatten()[:, None]; y_f3 = data['y_f3'].flatten()[:, None]
    xi1 = data['xi1'].flatten()[:, None]; yi1 = data['yi1'].flatten()[:, None]
    xi2 = data['xi2'].flatten()[:, None]; yi2 = data['yi2'].flatten()[:, None]
    xb = data['xb'].flatten()[:, None]; yb = data['yb'].flatten()[:, None]
    ub_train = data['ub'].flatten()[:, None]
    u_exact = data['u_exact'].flatten()[:, None]

    X_f1 = np.hstack([x_f1, y_f1]); X_f2 = np.hstack([x_f2, y_f2])
    X_f3 = np.hstack([x_f3, y_f3])
    X_fi1 = np.hstack([xi1, yi1]); X_fi2 = np.hstack([xi2, yi2])
    X_ub = np.hstack([xb, yb])
    X_star1 = X_f1; X_star2 = X_f2; X_star3 = X_f3
    X_fi1_plot = X_fi1; X_fi2_plot = X_fi2
    return (X_f1, X_f2, X_f3, X_fi1, X_fi2, X_ub, ub_train,
            X_star1, X_star2, X_star3, X_fi1_plot, X_fi2_plot, u_exact)


# =========================
# 3. XPINN Model (TensorFlow v1)
# =========================

class XPINN:
    def __init__(self, X_ub, ub, X_f1, X_f2, X_f3, X_fi1, X_fi2, layers1, layers2, layers3):
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
        self.x_ub=X_ub[:,0:1];self.y_ub=X_ub[:,1:2];self.ub=ub
        self.x_f1=X_f1[:,0:1];self.y_f1=X_f1[:,1:2];self.x_f2=X_f2[:,0:1];self.y_f2=X_f2[:,1:2]
        self.x_f3=X_f3[:,0:1];self.y_f3=X_f3[:,1:2];self.x_fi1=X_fi1[:,0:1];self.y_fi1=X_fi1[:,1:2]
        self.x_fi2=X_fi2[:,0:1];self.y_fi2=X_fi2[:,1:2]
        self.layers1=layers1;self.layers2=layers2;self.layers3=layers3

        # Placeholders
        self.x_ub_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_ub_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_f1_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_f1_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_f2_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_f2_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_f3_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_f3_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_fi1_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_fi1_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_fi2_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_fi2_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])

        # Build networks
        self.weights1,self.biases1,self.A1=self._init_nn(layers1)
        self.weights2,self.biases2,self.A2=self._init_nn(layers2)
        self.weights3,self.biases3,self.A3=self._init_nn(layers3)

        # Predictions
        self.ub1_pred=self._net(self.x_ub_tf,self.y_ub_tf,self.weights1,self.biases1,self.A1,"tanh")
        self.ub2_pred=self._net(self.x_f2_tf,self.y_f2_tf,self.weights2,self.biases2,self.A2,"sin")
        self.ub3_pred=self._net(self.x_f3_tf,self.y_f3_tf,self.weights3,self.biases3,self.A3,"cos")

        # PDE residuals
        f1,f2,f3,fi1,fi2,uavgi1,uavgi2,u1i1,u1i2,u2i1,u3i2 = self._net_f()

        # Losses
        self.loss1 = (
            20 * tf.reduce_mean(tf.square(self.ub - self.ub1_pred))
            + tf.reduce_mean(tf.square(f1))
            + tf.reduce_mean(tf.square(fi1))
            + tf.reduce_mean(tf.square(fi2))
            + 20 * tf.reduce_mean(tf.square(u1i1 - uavgi1))
            + 20 * tf.reduce_mean(tf.square(u1i2 - uavgi2))
        )
        self.loss2 = (
            tf.reduce_mean(tf.square(f2))
            + tf.reduce_mean(tf.square(fi1))
            + 20 * tf.reduce_mean(tf.square(u2i1 - uavgi1))
        )
        self.loss3 = (
            tf.reduce_mean(tf.square(f3))
            + tf.reduce_mean(tf.square(fi2))
            + 20 * tf.reduce_mean(tf.square(u3i2 - uavgi2))
        )

        opt=tf.compat.v1.train.AdamOptimizer(0.0008)
        self.train_op1=opt.minimize(self.loss1);self.train_op2=opt.minimize(self.loss2);self.train_op3=opt.minimize(self.loss3)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _xavier_init(self,size):
        return tf.Variable(tf.random.truncated_normal(size,stddev=np.sqrt(2/(size[0]+size[1])),dtype=tf.float64),dtype=tf.float64)

    def _init_nn(self,layers):
        w=[];b=[];A=[]
        for l in range(len(layers)-1):
            w.append(self._xavier_init([layers[l],layers[l+1]]))
            b.append(tf.Variable(tf.zeros([1,layers[l+1]],dtype=tf.float64),dtype=tf.float64))
            A.append(tf.Variable(0.05,dtype=tf.float64))
        return w,b,A

    def _net(self,x,y,w,b,A,act):
        H=tf.concat([x,y],1)
        for l in range(len(w)-1):
            if act=="tanh": H=tf.tanh(20*A[l]*tf.add(tf.matmul(H,w[l]),b[l]))
            elif act=="sin": H=tf.sin(20*A[l]*tf.add(tf.matmul(H,w[l]),b[l]))
            elif act=="cos": H=tf.cos(20*A[l]*tf.add(tf.matmul(H,w[l]),b[l]))
        return tf.add(tf.matmul(H,w[-1]),b[-1])

    def _net_f(self):
        def pde_res(xn,yn,net_fn):
            u=net_fn(xn,yn);ux=tf.gradients(u,xn)[0];uy=tf.gradients(u,yn)[0]
            uxx=tf.gradients(ux,xn)[0];uyy=tf.gradients(uy,yn)[0]
            return uxx+uyy-(tf.exp(xn)+tf.exp(yn))

        def residual_and_grads(xn,yn,net_fn):
            u=net_fn(xn,yn);ux=tf.gradients(u,xn)[0];uy=tf.gradients(u,yn)[0]
            uxx=tf.gradients(ux,xn)[0];uyy=tf.gradients(uy,yn)[0]
            return u,uxx+uyy-(tf.exp(xn)+tf.exp(yn))

        u1_fn=lambda x,y:self._net(x,y,self.weights1,self.biases1,self.A1,"tanh")
        u2_fn=lambda x,y:self._net(x,y,self.weights2,self.biases2,self.A2,"sin")
        u3_fn=lambda x,y:self._net(x,y,self.weights3,self.biases3,self.A3,"cos")

        f1=pde_res(self.x_f1_tf,self.y_f1_tf,u1_fn)
        f2=pde_res(self.x_f2_tf,self.y_f2_tf,u2_fn)
        f3=pde_res(self.x_f3_tf,self.y_f3_tf,u3_fn)

        u1i1,res1i1=residual_and_grads(self.x_fi1_tf,self.y_fi1_tf,u1_fn)
        u2i1,res2i1=residual_and_grads(self.x_fi1_tf,self.y_fi1_tf,u2_fn)
        u1i2,res1i2=residual_and_grads(self.x_fi2_tf,self.y_fi2_tf,u1_fn)
        u3i2,res3i2=residual_and_grads(self.x_fi2_tf,self.y_fi2_tf,u3_fn)

        fi1=res1i1-res2i1;fi2=res1i2-res3i2
        uavgi1=(u1i1+u2i1)/2;uavgi2=(u1i2+u3i2)/2
        return f1,f2,f3,fi1,fi2,uavgi1,uavgi2,u1i1,u1i2,u2i1,u3i2

    def train(self,n_iter):
        fd={self.x_ub_tf:self.x_ub,self.y_ub_tf:self.y_ub,self.x_f1_tf:self.x_f1,self.y_f1_tf:self.y_f1,
            self.x_f2_tf:self.x_f2,self.y_f2_tf:self.y_f2,self.x_f3_tf:self.x_f3,self.y_f3_tf:self.y_f3,
            self.x_fi1_tf:self.x_fi1,self.y_fi1_tf:self.y_fi1,self.x_fi2_tf:self.x_fi2,self.y_fi2_tf:self.y_fi2}
        history=[]
        for it in range(n_iter):
            self.sess.run(self.train_op1,fd);self.sess.run(self.train_op2,fd);self.sess.run(self.train_op3,fd)
            if it%100==0:
                l1,l2,l3=self.sess.run([self.loss1,self.loss2,self.loss3],fd);total=l1+l2+l3
                print(f"Iter {it:5d}/{n_iter} | L1={l1:.3e} L2={l2:.3e} L3={l3:.3e} Total={total:.3e}")
                history.append(total)
        return history

    def predict(self,X1,X2,X3):
        u1=self.sess.run(self.ub1_pred,{self.x_ub_tf:X1[:,0:1],self.y_ub_tf:X1[:,1:2]})
        u2=self.sess.run(self.ub2_pred,{self.x_f2_tf:X2[:,0:1],self.y_f2_tf:X2[:,1:2]})
        u3=self.sess.run(self.ub3_pred,{self.x_f3_tf:X3[:,0:1],self.y_f3_tf:X3[:,1:2]})
        return u1,u2,u3

    def export_state(self):
        return {
            "layers1": self.layers1,
            "layers2": self.layers2,
            "layers3": self.layers3,
            "weights1": self.sess.run(self.weights1),
            "biases1": self.sess.run(self.biases1),
            "A1": self.sess.run(self.A1),
            "weights2": self.sess.run(self.weights2),
            "biases2": self.sess.run(self.biases2),
            "A2": self.sess.run(self.A2),
            "weights3": self.sess.run(self.weights3),
            "biases3": self.sess.run(self.biases3),
            "A3": self.sess.run(self.A3),
        }

    def save_model(self,path,history=None):
        path=os.path.expanduser(path)
        if not path.endswith(".pt"):
            path=os.path.splitext(path)[0]+".pt"
        os.makedirs(os.path.dirname(path),exist_ok=True)
        checkpoint={
            "format":"tensorflow_v1_xpinn_numpy_weights",
            "project":"XPINN",
            "state_dict":self.export_state(),
            "history":history if history is not None else [],
        }
        torch.save(checkpoint,path)
        print(f"Model saved to: {path}")
        return path

    def load_model(self,path):
        path=os.path.expanduser(path);saver=tf.compat.v1.train.Saver();saver.restore(self.sess,path)
        print(f"Model loaded from: {path}")


# =========================
# 4. Visualization
# =========================

def show_train_result(model, training_data, history):
    (X_f1,X_f2,X_f3,X_fi1,X_fi2,X_ub,ub,X1,X2,X3,Xp1,Xp2,uex)=training_data
    u_pred=model.predict(X1,X2,X3)

    if len(history):
        plt.figure(figsize=(7,4))
        plt.plot(range(0,len(history)*100,100),history,'b-o',markersize=4,linewidth=2)
        plt.xlabel("Iter");plt.ylabel("Total Loss");plt.yscale("log")
        plt.title("Training Loss History");plt.grid(alpha=0.3);plt.tight_layout();plt.show()

    x_tot=np.concatenate([X1[:,0],X2[:,0],X3[:,0]])
    y_tot=np.concatenate([X1[:,1],X2[:,1],X3[:,1]])
    u_pred_all=np.concatenate([u_pred[0],u_pred[1],u_pred[2]])
    u_exact_all=np.squeeze(uex)
    error=np.abs(u_exact_all-u_pred_all.flatten())
    trg=tri.Triangulation(x_tot,y_tot)

    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,data,title in [(axes[0],u_exact_all,"u (Exact)"),(axes[1],u_pred_all.flatten(),"u (Predicted)"),(axes[2],error,"Point-wise Error")]:
        im=ax.tricontourf(trg,data,100,cmap='jet')
        ax.plot(Xp1[:,0],Xp1[:,1],'w-',lw=2);ax.plot(Xp2[:,0],Xp2[:,1],'w-',lw=2)
        ax.set_title(title);ax.set_xlabel("x");ax.set_ylabel("y");ax.grid(True,alpha=0.3)
        fig.colorbar(im,ax=ax)
    plt.tight_layout();plt.show()

    l2=np.linalg.norm(u_exact_all-u_pred_all.flatten())/np.linalg.norm(u_exact_all)
    print(f"Relative L2 error: {l2:.6e}")


# =========================
# 5. Train entry
# =========================

def train(data_path=DATA_PATH,model_path=MODEL_PATH,data_source='load',n_iter=2000):
    t0=time.time()
    data=load_data(data_path,data_source)
    td=prepare_training_data(data)

    layers1=[2,30,30,1];layers2=[2,20,20,20,20,1];layers3=[2,25,25,25,1]
    (X_f1,X_f2,X_f3,X_fi1,X_fi2,X_ub,ub,X1,X2,X3,Xp1,Xp2,uex)=td
    print(f"Sub-domain sizes: 1={len(X_f1)}, 2={len(X_f2)}, 3={len(X_f3)}")
    print(f"Training {n_iter} iterations...")

    model=XPINN(X_ub,ub,X_f1,X_f2,X_f3,X_fi1,X_fi2,layers1,layers2,layers3)
    history=model.train(n_iter)
    model_path=model.save_model(model_path,history)

    print("Training finished")
    print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time()-t0:.1f}s")

    show_train_result(model,td,history)
    return {"model":model,"model_path":model_path,"history":history}


train_result = train()
