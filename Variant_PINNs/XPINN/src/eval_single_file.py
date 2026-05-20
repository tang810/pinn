"""
Notebook-friendly eval file for XPINN (2D Poisson equation).

Online Jupyter usage:
1. Upload xpinn_trained_model.pt as a public asset.
2. Upload XPINN_2D_PoissonEqn.mat as a public asset (optional, simulated data used if missing).
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the XPINN model, evaluates on the 3 sub-regions,
and shows figures with plt.show(). It does not save images.
"""

import os

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.tri as tri

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import torch


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "e7e12ebd5fa84821a32b63edd24c2c9a"
MODEL_HASH = "e13586a54fb84e76862289059572f845"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/xpinn_trained_model.pt")


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
        'x_f1': Xf[idx1], 'y_f1': Yf[idx1], 'x_f2': Xf[idx2], 'y_f2': Yf[idx2],
        'x_f3': Xf[idx3], 'y_f3': Yf[idx3],
        'xi1': Xf[intf1], 'yi1': Yf[intf1], 'xi2': Xf[intf2], 'yi2': Yf[intf2],
        'xb': Xf[bdy], 'yb': Yf[bdy],
        'ub': u_exact.flatten()[bdy][:, None],
        'u_exact': u_exact.flatten()[:, None],
        'u_exact2': u_exact.flatten()[idx2][:, None],
        'u_exact3': u_exact.flatten()[idx3][:, None]
    }


def load_data(data_path, data_source='load'):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_source == 'simul' or not data_path:
        return generate_simulated_data()
    if os.path.isfile(data_path):
        return scipy.io.loadmat(data_path)
    elif os.path.isdir(data_path):
        for f in sorted(os.listdir(data_path)):
            if f.endswith('.mat'): return scipy.io.loadmat(os.path.join(data_path, f))
    return generate_simulated_data()


def prepare_training_data(data):
    x_f1=data['x_f1'].flatten()[:,None];y_f1=data['y_f1'].flatten()[:,None]
    x_f2=data['x_f2'].flatten()[:,None];y_f2=data['y_f2'].flatten()[:,None]
    x_f3=data['x_f3'].flatten()[:,None];y_f3=data['y_f3'].flatten()[:,None]
    xi1=data['xi1'].flatten()[:,None];yi1=data['yi1'].flatten()[:,None]
    xi2=data['xi2'].flatten()[:,None];yi2=data['yi2'].flatten()[:,None]
    xb=data['xb'].flatten()[:,None];yb=data['yb'].flatten()[:,None]
    ub_train=data['ub'].flatten()[:,None];u_exact=data['u_exact'].flatten()[:,None]
    X_f1=np.hstack([x_f1,y_f1]);X_f2=np.hstack([x_f2,y_f2]);X_f3=np.hstack([x_f3,y_f3])
    X_fi1=np.hstack([xi1,yi1]);X_fi2=np.hstack([xi2,yi2]);X_ub=np.hstack([xb,yb])
    return (X_f1,X_f2,X_f3,X_fi1,X_fi2,X_ub,ub_train,
            X_f1,X_f2,X_f3,X_fi1,X_fi2,u_exact)


# =========================
# 3. XPINN Model (TensorFlow v1)
# =========================

class XPINN:
    def __init__(self,X_ub,ub,X_f1,X_f2,X_f3,X_fi1,X_fi2,layers1,layers2,layers3):
        self.sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False))
        self.x_ub=X_ub[:,0:1];self.y_ub=X_ub[:,1:2];self.ub=ub
        self.x_f1=X_f1[:,0:1];self.y_f1=X_f1[:,1:2];self.x_f2=X_f2[:,0:1];self.y_f2=X_f2[:,1:2]
        self.x_f3=X_f3[:,0:1];self.y_f3=X_f3[:,1:2];self.x_fi1=X_fi1[:,0:1];self.y_fi1=X_fi1[:,1:2]
        self.x_fi2=X_fi2[:,0:1];self.y_fi2=X_fi2[:,1:2]
        self.x_ub_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_ub_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_f1_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_f1_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_f2_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_f2_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_f3_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_f3_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_fi1_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_fi1_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.x_fi2_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1]);self.y_fi2_tf=tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        self.weights1,self.biases1,self.A1=self._init_nn(layers1)
        self.weights2,self.biases2,self.A2=self._init_nn(layers2)
        self.weights3,self.biases3,self.A3=self._init_nn(layers3)
        self.ub1_pred=self._net(self.x_ub_tf,self.y_ub_tf,self.weights1,self.biases1,self.A1,"tanh")
        self.ub2_pred=self._net(self.x_f2_tf,self.y_f2_tf,self.weights2,self.biases2,self.A2,"sin")
        self.ub3_pred=self._net(self.x_f3_tf,self.y_f3_tf,self.weights3,self.biases3,self.A3,"cos")
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
            if act=="tanh":H=tf.tanh(20*A[l]*tf.add(tf.matmul(H,w[l]),b[l]))
            elif act=="sin":H=tf.sin(20*A[l]*tf.add(tf.matmul(H,w[l]),b[l]))
            elif act=="cos":H=tf.cos(20*A[l]*tf.add(tf.matmul(H,w[l]),b[l]))
        return tf.add(tf.matmul(H,w[-1]),b[-1])
    def predict(self,X1,X2,X3):
        u1=self.sess.run(self.ub1_pred,{self.x_ub_tf:X1[:,0:1],self.y_ub_tf:X1[:,1:2]})
        u2=self.sess.run(self.ub2_pred,{self.x_f2_tf:X2[:,0:1],self.y_f2_tf:X2[:,1:2]})
        u3=self.sess.run(self.ub3_pred,{self.x_f3_tf:X3[:,0:1],self.y_f3_tf:X3[:,1:2]})
        return u1,u2,u3

    def _assign_list(self, variables, values, name):
        if len(variables) != len(values):
            raise ValueError(f"{name} length mismatch: model={len(variables)}, checkpoint={len(values)}")
        assigns = [var.assign(value) for var, value in zip(variables, values)]
        self.sess.run(assigns)

    def load_pt_model(self,path):
        path=os.path.expanduser(path)
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        state=checkpoint.get("state_dict",checkpoint)
        self._assign_list(self.weights1,state["weights1"],"weights1")
        self._assign_list(self.biases1,state["biases1"],"biases1")
        self._assign_list(self.A1,state["A1"],"A1")
        self._assign_list(self.weights2,state["weights2"],"weights2")
        self._assign_list(self.biases2,state["biases2"],"biases2")
        self._assign_list(self.A2,state["A2"],"A2")
        self._assign_list(self.weights3,state["weights3"],"weights3")
        self._assign_list(self.biases3,state["biases3"],"biases3")
        self._assign_list(self.A3,state["A3"],"A3")
        print(f"Model loaded from: {path}")

    def load_model(self,path):
        path=os.path.expanduser(path)
        if path.endswith(".pt"):
            return self.load_pt_model(path)
        saver=tf.compat.v1.train.Saver();saver.restore(self.sess,path)
        print(f"Model loaded from: {path}")


# =========================
# 4. Visualization
# =========================

def show_eval_result(model,training_data):
    (X_f1,X_f2,X_f3,X_fi1,X_fi2,X_ub,ub,X1,X2,X3,Xp1,Xp2,uex)=training_data
    u_pred=model.predict(X1,X2,X3)
    x_tot=np.concatenate([X1[:,0],X2[:,0],X3[:,0]])
    y_tot=np.concatenate([X1[:,1],X2[:,1],X3[:,1]])
    u_pred_all=np.concatenate([u_pred[0],u_pred[1],u_pred[2]])
    u_exact_all=np.squeeze(uex)
    error=np.abs(u_exact_all-u_pred_all.flatten())
    trg=tri.Triangulation(x_tot,y_tot)

    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,data,title in [(axes[0],u_exact_all,"u (Exact)"),(axes[1],u_pred_all.flatten(),"u (Predicted)"),
                          (axes[2],error,"Point-wise Error")]:
        im=ax.tricontourf(trg,data,100,cmap='jet')
        ax.plot(Xp1[:,0],Xp1[:,1],'w-',lw=2);ax.plot(Xp2[:,0],Xp2[:,1],'w-',lw=2)
        ax.set_title(title);ax.set_xlabel("x");ax.set_ylabel("y");ax.grid(True,alpha=0.3)
        fig.colorbar(im,ax=ax)
    plt.tight_layout();plt.show()

    l2=np.linalg.norm(u_exact_all-u_pred_all.flatten())/np.linalg.norm(u_exact_all)
    print(f"Relative L2 error: {l2:.6e}")


# =========================
# 5. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH,data_path=DATA_PATH,data_source='load'):
    data=load_data(data_path,data_source)
    td=prepare_training_data(data)
    (X_f1,X_f2,X_f3,X_fi1,X_fi2,X_ub,ub,X1,X2,X3,Xp1,Xp2,uex)=td

    layers1=[2,30,30,1];layers2=[2,20,20,20,20,1];layers3=[2,25,25,25,1]
    print(f"Loading model from: {model_path}")
    print(f"Sub-domain sizes: 1={len(X_f1)}, 2={len(X_f2)}, 3={len(X_f3)}")

    model=XPINN(X_ub,ub,X_f1,X_f2,X_f3,X_fi1,X_fi2,layers1,layers2,layers3)
    model.load_model(model_path)

    show_eval_result(model,td)
    return {"model":model}


eval_result = evaluate()
