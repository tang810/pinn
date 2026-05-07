#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class XPINN:
    """XPINN模型类"""
    def __init__(self, X_ub, ub, X_f1, X_f2, X_f3, X_fi1, X_fi2, layers1, layers2, layers3):
        self.x_ub = X_ub[:, 0:1]
        self.y_ub = X_ub[:, 1:2]
        self.ub = ub

        self.x_f1 = X_f1[:, 0:1]
        self.y_f1 = X_f1[:, 1:2]
        self.x_f2 = X_f2[:, 0:1]
        self.y_f2 = X_f2[:, 1:2]
        self.x_f3 = X_f3[:, 0:1]
        self.y_f3 = X_f3[:, 1:2]
        self.x_fi1 = X_fi1[:, 0:1]
        self.y_fi1 = X_fi1[:, 1:2]
        self.x_fi2 = X_fi2[:, 0:1]
        self.y_fi2 = X_fi2[:, 1:2]

        self.layers1 = layers1
        self.layers2 = layers2
        self.layers3 = layers3
        
        # 初始化神经网络
        self.weights1, self.biases1, self.A1 = self.initialize_NN(layers1)
        self.weights2, self.biases2, self.A2 = self.initialize_NN(layers2)    
        self.weights3, self.biases3, self.A3 = self.initialize_NN(layers3)
        
        # 创建TensorFlow会话
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True
            )
        )

        
        # 创建占位符
        self.x_ub_tf = tf.compat.v1.placeholder(tf.float64,shape=[None, self.x_ub.shape[1]])

        self.y_ub_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.y_ub.shape[1]]) 
        
        self.x_f1_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.x_f1.shape[1]])
        self.y_f1_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.y_f1.shape[1]])
        self.x_f2_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.x_f2.shape[1]])
        self.y_f2_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.y_f2.shape[1]]) 
        self.x_f3_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.x_f3.shape[1]])
        self.y_f3_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.y_f3.shape[1]]) 
        self.x_fi1_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.x_fi1.shape[1]])
        self.y_fi1_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.y_fi1.shape[1]]) 
        self.x_fi2_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.x_fi2.shape[1]])
        self.y_fi2_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.y_fi2.shape[1]])         

        # 网络预测
        self.ub1_pred = self.net_u1(self.x_ub_tf, self.y_ub_tf)
        self.ub2_pred = self.net_u2(self.x_f2_tf, self.y_f2_tf)
        self.ub3_pred = self.net_u3(self.x_f3_tf, self.y_f3_tf)
        
        # 损失函数计算
        self.f1_pred, self.f2_pred, self.f3_pred, self.fi1_pred, self.fi2_pred,\
            self.uavgi1_pred, self.uavgi2_pred, self.u1i1_pred, self.u1i2_pred, self.u2i1_pred, self.u3i2_pred \
            = self.net_f(self.x_f1_tf, self.y_f1_tf, self.x_f2_tf, self.y_f2_tf, 
                        self.x_f3_tf, self.y_f3_tf, self.x_fi1_tf, self.y_fi1_tf, 
                        self.x_fi2_tf, self.y_fi2_tf)         
        
        # 三个子网络的损失
        self.loss1 = 20 * tf.reduce_mean(tf.square(self.ub - self.ub1_pred)) \
                    + tf.reduce_mean(tf.square(self.f1_pred)) + tf.reduce_mean(tf.square(self.fi1_pred))\
                    + tf.reduce_mean(tf.square(self.fi2_pred))\
                    + 20 * tf.reduce_mean(tf.square(self.u1i1_pred - self.uavgi1_pred))\
                    + 20 * tf.reduce_mean(tf.square(self.u1i2_pred - self.uavgi2_pred))
                
        self.loss2 = tf.reduce_mean(tf.square(self.f2_pred)) + tf.reduce_mean(tf.square(self.fi1_pred))\
                    + 20 * tf.reduce_mean(tf.square(self.u2i1_pred - self.uavgi1_pred))  
                            
        self.loss3 = tf.reduce_mean(tf.square(self.f3_pred)) + tf.reduce_mean(tf.square(self.fi2_pred))\
                    + 20 * tf.reduce_mean(tf.square(self.u3i2_pred - self.uavgi2_pred))                
                
        # 优化器
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(0.0008)
        self.train_op_Adam1 = self.optimizer_Adam.minimize(self.loss1) 
        self.train_op_Adam2 = self.optimizer_Adam.minimize(self.loss2)   
        self.train_op_Adam3 = self.optimizer_Adam.minimize(self.loss3)
        
        # 初始化变量
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        A = []
        num_layers = len(layers) 
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            a = tf.Variable(0.05, dtype=tf.float64)
            weights.append(W)
            biases.append(b)  
            A.append(a)
        return weights, biases, A
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        init = tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64)
        return tf.Variable(init, dtype=tf.float64)

    
    def neural_net_tanh(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X 
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(20 * A[l] * tf.add(tf.matmul(H, W), b)) 
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def neural_net_sin(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X 
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(20 * A[l] * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def neural_net_cos(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X 
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.cos(20 * A[l] * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u1(self, x, y):
        u = self.neural_net_tanh(tf.concat([x, y], 1), self.weights1, self.biases1, self.A1)
        return u
    
    def net_u2(self, x, y):
        u = self.neural_net_sin(tf.concat([x, y], 1), self.weights2, self.biases2, self.A2)
        return u
    
    def net_u3(self, x, y):
        u = self.neural_net_cos(tf.concat([x, y], 1), self.weights3, self.biases3, self.A3)
        return u
    
    def net_f(self, x1, y1, x2, y2, x3, y3, xi1, yi1, xi2, yi2):
        # 三个子网络的残差计算
        u1 = self.net_u1(x1, y1)
        u1_x = tf.gradients(u1, x1)[0]
        u1_y = tf.gradients(u1, y1)[0]
        u1_xx = tf.gradients(u1_x, x1)[0]
        u1_yy = tf.gradients(u1_y, y1)[0]
        
        u2 = self.net_u2(x2, y2)
        u2_x = tf.gradients(u2, x2)[0]
        u2_y = tf.gradients(u2, y2)[0]
        u2_xx = tf.gradients(u2_x, x2)[0]
        u2_yy = tf.gradients(u2_y, y2)[0]
        
        u3 = self.net_u3(x3, y3)
        u3_x = tf.gradients(u3, x3)[0]
        u3_y = tf.gradients(u3, y3)[0]
        u3_xx = tf.gradients(u3_x, x3)[0]
        u3_yy = tf.gradients(u3_y, y3)[0]
        
        # 界面上的计算
        u1i1 = self.net_u1(xi1, yi1)
        u1i1_x = tf.gradients(u1i1, xi1)[0]
        u1i1_y = tf.gradients(u1i1, yi1)[0]
        u1i1_xx = tf.gradients(u1i1_x, xi1)[0]
        u1i1_yy = tf.gradients(u1i1_y, yi1)[0]
        
        u2i1 = self.net_u2(xi1, yi1)
        u2i1_x = tf.gradients(u2i1, xi1)[0]
        u2i1_y = tf.gradients(u2i1, yi1)[0]
        u2i1_xx = tf.gradients(u2i1_x, xi1)[0]
        u2i1_yy = tf.gradients(u2i1_y, yi1)[0]
        
        u1i2 = self.net_u1(xi2, yi2)
        u1i2_x = tf.gradients(u1i2, xi2)[0]
        u1i2_y = tf.gradients(u1i2, yi2)[0]
        u1i2_xx = tf.gradients(u1i2_x, xi2)[0]
        u1i2_yy = tf.gradients(u1i2_y, yi2)[0]
        
        u3i2 = self.net_u3(xi2, yi2)
        u3i2_x = tf.gradients(u3i2, xi2)[0]
        u3i2_y = tf.gradients(u3i2, yi2)[0]
        u3i2_xx = tf.gradients(u3i2_x, xi2)[0]
        u3i2_yy = tf.gradients(u3i2_y, yi2)[0]
        
        # 平均值
        uavgi1 = (u1i1 + u2i1) / 2  
        uavgi2 = (u1i2 + u3i2) / 2

        # 残差
        f1 = u1_xx + u1_yy - (tf.exp(x1) + tf.exp(y1))
        f2 = u2_xx + u2_yy - (tf.exp(x2) + tf.exp(y2))
        f3 = u3_xx + u3_yy - (tf.exp(x3) + tf.exp(y3))
        
        # 界面残差连续性条件
        fi1 = (u1i1_xx + u1i1_yy - (tf.exp(xi1) + tf.exp(yi1))) - (u2i1_xx + u2i1_yy - (tf.exp(xi1) + tf.exp(yi1))) 
        fi2 = (u1i2_xx + u1i2_yy - (tf.exp(xi2) + tf.exp(yi2))) - (u3i2_xx + u3i2_yy - (tf.exp(xi2) + tf.exp(yi2))) 

        return f1, f2, f3, fi1, fi2, uavgi1, uavgi2, u1i1, u1i2, u2i1, u3i2
    
    def train(self, n_iter):
        tf_dict = {
            self.x_ub_tf: self.x_ub, 
            self.y_ub_tf: self.y_ub, 
            self.x_f1_tf: self.x_f1,
            self.y_f1_tf: self.y_f1, 
            self.x_f2_tf: self.x_f2, 
            self.y_f2_tf: self.y_f2,
            self.x_f3_tf: self.x_f3, 
            self.y_f3_tf: self.y_f3, 
            self.x_fi1_tf: self.x_fi1,
            self.y_fi1_tf: self.y_fi1, 
            self.x_fi2_tf: self.x_fi2, 
            self.y_fi2_tf: self.y_fi2
        }
        
        loss_history = []
        for it in range(n_iter):
            self.sess.run(self.train_op_Adam1, tf_dict)
            self.sess.run(self.train_op_Adam2, tf_dict)
            self.sess.run(self.train_op_Adam3, tf_dict)
            
            if it % 100 == 0:
                loss1 = self.sess.run(self.loss1, tf_dict)
                loss2 = self.sess.run(self.loss2, tf_dict)
                loss3 = self.sess.run(self.loss3, tf_dict)
                total_loss = loss1 + loss2 + loss3
                
                print(f"Iteration {it}: Loss1={loss1:.3e}, Loss2={loss2:.3e}, Loss3={loss3:.3e}, Total={total_loss:.3e}")
                loss_history.append(total_loss)
        
        return loss_history
    
    def predict(self, X_star1, X_star2, X_star3):
        u_star1 = self.sess.run(self.ub1_pred, {self.x_ub_tf: X_star1[:, 0:1], self.y_ub_tf: X_star1[:, 1:2]})  
        u_star2 = self.sess.run(self.ub2_pred, {self.x_f2_tf: X_star2[:, 0:1], self.y_f2_tf: X_star2[:, 1:2]})
        u_star3 = self.sess.run(self.ub3_pred, {self.x_f3_tf: X_star3[:, 0:1], self.y_f3_tf: X_star3[:, 1:2]})
        return u_star1, u_star2, u_star3
    
    def save_model(self, path):
        """保存模型"""
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(self.sess, path)
        print(f"模型已保存到: {save_path}")
        return save_path
    
    def load_model(self, path):
        """加载模型"""
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, path)
        print(f"模型已从 {path} 加载")