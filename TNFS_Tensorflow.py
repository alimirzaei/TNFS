# Ternary Net Feature Selection

import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from tensorflow.python.keras.losses import mean_squared_error
import matplotlib.pyplot as plt
from utils import load_data, getSyntheticDataset
from ternary import tw_ternarize, p_ternarize




class TNFS():
    def __init__(self, hidden = 100, input_dim = 28*28, thresh = .5, l1 = 0.01, l2=.01 , ternary = True):
        self.W = []
        self.thresh = thresh
        self.input_dim = input_dim
        self.input = tf.placeholder(tf.float32, shape=(None, input_dim), name='input')
        with tf.variable_scope('MaskLayer'):
          self.x = self._O2OTernaryLayer(self.input, thresh = thresh)
        
        with tf.variable_scope('L1'):
          if(ternary):
            self.y = self._TernaryFC(self.x, hidden, thresh = thresh, name ='1')
          else:
            self.y = self._fc(self.x, hidden, name ='1')
          
          self.y = tf.nn.sigmoid(self.y)
        #x = Dense(hidden, activation='sigmoid', kernel_regularizer='l2')(self.layer1)
        #self.output = Dense(input_dim, kernel_regularizer='l2')(x)
        with tf.variable_scope('L2'):
          if(ternary):
            self.output = self._TernaryFC(self.y, input_dim, thresh= thresh, name = '2')
            self.output = self._TernaryFC(self.output, input_dim, thresh= thresh, name = '3')
            self.output = self._TernaryFC(self.output, input_dim, thresh= thresh, name = '4')
          else:
            self.output = self._fc(self.y, input_dim, name = '2')

        self.desired_output = tf.placeholder(tf.float32, shape=(None, input_dim))

        #vars_all   = tf.trainable_variables() 
        #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars_all ]) 

        #var = [v for v in tf.trainable_variables() if v.name == "MaskLayer/O2OTernary:0"][0]
        #var = [v for v in tf.trainable_variables()]
        #print(var)

        #! To Do
        # regularization on ternarize

        l1_loss=tf.reduce_mean(tf.abs(self.W[0]))
        l2_loss=tf.reduce_mean(tf.square(self.W[0]))+tf.reduce_mean(tf.square(self.W[1]))+tf.reduce_mean(tf.square(self.W[3]))

        self.loss = tf.reduce_mean(tf.sqrt(mean_squared_error(self.output, self.desired_output)))+l1*l1_loss+l2*l2_loss


        #self.loss = tf.reduce_mean(mean_squared_error(self.output, self.desired_output))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess =  tf.Session()
        self.sess.run(tf.initialize_all_variables())
        print(tf.trainable_variables())

    
    def _get_variable(self, shape, name):
      with tf.name_scope(name) as scope:
        self.W.append(tf.get_variable(name=name, shape=shape, initializer=tf.initializers.random_normal))
      return self.W[-1]

    def _O2OTernaryLayer(self, x, thresh=.5, name='O2OTernary'):
      c_in = x.get_shape().as_list()[1]
      W = self._get_variable([1, c_in], name)
      self.feature_weights = tw_ternarize(W, thresh)
      x = tf.multiply(x, self.feature_weights)
      return x

    def _TernaryFC(self, x, c_out, thresh=.5, name='Ternary_fc'):
      c_in = x.get_shape().as_list()[1]
      W = self._get_variable([c_in, c_out], name)
      b = self._get_variable([1, c_out], 'b_'+name)
      with tf.variable_scope('weights'+name):
        W = tw_ternarize(W, thresh)
      with tf.variable_scope('biases'+name):
        b = tw_ternarize(b, thresh)
      x = tf.matmul(x, W) + b
      return x

    def _fc(self, x, c_out, name='fc'):
      c_in = x.get_shape().as_list()[1]
      W = self._get_variable([c_in, c_out], name)
      b = self._get_variable([1, c_out], 'b_'+name)
      x = tf.matmul(x, W) + b
      return x
    
    def train(self, X1, X2, batch_size = 32, num_batchs = 1000):
        
        for epoch in range(num_batchs):
          idx = np.random.choice(range(len(X1)), batch_size)
          batch_in = X1[idx]
          batch_out = X2[idx]
  #           calc_obj = [train_op, loss]
          Feed_data={self.input : batch_in, self.desired_output : batch_out}
          #tw_ternarize(self.W[0], thresh)
          calc_ans=self.sess.run([self.opt, self.loss, self.feature_weights], feed_dict=Feed_data)
          sum_loss = calc_ans[1]
          s = '[epoch {}] train-loss:{:.3}'
          print(s.format(epoch,sum_loss))   
        out = self.sess.run(self.output, feed_dict={self.input : batch_in})
        return (calc_ans[2], out, batch_in)
    #def getWeights(self):
    #    with tf.variable_scope("mask", reuse=tf.AUTO_REUSE):
    #        w = self.sess.run(tf.get_variable('kernel', shape=(1,self.input_dim)))
    #    return w 


from keras.datasets import mnist
if __name__ == '__main__':
    dataset = 'synth_linear_large' # face , mnist, channel
    if(dataset == 'face'):
        data = loadmat('/home/ali/Datasets/fs/warpPIE10P.mat')
        X = data['X']/255.
        X1= X2 = X + .000001
        Y = data['Y']-1
        shape = (44,55)
        input_dim = 44*55
        hidden = 20
        thresh = .4
        l1 = 0.1
        l2 = 0
    elif(dataset == 'mnist'):
        (x_train, y_train),(_,_) = mnist.load_data()
        X = x_train.reshape(-1, 28*28)
        X = X / 255.
        X1= X2 = X+.000001
        shape = (28,28)
        input_dim = 28*28
        hidden = 20
        thresh = .5
        l1=.4
        l2=0
    elif(dataset == 'channel'): 
        X1, X2 = load_data()
        X1 = X1.reshape(len(X1),-1)
        X2 = X2.reshape(len(X2),-1)
        shape = (72, 14)
        input_dim = 72*14
        hidden = 100
        thresh = .9
        l1 = 0
        l2 =0
    elif(dataset == 'synth_linear_small'):
        X1 = getSyntheticDataset(N=10000, indep=5 , dep=4, type='linear')
        X1 = X1 / np.max(X1)
        X2 = X1
        hidden = 5
        thresh = .5
        shape = (5, 5)
        input_dim = 25
        l1= 0.1 # linear 0.1 is good
        l2 = 0
    elif(dataset == 'synth_nonlinear_small'):
        X1 = getSyntheticDataset(N=10000, indep=5 , dep=4, type='nonlinear')
        X1 = X1 / np.max(X1)
        X2 = X1
        hidden = 5
        thresh = .5
        shape = (5, 5)
        input_dim = 25
        l1= 0.12 # linear 0.1 is good
        l2 = 0
    elif(dataset == 'synth_linear_large'):
        X1 = getSyntheticDataset(N=10000, indep=5 , dep=49, type='linear')
        X1 = X1 / np.max(X1)
        X2 = X1
        hidden = 5
        thresh = .5
        shape = (5, 50)
        input_dim = 5*50
        l1= 0.1 # linear 0.1 is good
        l2 = 0
    elif(dataset == 'synth_nonlinear_large'):
        X1 = getSyntheticDataset(N=10000, indep=5 , dep=49, type='nonlinear')
        X1 = X1 / np.max(X1)
        X2 = X1
        hidden = 5
        thresh = .5
        shape = (5, 50)
        input_dim = 250
        l1= 0.8 # linear 0.1 is good
        l2 = 0

    import os
    directory = 'TNFS/'+dataset
    if not os.path.exists(directory):
        os.makedirs(directory)
    model = TNFS(input_dim=input_dim, hidden= hidden, thresh = thresh, l1=l1, l2=l2, ternary=True)
    for i in range(100):
        (w, out, inp) = model.train(X1, X2, num_batchs=500)   
        fig = plt.figure(figsize=(10,3)) 
        ax = fig.add_subplot(1,3,1)
        ax.imshow(w.reshape(shape))
        #plt.savefig('%s/%03d.jpg'%(dataset,i))
        #plt.figure()
        ax = fig.add_subplot(1,3,2)
        ax.imshow(out[0].reshape(shape))
        ax = fig.add_subplot(1,3,3)
        ax.imshow(inp[0].reshape(shape))
        fig.savefig('%s/%03d_out.jpg'%(directory,i))



        
        