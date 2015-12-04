import numpy as np
import inDianajonES as InD
import tensorflow as tf


'''
ConvNNet implements a convolutional neural network 
using the TensorFlow framework. It consists of a 
ConvNNet class, which contains several functions:
    
    - Train inputs a list of images and artifacts,
        builds the design matrix, and implements
        the neural net (outputting the training)
        error as it goes
    
    - Test runs the neural net on a test data set.
        Can do whatever we make it do.
    
    - Save_model saves the session to the input filename.
    
    - Resume_from loads a saved session.
    
Also @Joe if you are wondering why I used a class, its 
because the class allowed the session to be saved as a 
global variable without it being a script.
'''

# ============================================================

class ConvNNet(object):
    '''
    ConvNNet implements a convolutional neural network 
    using the TensorFlow framework.
    '''
    def __init__(self):
        
        self.runs = None
        self.expids = None
        self.artifacts = None
        self.Nsteps = None
        
        return
        
    def load_data(self, runs, expids, artifacts, gridsize=128, cgfactor=8):
        '''
        load/make training data:  @Joe: edit as needed.  In particular
        we want X to have shape [Nexamples, Npixels], and y to 
        have shape [Nexamples , Ncategories] and be all zeros 
        along the second axis (except for the correct classification),
        which is a 1.  This is what ey2 is.
        
        Function inputs are as follows:
        
        - runs is the run info of the training data
    
        - expids are the exposure ids of the training data
    
        - artifacts is list of artifact objects (preprocessed)
        
        - gridsize is size (in pixels) of the postage stamps.
        * Note:  must be evenly divisible into 2048 *
        
        - cgfactor is coarse-graining factor.
        * Note:  must be evenly divisibile into gridsize *
        '''
        
        self.gridsize = gridsize        
        assert 2048 % self.gridsize == 0
        self.cgfactor = cgfactor        
        assert (self.gridsize//self.cgfactor)%4 == 0
        
        imagenames, bkgnames = InD.get_training_filenames(runs,expids)
        X , Y = InD.create_design_matrix(imagenames , bkgnames , artifacts , gridsize , cgfactor)
        ey = InD.enumerate_labels(Y)
     
        ey2 = np.zeros([len(ey),int(np.max(ey))],float)
        for i in range(len(ey)):
            ey2[i,ey[i]-1] = 1.0
            
        self.X = X
        self.y = ey2
        self.Nexamples = len(ey)
        self.Ncategories = int(np.max(ey))

        
    
    def Train(self, Nsteps, Nfeatures_conv1=32, Wsize_1=5, Nfeatures_conv2=64, \
                Wsize_2=5, Xlen_3=1024):
        '''
        This function creates the design matrix and loads the
        true clasifications (if they don't already exist).  
        It then runs the neural net to train the optimal 
        predicting scheme.
        
        * Currently the neural net is very similar to the
        one used in the MNIST tutorial from Tensorflow (except
        modified to use our images etc.).  We 
        should modify it further to fit our needs *
        
        Function inputs are below:
    
        - Nsteps is number of training steps to run
    
        - Nfeatures_conv1 is the number of convolution features (images)
        in the first layer
        
        - Wsize_1 is the size of the first convolution filter 
        (assumed to be square)
        
        - Nfeatures_conv2 is the number of convolution features (images) in
        the second layer

        - Wsize_2 is the size of the second convolution filter 
        (assumed to be square).
        
        - Xlen_3 is the length of the densely connected features vector.
        '''
        
        # start neural net: define x,y placeholders and create session
        #self.Session = tf.InteractiveSession()  # useful if running from notebook
        self.x = tf.placeholder("float",shape=[None,self.gridsize**2])
        self.x_image = tf.reshape(self.x,[-1,self.gridsize,self.gridsize,1])    
        self.y_ = tf.placeholder("float",shape=[None,self.Ncategories])
    
        # create first layer
        # here we create 32 new images using a convolution with a
        # 5x5x32 weights filter plus a bias (one for each new image)
        # This is equivalent to measuring 32 features for each 5x5 
        # pannel of the original image.  We'll likely want many more 
        # features, and to use more pixels.  Keep that in mind.
        self.W_conv1 = weight_variable([Wsize_1,Wsize_1,1,Nfeatures_conv1])  # play around with altering sizes
        self.b_conv1 = bias_variable([Nfeatures_conv1])# length should be same as last dimension of W_conv1
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1)+self.b_conv1)
        # split each image into 4, and obtain the maximum quadrant
        self.h_pool1 = max_pool_2x2(self.h_conv1)
    
        # create second layer
        # here each of our 32 intermediate images is convolved with
        # a 5x5x64 weights filter.  We create 64 new images by summing
        # over all 32 convolutions.  Each of the 64 images has its own bias
        # term.  The shape of the result is the shape of the original image 
        # divided by 4 on each axis by 64 (i.e. if you started with a 
        # 2048x2048 image, you now have a 512x512x64 image)
        self.W_conv2 = weight_variable([Wsize_2,Wsize_2,Nfeatures_conv1,Nfeatures_conv2]) # again, play with altering sizes
        self.b_conv2 = bias_variable([Nfeatures_conv2])          # of the first two axes
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        # split each image into 4, and obtain the maximum quadrant
        self.h_pool2 = max_pool_2x2(self.h_conv2)
    
        # Densely Connected layer
        # Here, the 7x7x64 image tensor is flattened, and we get a 
        # 1x1024 vector using the form h_fc1 = h_2 * W + b
        self.W_fc1 = weight_variable([(self.gridsize//self.cgfactor//4)**2*Nfeatures_conv2, Xlen_3])
        self.b_fc1 = bias_variable([Xlen_3])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, \
                                  (self.gridsize//self.cgfactor//4) \
                                  *(self.gridsize//self.cgfactor//4)*Nfeatures_conv2])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1)+self.b_fc1)
    
        # avoid overfitting using tensorflows dropout function.
        # specifically, we keep each component of h_fc1 with
        # probability keep_prob.
        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
    
        # finally, a softmax regression to predict the output
        self.W_fc2 = weight_variable([Xlen_3,self.Ncategories])
        self.b_fc2 = bias_variable([self.Ncategories])
    
        # output of NN
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)
        self.Session = tf.Session()
        # run the optimization.  We'll minimize the cross entropy
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,"float"))

        self.Session.run(tf.initialize_all_variables())
        
        # batch gradient descent ticker
        current_index = 0
        for i in range(Nsteps):
            # update the parameters using batch gradient descent.
            # use 50 examples per iteration (can change)
            next_set = np.arange(current_index,current_index+50,1)% self.Nexamples
            x_examples = self.X[next_set,:]
            y_examples = self.y[next_set,:]
            current_index = (current_index+50) % self.Nexamples
        
            #for every thousandth step, print the training error.
            if i%1000 ==0:
                train_accuracy = self.accuracy.eval(feed_dict={self.x:x_examples \
                             , self.y_: y_examples, self.keep_prob: 1.0},session=self.Session)
                print "step %d, training accuracy %g"%(i, train_accuracy)
        
            self.train_step.run(feed_dict={self.x: x_examples, self.y_: y_examples, self.keep_prob: 0.5},session=self.Session)


        return
    
    
    def Test(self,test_data_x,test_data_y):
        '''
        Test the current model on an input set of data
        '''
        test_accuracy = self.accuracy.eval(feed_dict={self.x:test_data_x \
                        , self.y_: test_data_y, self.keep_prob: 1.0},session=self.Session)        
        print('Test Accuracy: ', test_accuracy)
        #raise Exception('cannot test model yet \n')
        return
        
    def Save_model(self, filename, Nsteps):
        '''
        Use tensorflow's train.Saver to create checkpoint
        file.
        
        - Nsteps is number of training steps that have already
            been run.
        '''
        raise Exception('cannot save model yet \n')
        saver = tf.train.Saver()
        saver.save(self.Session, filename, global_step=Nsteps)
        return
    
    def Resume_from(self, filename):
        '''
        Use tensorflow's train.Saver to reload a saved
        checkpoint, and resume training.
        '''
        raise Exception('cannot resume training yet \n')
        saver = tf.train.Saver()
        saver.restore(self.Session, filename)
        return
        
# ------------------------------------------------------------   
'''
Neural net functions
'''
def weight_variable(shape):
    '''
    Initialize a tensorflow weight variable
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial) # note: this won't let us spread across multiple GPUs.

def bias_variable(shape):
    '''
    Initialize a tensorflow bias variable
    '''
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    '''
    Convolve a 2d image (x) with a filter (W)
    '''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    '''
    Return quadrant of image with max pixel values
    '''
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# ------------------------------------------------------------