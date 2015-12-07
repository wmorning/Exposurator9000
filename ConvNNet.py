import numpy as np
import inDianajonES as InD
import tensorflow as tf
import sys

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
    def __init__(self, nimg, farts, gridsize, cgfactor, mbsize=100,
                 mbpath='/home/jderose/scratch/des/data', batchsize=1,
                 cgafactor=1):
        
        self.nimg = nimg
        self.farts = farts
        self.gridsize = gridsize
        self.cgfactor = cgfactor
        self.cgafactor = cgafactor
        self.Nmb = (self.nimg+mbsize-1)//mbsize
        self.Ncategories = 29
        self.mbpath = mbpath
        self.mbsize = mbsize
        self.batchsize = batchsize
        self.Nstepspermb = 20
        self.savefreq = 1000


    def convert_labels(self, y):

        ey = InD.enumerate_labels(y)

        ey2 = np.zeros([len(ey),self.Ncategories],float)
        for i in range(len(ey)):
            ey2[i,ey[i]-1] = 1.0
            
        return ey2

    
    def load_minibatch(self, filepath, nimg, farts, gridsize, cg, num,cg_additional=1):
        """
        Load a mini batch of images and their labels. 
        Labels need to be converted to tensorflow
        format

        inputs:
        filepath -- Path where the files are located
        nimg -- Number of images in the total batch
        farts -- Fraction of artifacts
        gridsize -- Number of pixels to a side
        cg -- Coarsegraining factor 
        num -- The minibatch number 
        cg_additional -- additional coursegraining to perform on the fly
        """
            
        X = np.load('{0}/Xs_{1}_{2}_{3}_{4}_mb{5}.npy'.format(filepath, nimg, farts, gridsize, cg, num))
        y = np.load('{0}/ys_{1}_{2}_{3}_{4}_mb{5}.npy'.format(filepath, nimg, farts, gridsize, cg, num))
        X[X==-99] = 0
        if cg_additional!=1:
            X = np.mean(np.mean(X.reshape([X.shape[0],gridsize//cg,gridsize//cg//cg_additional,cg_additional]),axis=3).T.reshape(gridsize//cg//cg_additional,gridsize//cg//cg_additional,cg_additional,X.shape[0]),axis=2).T.reshape([X.shape[0],(gridsize//cg//cg_additional)**2])
        X -= np.mean(X, axis=0)
        ey = self.convert_labels(y)

        return X, ey
    
    def Train(self, Nsteps, Nfeatures_conv1=32, Wsize_1=5, Nfeatures_conv2=64, \
                Wsize_2=5, Xlen_3=1024, gpu=False):
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
        print('Allocating placeholders')
        self.x = tf.placeholder("float",shape=[None,(self.gridsize//(self.cgfactor*self.cgafactor))**2])
        self.x_image = tf.reshape(self.x,[-1,(self.gridsize//(self.cgfactor*self.cgafactor)),(self.gridsize//(self.cgfactor*self.cgafactor)),1])    
        self.y_ = tf.placeholder("float",shape=[None,self.Ncategories])
    
        # create first layer
        # here we create 32 new images using a convolution with a
        # 5x5x32 weights filter plus a bias (one for each new image)
        # This is equivalent to measuring 32 features for each 5x5 
        # pannel of the original image.  We'll likely want many more 
        # features, and to use more pixels.  Keep that in mind.
        print('Creating first layer')
        self.W_conv1 = weight_variable([Wsize_1,Wsize_1,1,Nfeatures_conv1])  # play around with altering sizes
        self.b_conv1 = bias_variable([Nfeatures_conv1])# length should be same as last dimension of W_conv1
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1)+self.b_conv1)
        # split each image into 4, and obtain the maximum quadrant
        self.h_pool1 = max_pool_2x2(self.h_conv1)
    
        print('Creating second layer')
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
    
        print('Creating densely connected layer')
        # Densely Connected layer
        # Here, the 7x7x64 image tensor is flattened, and we get a 
        # 1x1024 vector using the form h_fc1 = h_2 * W + b
        self.W_fc1 = weight_variable([(self.gridsize//(self.cgfactor*self.cgafactor)//4)**2*Nfeatures_conv2, Xlen_3])
        self.b_fc1 = bias_variable([Xlen_3])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, \
                                  (self.gridsize//(self.cgfactor*self.cgafactor)//4) \
                                  *(self.gridsize//(self.cgfactor*self.cgafactor)//4)*Nfeatures_conv2])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1)+self.b_fc1)
    
        print('Dropout')
        # avoid overfitting using tensorflows dropout function.
        # specifically, we keep each component of h_fc1 with
        # probability keep_prob.
        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        
        print('Softmax')
        # finally, a softmax regression to predict the output
        self.W_fc2 = weight_variable([Xlen_3,self.Ncategories])
        self.b_fc2 = bias_variable([self.Ncategories])
    
        print('Setting output format')
        # output of NN
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)
        self.Session = tf.Session()
        
        print('Setting optimization parameters')
        # run the optimization.  We'll minimize the cross entropy
        #self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(self.y_conv,1e-10,1.0)))
        #self.train_step = tf.train.AdamOptimizer(1e-15, epsilon=0.1).minimize(self.cross_entropy)
        self.chisq = tf.reduce_mean(tf.pow(tf.sub(self.y_,self.y_conv),2)+1e-4)
        self.train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(self.chisq)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,"float"))

        print('Running session')
        self.Session.run(tf.initialize_all_variables())
        
        # batch gradient descent ticker
        current_index = 0
        testX, testy = self.load_minibatch(self.mbpath, self.nimg, self.farts, self.gridsize,
                                           self.cgfactor, 18, cg_additional=self.cgafactor)

        for npass in range(Nsteps):
            for i in range(self.Nmb):
                print('Minibatch {0}'.format(i))
                self.X, self.y = self.load_minibatch(self.mbpath, self.nimg, self.farts, self.gridsize,
                                                     self.cgfactor, i, cg_additional=self.cgafactor)
                for j in range(self.Nstepspermb):
                    print('Batch {0}'.format(j))
                    # update the parameters using batch gradient descent.
                    # use 50 examples per iteration (can change)
                    next_set = np.arange(current_index,current_index+self.batchsize,1)% self.mbsize
                    x_examples = self.X[next_set,:]
                    y_examples = self.y[next_set,:]
                    current_index = (current_index+self.batchsize) % self.mbsize
                    
                #for every thousandth step, print the training error.
                    if (i*self.Nstepspermb+j)%10 ==0:
                        train_accuracy = self.accuracy.eval(feed_dict={self.x:x_examples \
                                                                       , self.y_: y_examples, self.keep_prob: 1.0},session=self.Session)
                        print "step %d, training accuracy %g"%(i, train_accuracy)
                        
                    self.train_step.run(feed_dict={self.x: x_examples, self.y_: y_examples, self.keep_prob: 0.5},session=self.Session)
                    #print( 'Loss value: {0}'.format(loss_value))
                    #print( 'W_conv1: {0}'.format(self.W_conv1.eval(session=self.Session) ))
                    #print( 'b_conv1: {0}'.format(self.b_conv1.eval(session=self.Session) ))
                    #print( 'y_conv: {0}'.format(self.y_conv.eval(feed_dict={self.x:x_examples, self.y_:y_examples, self.keep_prob: 1.0}, session=self.Session) ))
                    if (i%self.savefreq==0) & (i!=0):
                        if gpu:
                            self.Save_model('Trained_Model_{0}_{1}_{2}_{3}_gpu_mb{4}.tfm'.format(self.nimg, self.farts, self.gridsize,  (self.cgfactor*self.cgafactor), i), i*self.Nstepspermb)
                        else:
                            self.Save_model('Trained_Model_{0}_{1}_{2}_{3}_mb{4}.tfm'.format(self.nimg, self.farts, self.gridsize, (self.cgfactor*self.cgafactor), i), i*self.Nstepspermb)
            
            self.Test(testX, testy)

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
        #raise Exception('cannot save model yet \n')
        saver = tf.train.Saver()
        saver.save(self.Session, filename, global_step=Nsteps)
        return
    
    def Resume_from(self, filename):
        '''
        Use tensorflow's train.Saver to reload a saved
        checkpoint, and resume training.
        '''
        raise Exception('cannot resume training yet \n')
        self.Session = tf.Session()
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
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial) # note: this won't let us spread across multiple GPUs.

def bias_variable(shape):
    '''
    Initialize a tensorflow bias variable
    '''
    initial = tf.random_normal(shape, stddev=0.1)
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

if __name__=='__main__':

    if len(sys.argv)>1:
        gpu = True
    else:
        gpu = False

    cnn = ConvNNet(1000, 1.0, 2048, 2, cgafactor=8)
    cnn.Train(100, Nfeatures_conv1=16, Nfeatures_conv2=32, Xlen_3=10, gpu=gpu)
