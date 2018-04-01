"""
This script performs a one-shot kill attack on the maltese dog vs fish classifier which the feature extractions are done via inception-v3

ashafahi @ March 11 2018
"""
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy import misc
from os import listdir
from tensorflow.python.platform import gfile
from scipy.spatial.distance import cdist
from datetime import datetime
import os





def load_images_from_directory(Specie, directory):
    """
    Returns an numpy array of the images in a folder directory
    Parameters
    ----------
    Specie : string
        just the name of the class - used for reporting
    directory : string
        directory where the image files are in there (jpeg or any other format)
    Returns
    -------
    res: ndarray
        all of the images in the directory dumped into a numpy array
    """
    res = []
    for file in listdir(directory):
        thisOne = imageio.imread(directory+file)
        res.append(thisOne)
    res = np.array(res)
    print('Done loading %d %s\'s !'%(len(res),Specie))
    return res

def clean_data(X):
    """
    this method takes the data input images and removes those that do not have a 3rd dimension
     to prevent issues with inception and returns the clean numpy array
    Parameters
    ----------
    X : ndarray
        images all in one huge numpy array
    Returns
    -------
    res: ndarray
        all of the images without the ones which have less than 3 dimensions
    """

    indices = []
    for i,d in enumerate(X):
        if d.ndim !=3:
            indices.append(i)
            print('removing index %d with shape:'%i,d.shape)
    
    if len(indices) > 0:
        newX = np.delete(X,indices)
    else:
        newX = X
    return newX



def create_graph(graphDir=None):
    """"Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    # if graph directory is not given, it is the default
    if graphDir == None:
        graphDir = './inceptionModel/inception-2015-12-05/classify_image_graph_def.pb'
    with tf.Session() as sess:
        with gfile.FastGFile(graphDir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    return sess.graph

def get_feat_reps(X,class_t):
    """
    Returns the feature representation of some images by looking at the penultimate layer of inception-v3
    Parameters
    ----------
    X : ndarray
        input images all put in a numpy array
    class_t : string
        class of the images which we are doing feature extractions for.
        Note that this is only used for printing summary of progress. So just give it some
        random name if you don't care
    Returns
    -------
    res: ndarray
        feature represntation of the input images X. should have same length of X
    """
    #parameters
    feat_tensor_name = 'pool_3:0'
    input_tensor_name = 'DecodeJpeg:0'

    #get a session and create the graph
    sess = tf.Session()
    create_graph()

    #get needed tensors
    feat_tensor = sess.graph.get_tensor_by_name(feat_tensor_name)
    input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

    #get the feature representations
    res = []
    for i,x in enumerate(X):
        res.append(sess.run(feat_tensor, feed_dict={input_tensor:x}))
        if i % 50 == 0:
            print('finished %d\'th example of %s'%(i,class_t))
    res = np.array(res)

    #rest graph and close session to free memory
    tf.reset_default_graph()
    sess.close()
    return res

def id_duplicates_of_training_from_test(X_test,X_training, threshold = 3.5):
    """
    Returns the ids for the duplicates of training in test
    Parameters
    ----------
    X_test : ndarray
        the feature represenatons of the test data.
    X_training : ndarray
        the feature represenatons of the training data
    threshold : float
        threshold for reporting the similarity
    Returns
    -------
    ids : list of integer
        The difference in feature space measure by the 2-norm

    """
    list_ind = []
    for i in range(len(X_test)):
        distsToTargs = cdist(np.expand_dims(X_test[i], axis = 0), X_training)
        # print distsToTargs
        report_inds = np.argwhere(distsToTargs <= threshold)
        if len(report_inds) > 0:
            print report_inds
            print(distsToTargs[0][report_inds])
        if len(np.argwhere(distsToTargs == 0.)) > 0:
            list_ind.append(i)
    print("number of test examples removed due to having duplicates in training data is:%d"%len(list_ind))
    return list_ind


def load_bottleNeckTensor_data(directory=None, saveEm=False, random_state=123, train_size=900):
    """
    Returns the train-test splits of images and their feature representations.
    Parameters
    ----------
    directory : string, optional
        directory that the feature representations and image numpy formats are saved.
    saveEm : Boolean, optional
        whether to save the training and test data on disk or not
    random_state : integer, optional
        random seed used in train_test_split for splitting the training and test data
    train_size : integer, optional
        the number of elements in the training data for each of the classes. The remaining
        would be assigned to the test data
    Returns
    -------
    X_tr_feats, X_tst_feats, X_tr_inp, X_tst_inp, Y_tr, Y_tst : ndarray
        Arrays used for training.
    """
    #some parameters 
    directorySaving = './Data/XY/' #directory to save the X and Ys
    allDogs = 'dogInput.npy'
    allFishes  = 'fishInput.npy'
    dog_X_feats = 'dogFeats.npy'
    fish_X_feats = 'fishFeats.npy'
    if directory != None:
        dog_X_feats = directory + dog_X_feats
        fish_X_feats = directory + fish_X_feats
        allDogs = directory + allDogs
        allFishes = directory + allFishes

    
    #load the data
    dog_x_feats = np.load(dog_X_feats)
    fish_x_feats = np.load(fish_X_feats)
    allFishes = np.load(allFishes)
    allDogs = np.load(allDogs)
    
    #do train and test split number of training dogs and number of training fishes = 800 from each class
    x_d_tr, x_d_tst, y_d_tr, y_d_tst, inp_d_tr, inp_d_tst = train_test_split(dog_x_feats, np.zeros(len(dog_x_feats)), allDogs ,train_size=train_size, random_state=random_state)
    x_f_tr, x_f_tst, y_f_tr, y_f_tst, inp_f_tr, inp_f_tst = train_test_split(fish_x_feats, np.ones(len(fish_x_feats)),allFishes, train_size=train_size, random_state=random_state)
    
    assert len(x_d_tr) + len(x_d_tst) == len(dog_x_feats), "There is some issue with the spliting"
    assert len(inp_d_tr) + len(inp_d_tst) == len(dog_x_feats), "There's issues with splitting of the input images - maybe there is an issue with the raw images"
    
    #concatenate all of the X's
    X_tr_feats = np.squeeze(np.concatenate((x_d_tr, x_f_tr), axis=0))
    X_tst_feats = np.squeeze(np.concatenate((x_d_tst, x_f_tst), axis=0))
    X_tr_inp = np.squeeze(np.concatenate((inp_d_tr, inp_f_tr), axis=0))
    X_tst_inp = np.squeeze(np.concatenate((inp_d_tst, inp_f_tst), axis=0))
    #make a Y vector
    Y_tr = np.concatenate((y_d_tr,y_f_tr),axis=0)
    Y_tst = np.concatenate((y_d_tst,y_f_tst),axis=0)

    #remove the duplicates of the test data which are already present in the training data
    ids_for_test_removal = id_duplicates_of_training_from_test(X_test=X_tst_feats,X_training=X_tr_feats, threshold = 3.5)
    print ids_for_test_removal
    #sort the ids in descending order
    ids_for_test_removal.sort(reverse=True)
    for k in ids_for_test_removal:
        X_tst_feats = np.delete(X_tst_feats,k,axis=0)
        X_tst_inp = np.delete(X_tst_inp,k,axis=0)
        Y_tst = np.delete(Y_tst,k,axis=0)

    
    all_datas = ['X_tr_feats', 'X_tst_feats', 'X_tr_inp', 'X_tst_inp', 'Y_tr', 'Y_tst']
    if saveEm:
        if not os.path.exists(directorySaving):
            os.makedirs(directorySaving)
        for d in all_datas:
            np.save(directorySaving+d+'.npy',eval(d))
    
    return X_tr_feats, X_tst_feats, X_tr_inp, X_tst_inp, Y_tr, Y_tst

def adam_one_step(sess,grad_op,m,v,t,currentImage,featRepTarget,tarFeatRepPL,inputCastImgTensor,learning_rate,beta_1=0.9, beta_2=0.999, eps=1e-8):
    t += 1
    grad_t = np.squeeze(np.array(sess.run(grad_op, feed_dict={inputCastImgTensor: currentImage, tarFeatRepPL:featRepTarget})))
    m = beta_1 * m + (1-beta_1)*grad_t
    v = beta_2 * v + (1-beta_2)*grad_t*grad_t
    m_hat = m/(1-beta_1**t)
    v_hat = v/(1-beta_2**t)
    currentImage -= learning_rate*m_hat/(np.sqrt(v_hat)+eps)
    return currentImage,m,v,t

def do_forward(sess,grad_op,inputCastImgTensor, currentImage,featRepCurrentImage,featRepTarget,tarFeatRepPL,learning_rate=0.01):
    """helper function doing the forward step in the FWD-BCKWD splitting algorithm"""
    grad_now = sess.run(grad_op, feed_dict={inputCastImgTensor: currentImage, tarFeatRepPL:featRepTarget})      #evaluate the gradient at the current point
    currentImage = currentImage - learning_rate*np.squeeze(np.array(grad_now))                                  #gradient descent
    return currentImage                                                                                         #get the new current point

def do_backward(baseInpImage,currentImage,coeff_sim_inp,learning_rate,eps=0.1,do_clipping=True,inf_norm=False):
    """helper function doing the backward step in the FWD-BCKWD splitting algorithm"""
    if inf_norm:
        back_res = baseInpImage + np.maximum(np.minimum(currentImage - baseInpImage,eps) ,-eps)
    else:
        back_res = (coeff_sim_inp*learning_rate*baseInpImage + currentImage)/(coeff_sim_inp*learning_rate + 1)
    if do_clipping:
        back_res = np.clip(back_res,0,255)
    return back_res

def do_optimization(targetImg, baseImg, MaxIter=200,coeffSimInp=0.25, saveInterim=False, imageID=0):
    """
    Returns the poison image and the difference between the poison and target in feature space.
    Parameters
    ----------
    targetImg : ndarray
        the input image of the target from the  test set.
    baseImg : ndarray
        the input image of the base class (this should have a differet class than the target)
    MaxIter : integer
        this is the maximum number of fwd backward iterations
    coeffSimInp : flaot
        the coefficient of similarity to the base image in input image space relative to the 
        similarity to the feature representation of the target when everything is normalized
        the objective function of the optimization is:
                || f(x)-f(t) ||^2 + coeffSimInp * || x-b ||^2
    Returns
    -------
    old_image, finalDiff : ndarray, float
        The poison in uin8 format
        The difference in feature space measure by the 2-norm
    """

    #parameters:
    Adam = False
    objThreshold = 2.9              #the threshold for the objective function: whenever we go below this, we stop - measure of dist in feat space
    decayCoef = 0.5                 #decay coeffiencet of learning rate
    learning_rate = 500.0*255      #iniital learning rate for optimiz
    stopping_tol = 1e-10            #for the relative change
    EveryThisNThen = 20             #for printing reports
    M = 40                          #used for getting the average of last M objective function values
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
    INPUT_TENSOR_NAME = 'DecodeJpeg:0'

    #calculations for getting a reasonable value for coefficient of similarity of the input to the base image
    bI_shape = np.squeeze(baseImg).shape
    coeff_sim_inp = coeffSimInp*(2048/float(bI_shape[0]*bI_shape[1]*bI_shape[2]))**2
    print('coeff_sim_inp is:', coeff_sim_inp)

    #load the inception v3 graph
    sess = tf.Session()
    graph = create_graph()

    #add some of the needed operations
    featRepTensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME+':0')
    inputImgTensor = sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
    inputCastImgTensor = graph.get_tensor_by_name('Cast:0')#'ResizeBilinear:0')
    tarFeatRepPL = tf.placeholder(tf.float32,[None,2048])
    forward_loss = tf.norm(featRepTensor - tarFeatRepPL)
    grad_op = tf.gradients(forward_loss, inputCastImgTensor)

    #initializations
    last_M_objs = []
    rel_change_val = 1e5
    baseImg = sess.run(inputCastImgTensor, feed_dict={inputImgTensor: baseImg})         #get cast:0 output of input base image
    targetFeatRep = sess.run(featRepTensor, feed_dict={inputImgTensor: targetImg})      #get the feature reprsentation of the target
    old_image = baseImg                                                                 #set the poison's starting point to be the base image
    old_featRep = sess.run(featRepTensor, feed_dict={inputCastImgTensor: baseImg})      #get the feature representation of current poison
    old_obj = np.linalg.norm(old_featRep - targetFeatRep) + coeff_sim_inp*np.linalg.norm(old_image - baseImg)
    last_M_objs.append(old_obj)

    #intializations for ADAM
    if Adam:
        m = 0.
        v = 0.
        t = 0

    #optimization being done here
    for iter in range(MaxIter):
        #save images every now and then
        if iter % EveryThisNThen == 0:
            the_diffHere = np.linalg.norm(old_featRep - targetFeatRep)      #get the diff
            theNPimg = old_image                                            #get the image
            print("iter: %d | diff: %.3f | obj: %.3f"%(iter,the_diffHere,old_obj))
            print(" (%d) Rel change =  %0.5e   |   lr = %0.5e |   obj = %0.10e"%(iter,rel_change_val,learning_rate,old_obj))
            if saveInterim:
                name = '%d_%d_%.5f.jpeg'%(imageID,iter,the_diffHere)
                misc.imsave('./interimPoison/'+name, np.squeeze(old_image).astype(np.uint8))
            # plt.imshow(np.squeeze(old_image).astype(np.uint8))
            # plt.show()

        # forward update gradient update
        if Adam:
            new_image,m,v,t = adam_one_step(sess=sess,grad_op=grad_op,m=m,v=v,t=t,currentImage=old_image,featRepTarget=targetFeatRep,tarFeatRepPL=tarFeatRepPL,inputCastImgTensor=inputCastImgTensor,learning_rate=learning_rate)
        else:
            new_image = do_forward(sess=sess,grad_op=grad_op,inputCastImgTensor=inputCastImgTensor, currentImage=old_image,featRepCurrentImage=old_featRep,featRepTarget=targetFeatRep,tarFeatRepPL=tarFeatRepPL,learning_rate=learning_rate)
        
        # The backward step in the forward-backward iteration
        new_image = do_backward(baseInpImage=baseImg,currentImage=new_image,coeff_sim_inp=coeff_sim_inp,learning_rate=learning_rate,eps=0.1)
        
        # check stopping condition:  compute relative change in image between iterations
        rel_change_val =  np.linalg.norm(new_image-old_image)/np.linalg.norm(new_image)
        if (rel_change_val<stopping_tol) or (old_obj<=objThreshold):
            break

        # compute new objective value
        new_featRep = sess.run(featRepTensor, feed_dict={inputCastImgTensor: new_image})
        new_obj = np.linalg.norm(new_featRep - targetFeatRep) + coeff_sim_inp*np.linalg.norm(new_image - baseImg)
        
        if Adam:
            learning_rate = 0.1*255.
            old_image = new_image
            old_obj = new_obj
            old_featRep = new_featRep
        else:

            avg_of_last_M = sum(last_M_objs)/float(min(M,iter+1)) #find the mean of the last M iterations
            # If the objective went up, then learning rate is too big.  Chop it, and throw out the latest iteration
            if  new_obj >= avg_of_last_M and (iter % M/2 == 0):
                learning_rate *= decayCoef
                new_image = old_image
            else:
                old_image = new_image
                old_obj = new_obj
                old_featRep = new_featRep
                
            if iter < M-1:
                last_M_objs.append(new_obj)
            else:
                #first remove the oldest obj then append the new obj
                del last_M_objs[0]
                last_M_objs.append(new_obj)
            if iter > MaxIter:
                m = 0.
                v = 0.
                t = 0
                Adam = True

    finalDiff = np.linalg.norm(old_featRep - targetFeatRep)
    print('final diff: %.3f | final obj: %.3f'%(finalDiff,old_obj))
    #close the session and reset the graph to clear memory
    sess.close()
    tf.reset_default_graph()

    return np.squeeze(old_image).astype(np.uint8), finalDiff



def closest_to_target_from_class(classBase,targetFeatRep,allTestFeatReps, allTestClass):
    """
    Returns an index within the allTestFeatReps matrix for which is the closes to the target in feature spacee and belongs to the base class

    Parameters
    ----------
    classBase : int
        the class for the base class - we want the target to be misclassified as this class.
    targetFeatRep : ndarray
        the feature representation of the target image (2048 for inception-v3)
    allTestFeatReps : ndarray
        feature reprsentation of all the test data
    allTestClass : ndarray
        array contatining the class for all of the test data. In a binary classification task, it would be an array of 0s and 1s
    Returns
    -------
    ind_min : int
        The index of the poison base in test data. The poison base is from the base class and has the smallest distance to the target in feature space
        The difference in feature space measured by the 2-norm
    """
    if allTestFeatReps.ndim > 2: #if needed, squeeze the feat rep
        allTestFeatReps = np.squeeze(allTestFeatReps)
    assert allTestFeatReps.ndim == 2, 'the feat rep matrix should have 2 dimensions it has %d dimensions...'%allTestFeatReps.ndim
    
    possible_indices = np.argwhere(allTestClass == classBase)
    featRepCandidantes = np.squeeze(allTestFeatReps[possible_indices])
    
    #calculate distance from the target to the candidates:
    print(featRepCandidantes.ndim,targetFeatRep.ndim)
    Dists = cdist(featRepCandidantes,np.expand_dims(targetFeatRep,axis=0))
    min_ind = Dists.argmin()
    print('distance from base to target in feat space:',Dists[min_ind])
    
    return possible_indices[min_ind][0]


def encode_one_hot(nclasses,y):
    return np.eye(nclasses)[y.astype(int)]

def iterate_mini_batches(X_input,Y_input,batch_size):
    n_train = X_input.shape[0]
    for ndx in range(0, n_train, batch_size):
        yield X_input[ndx:min(ndx + batch_size, n_train)], Y_input[ndx:min(ndx + batch_size, n_train)]

def train_last_layer_of_inception(targetFeatRep,poisonInpImage,poisonClass,X_tr,Y_tr,Y_validation,X_validation,cold=True):
    """
    This function does training for the last layer of inception-v3. It either performs a cold start in which it starts from a pre-saved graph or
    does a warm start during which it again starts from a presaved graph but the pre-saved graph is for an already pretrained net.

    Parameters
    ----------
    targetFeatRep : ndarray of type float32
        the feature representation of the target.
    poisonInpImage : ndarray of type uint8
        the input image for the poison . This will be fed to the pool_3 input
    poisonClass : int
        the class of the poison - this should be the correct label for the poison/the class that we would like our target to be from
    X_tr : ndarray, float32
        array containing all of the training data. This is the feature representation of the data. It would have dim: n_t X 2048 for inception-v3
    Y_tr : ndarray, int
        array containing the class labels of the training data. the dimensions would be n_t. All the values in the array are 0s or 1s
    Y_validation: ndarray, int
        similar to the Y_tr but contatining the test data
    X_validation: ndarray, float32
        similar to X_tr but contatining the test data
    cold: Boolean
        if True, the evaluation will be based using cold start and not pretrained weights.
        if False, the evaluation would be using warm start. It starts from a set of weights that are pretrained.    

    Returns
    -------
    ind_min : int
        The index of the poison base in test data. The poison base is from the base class and has the smallest distance to the target in feature space
        The difference in feature space measured by the 2-norm
    """
    # parameters
    learning_rate = 0.01 #note that this learning rate is for only the cold start. If doing warm start, we will be using the last learning rate used during pretraining the weights (0.01 by default)
    mini_batch_size = 32
    epocs = 100
    classes = ['dog','fish']
    how_many_training_steps = 10000
    eval_step_interval = 100
    
    #fixing target shapes
    Y_target = np.ones((1,2))
    Y_target[0,int(poisonClass)] = 0.
    Y_poison = np.zeros((1,2))
    Y_poison[0,int(poisonClass)] = 1.
    targetFeatRep = targetFeatRep.reshape(1,len(targetFeatRep))
    print("Y_target is:",Y_target)
    
    #do initializations
    tf.reset_default_graph()                                    #reset the default graph to free up memory
    sess = tf.Session()                                         #get session
    random_permutation = np.arange(len(X_tr)+1)                 #for training - we add one to the number of training data to account for the poison
    
    #based on whether we are doing cold start or warm start, load the appropriate graph
    if cold: 
        saver = tf.train.import_meta_graph('./dog_v_fish_cold_graph/dog_v_fish_cold_graph.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./dog_v_fish_cold_graph/'))
        reportWeightChanges = False
    else:
        saver = tf.train.import_meta_graph('./dog_v_fish_hot_graph/dog_v_fish_hot_graph.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./dog_v_fish_hot_graph/'))
        reportWeightChanges = True
 
    graph = tf.get_default_graph()
    
    #getting feature representation of the poison - we need to get this first before adding it to the training data
    feat_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    input_tensor = sess.graph.get_tensor_by_name('DecodeJpeg:0')#'Cast:0')
    poisonFeatRep = np.expand_dims(np.squeeze(sess.run(feat_tensor, feed_dict={input_tensor: poisonInpImage})), axis=0)
    
    #append the training data and add the poison to the end of it
    X_tr = np.vstack((X_tr,poisonFeatRep))
    Y_tr = np.append(Y_tr,poisonClass)
    if X_tr.ndim > 2:
        X_tr = np.squeeze(n_dim)

    # getting required tensors or making required ops as needed.
    X_Bottleneck = sess.graph.get_tensor_by_name('X_bottleneck:0')
    Y_true = sess.graph.get_tensor_by_name('Y_true:0')
    Ylogits = sess.graph.get_tensor_by_name('logits:0')
    biasvar = sess.graph.get_tensor_by_name('final_biases:0')
    weightsvar = sess.graph.get_tensor_by_name('final_weights:0')
    cross_entropy = sess.graph.get_tensor_by_name('cross_entropy_mean_2class:0')
    evaluation_step = sess.graph.get_tensor_by_name('eval_step_2class:0')
    print("********>>>>>>",sess.run(evaluation_step,feed_dict={X_Bottleneck: X_validation,Y_true: encode_one_hot(len(classes), Y_validation)}))
    if not cold:
        train_step = sess.graph.get_operation_by_name('Adam')
        print("hot start")
    else:
        print("cold start")
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy) #, var_list=[biasvar, weightsvar] # GradientDescent
        sess.run(tf.global_variables_initializer())             #also intialize the variables if doing cold start. they are random or somewhat random numbers
    correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_true, 1))
    class_probs = tf.nn.softmax(Ylogits)


    print("********>>>>>>",sess.run(evaluation_step,feed_dict={X_Bottleneck: X_validation,Y_true: encode_one_hot(len(classes), Y_validation)}))
    # doing the training
    n_train = X_tr.shape[0]
    i=0
    if reportWeightChanges:
        allWeights = []
        allbiases = []
    for epoch in range(epocs):
        if reportWeightChanges:
            allWeights.append(sess.run(weightsvar))
            allbiases.append(sess.run(biasvar))

        shuffledRange = np.random.permutation(n_train)
        y_one_hot_train = encode_one_hot(len(classes), Y_tr)
        y_one_hot_validation = encode_one_hot(len(classes), Y_validation)
        shuffledX = X_tr[shuffledRange,:]
        shuffledY = y_one_hot_train[shuffledRange]
        
        for Xi, Yi in iterate_mini_batches(shuffledX, shuffledY, mini_batch_size):
            sess.run(train_step, feed_dict={X_Bottleneck: Xi, Y_true: Yi})
            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == how_many_training_steps)

            if (i % eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy],feed_dict={X_Bottleneck: Xi,Y_true: Yi})
                validation_accuracy = sess.run(evaluation_step,feed_dict={X_Bottleneck: X_validation,Y_true: y_one_hot_validation})
                print('%s: Step %d: Train accuracy = %.1f%%, Cross entropy = %f, Validation accuracy = %.2f%%' %
                    (datetime.now(), i, train_accuracy * 100, cross_entropy_value, validation_accuracy * 100))
            i+=1

    if epocs > 0:
        test_accuracy = sess.run(evaluation_step, feed_dict={X_Bottleneck: X_validation,Y_true:y_one_hot_validation })
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

    target_class_probs = sess.run(class_probs,feed_dict={X_Bottleneck:targetFeatRep, Y_true:Y_target})
    target_corr_pred = sess.run(correct_prediction, feed_dict={X_Bottleneck:targetFeatRep, Y_true:Y_target})

    poison_class_probs = sess.run(class_probs,feed_dict={X_Bottleneck:poisonFeatRep, Y_true:Y_poison})
    poison_corr_pred = sess.run(correct_prediction, feed_dict={X_Bottleneck:poisonFeatRep, Y_true:Y_poison})
    

    print('The target is now classified correctly:',target_corr_pred,'class probs:',target_class_probs)
    print('The poison is now classified correctly:',poison_corr_pred,'class probs:',poison_class_probs)

    print('Dist in feat space:',np.linalg.norm(targetFeatRep-poisonFeatRep))

    if reportWeightChanges:
        allWeights = np.array(allWeights)
        allbiases = np.array(allbiases)
        return target_class_probs, target_corr_pred, poison_class_probs, poison_corr_pred, allWeights, allbiases
    else:
        return  target_class_probs, target_corr_pred, poison_class_probs, poison_corr_pred
