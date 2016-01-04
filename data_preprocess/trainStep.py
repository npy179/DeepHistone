#!/usr/bin/python
import numpy as np 
from covpoolmlp import ConvPoolMlp 
import theano
import theano.tensor as T
import numpy

def load_data(sequences, labels):
    print "loading data.............."
    sequence = np.load(sequences)#make sure the data type is float 32
    label = np.load(labels)#the label's datatype is also float32, should convert to 
    #partition the data into training data, validation data, test data
    #convert the data to format tuple(input,target)
    #input is 2D matrix, each row is a example, target is vector has same length with rows of input
    
    num_train = 1000
    num_valid = 500
    num_test = 500
    
    train_sequence = sequence[0:num_train,]
    train_label = label[0:num_train]
    train_set = (train_sequence, train_label)
    
    valid_sequence = sequence[num_train+1:num_train+num_valid,]
    valid_label = label[num_train+1:num_train+num_valid]
    valid_set = (valid_sequence, valid_label)

    test_sequence = sequence[num_train+num_valid+1:]
    test_label = label[num_train+num_valid+1:]
    test_set = (test_sequence, test_label)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))

        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)


    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval

def evaluate_convpoolmlp(learning_rate, n_epoches, sequence, labels, batch_size):
    """
    load the dataset
    """
    datasets=load_data(sequence, labels)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validating and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    
    #######################################################################
    #                         BUILD MODEL                                 #
    #######################################################################
    print ".........building the model"

    rng = numpy.random.RandomState(23455)
    #######################################################                         
    """                                                                             
    The following step build the convulotion model (3 layers)                   
    and the hyperparameters should be set resonable                                 
    """
    
    

    nkerns = [320, 480, 960]#number of kerns for each layer                         
    layer0_input = x.reshape((batch_size, 1, 600, 4)) # layer0_input is mini_batch                               
    print layer0_input.eval()
    """
    convpoolmlp = ConvPoolMlp(
        rng=rng,
        input = layer0_input,
        batch_size = batch_size,
        nkerns = nkerns,
        lamda1 = 0.0000005,
        lamda2 = 0.0000002
        ) 

    params = convpoolmlp.params
    grads = T.grad(cost, params)
    
    updates = [
        (param_i, param_i - learning_rate*grad_i)
        for param_i, grad_i in zip(params, grads)
        ]
 

    ###############################################################
    #                TRAIN FUNCTION                               #
    ###############################################################
    train_model = theano.function(
        inputs=[index],
        outputs=convpoolmlp.cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index+1)*batch_size],
            y: train_set_y[index * batch_size:(index+1)*batch_size]
            }
        )
    

    ###############################################################
    #                 TEST FUNCTION                               #
    ###############################################################
    test_model = theano.function(
        input=[index],
        outputs=convpoolmlp.LogisticRegression.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index+1)*batch_size],
            y: test_set_y[index * batch_size:(index+1)*batch_size]
            }
        )

    ###############################################################
    #               VALIDATION FUNCTION                           #
    ###############################################################
    validate_model = theano.function(
        input=[index],
        outputs=convpoolmlp.LogisticRegression.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index+1)*batch_size],
            y: valid_set_y[index * batch_size:(index+1)*batch_size]
            }
        )


    #################################################################
    #                  TRAIN MODEL                                  #
    #################################################################
    print "......training"
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.

    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    while(epoch < n_epoches) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
                print 'training @ iter = ', iter

            cost_ij = train_model(minibatch_index)
            if (iter + 1) % validation_frequency == 0:
                # use all validate data to test whether classification is right
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if patience <= iter:
                done_looping = True
                break
    
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    """

                    
if __name__=="__main__":
    evaluate_convpoolmlp(learning_rate=0.13, n_epoches=1000, sequence = "sequence_shuffle_float32_2000.npy", labels = "label_shuffle_float32_2000.npy", batch_size=500)
