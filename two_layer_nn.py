import numpy as np

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def loss(self, X, y, reg=0.0):
        # unpacking data
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        num_train, D = X.shape 
        scores = None

        # forward pass
        fc1 = X@W1 + b1     # fully connected
        X2 = np.maximum(0, fc1)  # ReLU
        scores = X2@W2 + b2 # fully connected
    
        # loss ---------------------------------------------------------------------
        loss = None
        scores -= np.max(scores, axis=1, keepdims=True) # avoid numeric instability
        scores_exp = np.exp(scores)
        softmax_matrix = scores_exp / np.sum(scores_exp, axis=1, keepdims=True) 
        loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]))
        loss /= num_train
        loss += reg * (np.sum(W2 * W2) + np.sum( W1 * W1 )) # regularization
        #----------------------------------------------------------------------------

        # backpropagation -----------------------------------------------------------
        grads = {}
       
        softmax_matrix[np.arange(num_train) ,y] -= 1
        softmax_matrix /= num_train

        # W2 gradient
        dW2 = (X2.T)@(softmax_matrix)   # [HxN] * [NxC] = [HxC]

        # b2 gradient
        db2 = softmax_matrix.sum(axis=0)

        # W1 gradient
        dW1 = (softmax_matrix)@(W2.T)   # [NxC] * [CxH] = [NxH]
        dfc1 = dW1 * (fc1>0)             # [NxH] . [NxH] = [NxH]
        dW1 = (X.T)@(dfc1)              # [DxN] * [NxH] = [DxH]

        # b1 gradient
        db1 = dfc1.sum(axis=0)

        # regularization gradient
        dW1 += reg * 2 * W1
        dW2 += reg * 2 * W2

        grads = {'W1':dW1, 'b1':db1, 'W2':dW2, 'b2':db2}

        return loss, grads
    
    def predict(self,X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        scores = None
      
        fc1 = X@W1 + b1 # input to fc
        X2 = np.maximum(0, fc1) # fc to relu-fc
        scores = X2@W2 + b2 # relu-fc to output
        y_pred = np.argmax(scores, axis=1) # batch of predictions
        return y_pred

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
        X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
        after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for key in self.params:
                self.params[key] -= learning_rate * grads[key]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        }