import numpy as np
import matplotlib.pyplot as plt
import data

def binlogreg_train(X, Y_):
    '''
        Argumenti
            X: podaci, np.array NxD
            Y_: indeksi razreda, np.array Nx1
        Povratne vrijednosti
            w, b: parametri logisticke regresije, Nx1, 1x1
    '''
    # gradijentni spust (param_niter iteracija)
    N = len(X)
    D = len(X[0])
    
    w = np.random.randn(D, 1)
    b = 0
    
    param_niter = 10000
    param_delta = 0.1
    
    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, w)+b             # N x 1
        
        # vjerojatnosti razreda c_1
        probs =  1 / (1+np.exp(-scores))    # N x 1, omotati sigmoidom; to je h
        
        # gubitak
        loss  = 0                           # scalar, negativni log probsa
        for j in range (N):
            if (Y_[j] == 0):
                loss += -np.log(1-probs[j])
            else:
                loss += -np.log(probs[j])
        loss /= N

        
        # dijagnostički ispis
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))


        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_             # N x 1, dL/ds, probs - (y == 1)

        
        # gradijenti parametara
        grad_w = np.dot(X.T, dL_dscores)/N  # D x 1, dL/ds * x
        grad_b = np.sum(dL_dscores)/N       # 1 x 1, dL/ds, suma
        
        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b
    return w, b


def binlogreg_classify (X, w, b):
    return 1 / (1+np.exp(-(np.dot(X, w)+b)))


def binlogreg_decfun(w,b):
    def classify(X):
        return binlogreg_classify(X, w,b)
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    # instantiate the dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the logistic regression model
    w, b = binlogreg_train(X, Y_)
      
    # evaluate the model on the train set
    probs = binlogreg_classify(X, w, b)
    
    # recover the predicted classes Y
    Y = []
    for i in probs:
        if (i >= 0.5):                  # odabiremo klasu koja je vjerojatnija
            Y.append(1)
        else:
            Y.append(0)
    Y = np.vstack(Y)                    # da bude dimenzije Nx1

    # evaluate and print performance measures
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)

    Y_2 = []                            # dodano da pretvori Nx1 matrice u 1xN nizove
    for i in range(len(Y_)):
        Y_2.append(int(Y_[i][0]))
    Y_2 = np.array(Y_2)
    probs2 = np.ravel(probs)
    
    AP = data.eval_AP(Y_2[probs2.argsort()])     
    print (accuracy, recall, precision, AP)
    Y_ = np.vstack(Y_)                              


    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
          
    # graph the data points
    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()
    



