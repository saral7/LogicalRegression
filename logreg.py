import numpy as np
import matplotlib.pyplot as plt
import data


def logreg_train (X, Y_):
    '''
        Argumenti
            X: NxD primjeri
            Y_:NxC, 1 na stupcu tocne klase
        Povratne vrijednosti
            W: CxD, tezine hipoteze svake klase
            b: Cx1
    '''
    N = len(X)
    D = len(X[0])
    C = max((Y_.flatten()))+1
    
    param_niter = 7000
    param_delta = 0.5

    Ymat = []
    for i in range (C):
        row = [0]*C
        row[i] = 1
        for j in range (N//C):
            Ymat.append(row)
    Ymat = np.reshape(Ymat, (N, C))     # Ymat = matrica koja Y_ pretvara u matricu NxC gdje je za svaki redak jedinica u stupcu pripadajuce klase, koristi se u dL_dscores


    # inicijalizacija W i b
    W = np.random.randn(C, D)
    b = np.vstack(np.array([0]*C))


    # gradijentni spust
    for i in range(param_niter):
        scores = np.dot(X, W.T)+np.vstack(np.array([np.hstack(b)]*N))  # NxC, svakom retku jos dodamo b
        expscores = np.exp(scores)

        # nazivnici softmaxa
        sumexp = np.sum(expscores, axis = 1)        # Nx1

        # primjenjeni softmax, vjerojatnosti
        probs = expscores / sumexp[:, None]         # NxC
        logprobs = np.log(probs)                    # NxC
        

        # gubitak
        loss = 0
        for j in range (N):
            loss += -logprobs[j][Y_[j]]
        loss /= N
        
        # dijagnostički ispis
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))


        # derivacija gubitka po mjerama klasifikacije
        dL_dscores = probs - Ymat         # Gs, NxC

        # gradijenti parametara
        grad_W = np.dot(dL_dscores.T, X)/N                  # CxD
        grad_b = np.vstack(np.sum(dL_dscores.T, axis=1))/N  # Cx1
        
        # poboljšani parametri
        W += -param_delta * grad_W
        b = b - param_delta * grad_b
        
    return W, b

def logreg_classify (X, W, b):
    '''
        Argumenti
            X: NxD, primjeri
            W: CxD, vektor tezina za svaku klasu
            b: Cx1
        Povratne vrijednosti
            NxC  vjerojatnosti primjera za pojedinu klasu
    '''
    N = len(X)
    scores = np.dot(X, W.T)+np.vstack(np.array([np.hstack(b)]*N))  
    expscores = np.exp(scores)
    sumexp = np.sum(expscores, axis = 1)   
    return expscores / sumexp[:, None]         # NxC

def logreg_decfun(W, b):
    def classify (X):
        '''
            vraca niz Nx1 maksimalnih vrijednosti vjerojatnost za pojedini primjer
        '''
        return np.max(logreg_classify (X, W, b), axis = 1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    
    # instantiate the dataset
    X, Y_ = data.sample_gauss_2d(5, 30)

    
    # train the logistic regression model
    W, b = logreg_train(X, Y_)

    
    # evaluate the model on the train set
    probs = logreg_classify(X, W, b)

    
    # recover the predicted classes Y
    Y = []
    for i in range(len(X)):
        Y.append(np.argmax(probs[i]))   # uzmemo onu klasu koja ima najvecu vjerojatnost
    Y = np.vstack(Y)                    # da bude dimenzije Nx1

    # evaluate and print performance measures
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(accuracy, recall, precision)

    # graph the decision surface
    decfun = logreg_decfun(W,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
          
    # graph the data points
    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()
