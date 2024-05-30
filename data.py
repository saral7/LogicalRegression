import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
    minx = 0
    maxx = 10
    miny = 0
    maxy = 10
    
    def __init__(self):
        self. means = np.random.random_sample(2)*(self.maxx-self.minx)+self.minx
        eigvalx1 = (np.random.random_sample()*(self.maxx - self.minx)/5)**2 
        eigvalx2 = (np.random.random_sample()*(self.maxx - self.minx)/5)**2 

        self.D = np.matrix([[eigvalx1, 0],
                           [0, eigvalx2]])
        
        theta = np.random.rand()*np.pi/2
        self.R = np.matrix([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        
        self.cov = self.R.transpose() * self.D * self.R


    def get_sample(self, n):
        return np.random.multivariate_normal(self.means, self.cov, n)


'''
    Funkcija za sample_gauss_2d koja generira manje zanimljiv primjer
'''
##def sample_gauss_2d(C, N):
##    '''
##        Argumenti
##            C: broj klasa
##            N: broj primjera svake klase
##        Povratne vrijednosti
##            x: np.array (N*C)x2, N*C primjera 2d tocaka
##            y: np.array (N*C)x1, N*C tocnih razreda za svaki primjer
##    '''
##    x = np.array([[0, 0]])
##    y = np.array([])
##    for i in range(C):
##        gauss = Random2DGaussian()
##        toAdd = gauss.get_sample(N)
##        x = np.append(x, toAdd, axis = 0)
##        y = np.append(y, [i]*N)
##    x = np.delete(x, 0, axis = 0)
##    return x, np.vstack(y)

def sample_gauss_2d(nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs=[]
    Ys=[]
    for i in range(nclasses):
      Gs.append(Random2DGaussian())
      Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_= np.hstack([[Y]*nsamples for Y in Ys])
    Y_= np.vstack(Y_)
    return X,Y_

# funkcija koja mi na kraju nije trebala
def sample_gauss_2d_general(C, N):
    '''
        Argumenti
            C: broj klasa
            N: broj primjeraka za svaku klasu
        Povratne vrijednosti
            X: (N*C)x2 matrica 2d primjeraka
            Y_: (N*C)xC matrica gdje je za svaki primjer u j-tom stupcu 1 ako je taj primjer j-te klase
    '''
    # create the distributions and groundtruth labels
    Gs=[]
    Ys=[]
    for i in range(C):
      Gs.append(Random2DGaussian())
      Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_ = []
    for i in range (C):
        row = [0]*C
        row[i] = 1
        for j in range (N):
            Y_.append(row)
    Y_ = np.reshape(Y_, (N*C, C))
    return X,Y_


def eval_perf_binary(Y,Y_): #Y predvideni, Y_ tocni
    '''
        Argumenti
            Y: np.array Nx1, predvidene vrijednosti klasa
            Y_: np.array Nx1, tocne vrijednosti klasa
        Povratne vrijednosti
            accuracy, recall, precision
    '''
    Y = np.ravel(Y)
    Y_ = np.ravel(Y_)
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    p = 0
    n = 0
    for i in range (len(Y)):
        p += (Y_[i] == 0)
        n += (Y_[i] == 1)
        if (Y[i] == Y_[i]):
            if (Y[i] == 0):
                tn+=1
            if (Y[i] == 1):
                tp+=1
        else:
            if (Y[i] == 0):
                fn+=1
            if (Y[i] == 1):
                fp+=1
    population = tn+tp+fn+fp
    accuracy = (tn+tp)/population
    recall = tp/p
    precision = tp/(tp+fp)

    return accuracy, recall, precision

def eval_AP (Yr):
    '''
        Argumenti
            Yr: np.array Nx1, sortiran iz Y_ prema argsortu
        Povratne vrijednosti
            prosjecna preciznost
    '''
    sol = 0
    Yr = np.vstack(Yr)
    precision_list = np.vstack([1]*len(Yr))
    for i in range(len(Yr)):
       a, r, precision = eval_perf_binary(precision_list, Yr)
       sol += precision * Yr[i]
       precision_list[i] = 0
    return (sol/np.sum(Yr))[0]

def eval_perf_multi (Y, Y_):
    '''
        Argumenti
            Y: np.array Nx1, predvidene vrijednosti klasa
            Y_: np.array Nx1, tocne vrijednosti klasa
        Povratne vrijednosti
            
    '''
    Y = np.ravel(Y)
    Y_ = np.ravel(Y_)
    
    C = max(Y)+1
    confMat = []

    recall = []
    precision = []
    for i in range (C):
        row = []
        for j in range (C):
            row.append(sum(np.logical_and(Y_==i, Y==j)))
        confMat.append(row)
        
    confMat = np.matrix(confMat)

    for i in range (C):
        recall.append(np.ravel(confMat[i])[i]/(np.ravel(np.sum(confMat, axis = 1))[i]))      # za svaku klasu: tocno klasif. / svi iz klase
        precision.append(np.ravel(confMat[i])[i]/(np.ravel(np.sum(confMat, axis = 0))[i]))   # za svaku klasu: tocno klsif. / svi tako predvideni

    accuracy = np.trace(confMat)/np.sum(confMat)                                             # tocno klasificirani / svi
    return accuracy, recall, precision
    


def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

def graph_data (X, Y_, Y):
    '''
        Argumenti
            X: podatci (np.array dimenzija Nx2)
            Y_: točni indeksi razreda podataka (Nx1)
            Y: predviđeni indeksi razreda podataka (Nx1)
    '''
    N = len(X)
    C = max((Y_.flatten()))+1
    
    for i in range (N):
        mark = ""
        col = []
        if (Y[i] != Y_[i]):
            mark = "s"              # kvadratic ako je netocno klasificiran
        else:
            mark = "o"              # kruzic ako je tocno klasificiran
            
        col = [0.2+0.8/(C-1) * Y_[i][0]]*3    # sto ranije klasa, to tamnija
        plt.scatter(X[i, 0], X[i, 1], marker = mark, color = col, edgecolor = 'black')

    return


def graph_surface (fun, rect, offset, width = 256, height = 256):
    '''
        Argumenti
            fun: decizijska funkcija (Nx2)->(Nx1)
            rect: željena domena prikaza zadana kao:
                       ([x_min,y_min], [x_max,y_max])
            offset: "nulta" vrijednost decizijske funkcije na koju je potrebno
                    poravnati središte palete boja;
                    tipično imamo:
                       offset = 0.5 za probabilističke modele 
                          (npr. logistička regresija)
                       offset = 0 za modele koji ne spljošćuju
                          klasifikacijske mjere (npr. SVM)
            width,height: rezolucija koordinatne mreže
    '''
    xaxis = np.linspace(rect[0][0], rect[1][0], width)          # num = broj koraka, rezolucija
    yaxis = np.linspace(rect[0][1], rect[1][1], height)
   
    x2, y2 = np.meshgrid(xaxis, yaxis, indexing = 'xy')         # parovi koordinata u tocnom broju

    grid = np.stack((x2.flatten(), y2.flatten()), axis=1)       # parovi koordinata
    scores = fun(grid).reshape((width, height))                 # scores = za svaku tocku grafa vrijednost funkcije
    mini = np.min(scores)
    maxi = np.max(scores)
    dist = max(abs(maxi-offset), abs(offset-mini))              # za odredivanje vmin i vmax
    plt.pcolormesh(xaxis, yaxis, scores, vmin = offset-dist, vmax = offset+dist) # vmin i vmax odreduju range boja

    #TODO: plt.contour(xaxis, yaxis, scores, levels = [0], colors = "black")
    return



##if __name__=="__main__":
##    np.random.seed(100)
##  
##    # get the training dataset
##    X,Y_ = sample_gauss_2d(2, 100)
##    Y_ = np.vstack(Y_)      # dodatno
####    
##    # get the class predictions
##    Y = np.vstack(myDummyDecision(X)>0.5)  # granica je pravac y+x-5.5=0, dodan vstack
##  
##    # graph the surface
##    bbox=(np.min(X, axis=0), np.max(X, axis=0))
##    print("bbox: ", bbox)
##    graph_surface(myDummyDecision, bbox, offset=0.5)        # stavljam offset = 0.5, inace je krivo
####
##    # graph the data points
##    graph_data(X, Y_, Y) 
##  
##    # show the results
##    plt.show()


