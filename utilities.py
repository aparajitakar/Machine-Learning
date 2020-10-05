import numpy as np
import sklearn
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import random

def my_cross_val(method,X,y,k):
    
    fold_batch = float(len(y)/k)
    error_rate = np.zeros(k)
    index = list(range(len(X)))
    random.shuffle(index)
    X_mod = [0]*len(X)
    y_mod = [0]*len(X)
    for i in range(len(X)):
        X_mod[i] = X[index[i]]
        y_mod[i] = y[index[i]]
    
    for i in range(k):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for j in range(len(y)):
            if(j>= int(i*fold_batch) and j<int((i+1)*fold_batch)):
                test_x.append(X_mod[j])
                test_y.append(y_mod[j])
            else:
                train_x.append(X_mod[j])
                train_y.append(y_mod[j])
        if(method == 'LinearSVC'):
            model= LinearSVC(max_iter=2000)
        if(method == 'SVC'):
            model = SVC(gamma='scale',C=10)
        if(method == 'LogisticRegression'):
            model = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
        model.fit(train_x,train_y)
        predict = model.predict(test_x)
        error_count = 0
        for a in range(len(test_y)):
            if(predict[a] != test_y[a]):
                error_count += 1
        error_rate[i] = error_count/len(test_y)
    mean = np.mean(error_rate, axis=0)
    sigma = np.std(error_rate)
    return(error_rate,mean,sigma)

def my_train_test(method,X,y,pi,k):
    
    error_rate = np.zeros(k)    
    for i in range(k):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        
        #random shuffling 
        
        index = list(range(len(y)))
        random.shuffle(index)
        X_mod = [0]*len(y)
        y_mod = [0]*len(y)
        for ele in range(len(y)):
            X_mod[ele] = X[index[ele]]
            y_mod[ele] = y[index[ele]]
            
        split = int(pi*len(y)) 
        
        for j in range(len(y)):
            if(j < split):
                train_x.append(X_mod[j])
                train_y.append(y_mod[j])
            else:
                test_x.append(X_mod[j])
                test_y.append(y_mod[j])
               
        if(method == 'LinearSVC'):
            model= LinearSVC(max_iter=2000)
        if(method == 'SVC'):
            model = SVC(gamma='scale',C=10)
        if(method == 'LogisticRegression'):
            model = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
        model.fit(train_x,train_y)
        predict = model.predict(test_x)
        error_count = 0
        for a in range(len(test_y)):
            if(predict[a] != test_y[a]):
                error_count += 1
        error_rate[i] = error_count/len(test_y)
    mean = np.mean(error_rate, axis=0)
    sigma = np.std(error_rate)
    return(error_rate,mean,sigma)


def rand_proj(X,d):
    size = 64*d
    s = np.random.normal(0, 1,size)
    s_new = s.reshape(64,d)
    X_dash = np.dot(X,s_new)
    return X_dash

def quad_proj(X):
    combination_value = int((len(X[0])*(len(X[0])-1))/2)
    X_combination = np.empty([len(X),combination_value])
    temp_X = [0]*len(X)
    index2 = []
    X_sq = np.empty((X.shape))
    for i in range(len(X)):
        for j in range(len(X[0])):
            X_sq[i][j]=np.square(X[i][j])
    
    jcounter = 0
    l=1
    for j in range(len(X[0])-1):
        for jdash in range(j+1,len(X[0])):
            for i in range(len(X)):
                X_combination[i][jcounter] = X[i][j] * X[i][jdash]
            jcounter += 1
    X = np.column_stack((X, X_sq))
    X = np.column_stack((X, X_combination))
    return(X)

def print_table_values(method,dataset,error_rate,mean,std,q):
    print(f'Error rates for {method} with {dataset}')
    filename = method+dataset+q+".txt"
    f = open(filename,'w')
    print(f'Error rates for {method} with {dataset}', file =f)
    for i in range(len(error_rate)):
        print(f'Fold {i}: {error_rate[i]}')
    print(f'Mean: {mean}')
    print(f'Standard Deviation: {std}')
    for i in range(len(error_rate)):
        print(f'F{i}\t', end="", file =f)
    print('Mean\t', end="", file =f)
    print('SD', file =f)
    for i in error_rate:
        print(f'{i}\t', end="", file =f)
    print(f'{mean}\t', end="", file =f)
    print(f'{std}\t', file =f)
    f.close()
    return