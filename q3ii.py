import utilities as util

def q3ii():
    
    boston = util.load_boston()
    boston50_X = boston.data
    boston50_Y = []
    boston25_X = boston.data
    boston25_Y = []
    
    percentile50 = util.np.percentile(boston.target,50)
    percentile25 = util.np.percentile(boston.target,25)

    for i in range(len(boston.target)):
        if(boston.target[i] >= percentile50):
            boston50_Y.append(1)            
        else:
            boston50_Y.append(0)
        if(boston.target[i] >= percentile25):
            boston25_Y.append(1)
        else:
            boston25_Y.append(0)        

    digits = util.load_digits()    
    digits_X = digits.data
    digits_Y = digits.target
    
    X = [boston50_X, boston25_X, digits_X]
    Y = [boston50_Y, boston25_Y, digits_Y]
    for method in ('LinearSVC', 'SVC', 'LogisticRegression'):
        x = 0
        for j in ('Boston50','Boston25','Digits'):
            error_rate,mean,std = util.my_train_test(method,X[x],Y[x],0.75,10)
            x = x + 1
            util.print_table_values(method,j,error_rate,mean,std,'q3ii')
    
       