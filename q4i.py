import utilities as util

def q4i():
    digits = util.load_digits()    
    digits_X = digits.data
    X = util.rand_proj(digits_X,32)
    Y = digits.target
    for method in ('LinearSVC', 'SVC', 'LogisticRegression'):
        error_rate,mean,std = util.my_cross_val(method,X,Y,10)
        util.print_table_values(method,'digits',error_rate,mean,std,'q4i')
        
