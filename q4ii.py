import utilities as util

def q4ii():
    digits = util.load_digits()    
    digits_X = digits.data
    X = util.quad_proj(digits_X)
    Y = digits.target
    for method in ('LinearSVC', 'SVC', 'LogisticRegression'):
        error_rate,mean,std = util.my_cross_val(method,X,Y,10)
        util.print_table_values(method,'digits',error_rate,mean,std,'q4ii')
        
