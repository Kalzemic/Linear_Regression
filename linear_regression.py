from numpy import * 


def compute_error(b,m,points):
    #initialize error at 0
    totalError= 0
    for i in range(0,len(points)):
        #get the x value
        x=points[i,0]
        #get the y value
        y= points[i,1]
        #get the difference, square it, add it to the total
        totalError+=((y-(m*x+b))**2) 
    #get the average 
    return totalError/ float(len(points))


def gradient_descent(points,starting_b,starting_m,learning_rate,num_iterations):
    #starting b and m
    b=starting_b
    m=starting_m

    #gradient descent
    for i in range(num_iterations):
        #update b and m with the more accurate iteration after this gradient step
        b,m =gradient_step(b,m,array(points),learning_rate)
    return[b,m]


def gradient_step(b_current,m_current,points,learning_rate):
    #starting points for our gradients
    b_gradient=0
    m_gradient=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        #direction with respect to b and m
        #computing partial derevatives of our error function
        b_gradient-= (2/len(points))*(y-((m_current*x)+b_current))
        m_gradient-= (2/len(points))* x *(y-((m_current*x)+b_current))
    #update our b and m values
    new_b =b_current- (learning_rate * b_gradient)
    new_m= m_current- (learning_rate* m_gradient)
    return [new_b,new_m]



def run():
    
    #Step 1- collect our data
    points = genfromtxt("https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv", delimiter= ',')

    #Step 2- define hyperparameters
    # how fast should our model converge?  
    
    learning_rate= 0.0001
    
    #y=mx+b (slope formula)
    initial_b=0 
    initial_m=0
    num_iterations=1000


    #Step 3- train our model 
    print ('starting gradient descent at b={0}, m={1}, error= {2}'.format(initial_b, initial_m, compute_error(initial_b,initial_m,points)))
    [b,m]=gradient_descent(points,initial_b,initial_m,learning_rate,num_iterations)
    
    print ('ending gradient descent at b={1}, m={2}, error= {3}'.format(num_iterations,b, m, compute_error(b,m,points)))






if __name__ == '__main__':
    run()