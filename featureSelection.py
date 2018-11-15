import data
import sol_greedysubset as gr
import project as pr
import matplotlib.pyplot as plt
import numpy as np

def run(F):
    c_X, c_Y = data.load()

    X = c_X.values
    #Greedy Subset
    t2, t2_S = gr.run(F,X,c_Y)

    #Plot Greedy Subset
    plt.figure()
    plt.bar(range(t2_S.size), t2_S)
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.title('Greedy Subset')
    plt.show()

    return t2, t2_S
    
run(50)
