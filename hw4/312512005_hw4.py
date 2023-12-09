import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

def read_to_array(filepath):
    iris_dataframe = pd.read_csv(filepath, header=None, sep='\s+')
    iris_data_array = iris_dataframe.to_numpy()

    return iris_data_array

def linear_SVM_table(C,b,CR):
    CR=["{:.4f}".format(CR_f) for CR_f in CR]
    b=["{:.4f}".format(bias) for bias in b]

    data = {'penalty weight C': C,
            'bias' : b,
            'CR(%)' : CR}
    
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.scale(1.2,2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title('Linear SVM',y=0.8)
    plt.show()

def nonlinear_SVM_table(C,b,kernel,para,CR):
    CR=["{:.4f}".format(CR_f) for CR_f in CR]
    b=["{:.4f}".format(bias) for bias in b]
    para=["{:.2f}".format(x) for x in para]
    if(kernel == 'RBF'):
        index = 'sigma'
    else:
        index = 'p'

    data = {'penalty weight C': C,
            index: para,
            'bias' : b,
            'CR(%)' : CR}
    
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.scale(1.2,2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title('Nonlinear SVM ({:s})'.format(kernel),y=0.9)
    plt.show()

def SVM_one_against_one(x1_train, x2_train, x3_train, x_test, y_test, num_class_type, C, sigma):
    y_train = np.ones(50)
    y_train[25:] = -1

    #training multiclass SVM
    SVM_1_2 =  SVM(C=C,sigma=sigma,kernel='RBF')
    x_train = np.concatenate((x1_train, x2_train),axis=0)
    SVM_1_2.train_model(x_train,y_train)

    SVM_1_3 =  SVM(C=C,sigma=sigma,kernel='RBF')
    x_train = np.concatenate((x1_train, x3_train),axis=0)
    SVM_1_3.train_model(x_train,y_train)

    SVM_2_3 =  SVM(C=C,sigma=sigma,kernel='RBF')
    x_train = np.concatenate((x2_train, x3_train),axis=0)
    SVM_2_3.train_model(x_train,y_train)

    #pridiction of each classifier
    pred_1_2_original = SVM_1_2.predictions(x_test)
    pred_1_2 = np.where(pred_1_2_original == 1, 1,2)
    pred_1_3_original = SVM_1_3.predictions(x_test)
    pred_1_3 = np.where(pred_1_3_original == 1, 1,3)
    pred_2_3_original = SVM_2_3.predictions(x_test)
    pred_2_3 = np.where(pred_2_3_original == 1, 2,3)
    predictions=np.stack((pred_1_2,pred_1_3,pred_2_3),axis=0)

    #if there is a model can calculate the b parameter, then output CR=nan
    if(np.isnan(SVM_1_2.b_optimal) or np.isnan(SVM_1_3.b_optimal) or np.isnan(SVM_2_3.b_optimal)):
        return np.nan
    
    #find final pridiction
    final_prediction = []
    for i in range(len(y_test)):
        class_vote_arr=np.zeros(num_class_type)
        #count the prediction of each model votes
        for j in range(num_class_type):
            if(predictions[j][i]!=1 and predictions[j][i]!=2 and predictions[j][i]!=3):
                pass
            else:
                temp=int(predictions[j][i])
                class_vote_arr[temp-1] +=1
        
        max_value = np.max(class_vote_arr)
        if(np.isnan(max_value)):
            final_prediction.append(np.nan)
        else:
            #np.where return a tuple
            max_indices = np.where(class_vote_arr == max_value)[0]
            if len(max_indices) == 1:
                final_prediction.append(max_indices[0]+1)
            else:
                final_prediction.append(np.nan)

    final_prediction_arr=np.array(final_prediction)

    #count CR
    equal_bool=(final_prediction_arr==y_test)
    num_correct_pred=0
    for i in equal_bool:
        if i == True:
            num_correct_pred+=1
    CR= (num_correct_pred/len(y_test))*100
    return CR
class SVM:
    def __init__(self,C=1, sigma=5, p=1, kernel='RBF'):
        self.C = C
        self.sigma = sigma
        self.p = p
        self.kernel = kernel

    def kernel_func(self, xi, xj):
        if self.kernel == 'RBF':
            power = -(np.linalg.norm((xi-xj), ord=2)**2) / (2*(self.sigma**2))
            K = np.exp(power)
        elif self.kernel == 'polynomial':
            K = (xi@xj)**self.p
        elif self.kernel ==  'linear':
            K = (xi@xj)
        return K
    
    def train_model(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        
        #use qpsolvers solve alpha
        P = []
        for i in range(len(self.x_train)):
            row=[]
            for j in range(len(self.x_train)):
                K = self.kernel_func(self.x_train[i], self.x_train[j])
                element = self.y_train[i]*self.y_train[j]*K
                row.append(element)
            P.append(row)
        
        P = np.array(P)
        q = -np.ones(len(self.x_train))
        A = self.y_train
        b = np.array([0])
        lb = np.zeros(len(self.x_train))
        ub = np.ones(len(self.x_train))*self.C
        alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")

        eps =   2.2204e-16
        for i in range(alpha.size):
            if alpha[i] >= self.C - np.sqrt(eps):
                alpha[i] = self.C
                alpha[i] = np.round(alpha[i],6)
            elif  alpha[i] <= 0 + np.sqrt(eps):
                alpha[i] = 0
                alpha[i] = np.round(alpha[i],6)
            else:
                alpha[i] = np.round(alpha[i],6)
                #print(f"support vector: alpha = {alpha[i]}")
        self.alpha = alpha
        self.alpha_sum = np.round(np.sum(self.alpha),4)

        #find b*
        b_list=[]
        for i in range(len(self.alpha)):
            if self.alpha[i]>0 and self.alpha[i]<self.C:
                sum=0
                for j in range(len(self.alpha)):
                    K = self.kernel_func(self.x_train[j], self.x_train[i])
                    sum = sum + (self.alpha[j]*self.y_train[j]*K)
                bias=(1/y_train[i])-sum
                b_list.append(bias)
        self.b_optimal = np.mean(np.array(b_list))

    def predictions(self,x_test):
        predictions = []
        if np.isnan(self.b_optimal):
            predictions = [np.nan for i in range(len(x_test))]
        else:
            #D(X)=W_T*X+b
            for i in range(len(x_test)):
                W_T_X = 0
                for j in range(len(self.x_train)):
                    K = self.kernel_func(self.x_train[j], x_test[i])
                    W_T_X = W_T_X + (self.alpha[j]*self.y_train[j]*K)

                D = W_T_X + self.b_optimal 
                #D>=0 is positive, D<0 is negative
                if D>=0:
                    predictions.append(1)
                elif D<0:
                    predictions.append(-1)

        return np.array(predictions)
    

    def CR(self,test_data_features,test_data_labels):
        #number of data
        num_all=len(test_data_labels)

        #count number of correct prediction
        test_pred=self.predictions(test_data_features)
        equal_bool=(test_pred==test_data_labels)
        num_correct_pred=0
        for i in equal_bool:
            if i == True:
                num_correct_pred+=1
        #count CR
        CR= (num_correct_pred/num_all)*100
        return CR
    
def main():
    iris_arr=read_to_array("iris.txt")
    features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    species = ['Setosa', 'Versicolor', 'Virginica']
    setosa=iris_arr[:50]
    versicolor=iris_arr[50:100]
    virginica=iris_arr[100:]
    
    #define training and test data 
    x1_first_half = setosa[:25, :-1]
    x2_first_half = versicolor[:25, :-1]
    x3_first_half = virginica[:25, :-1]
    y1_first_half = setosa[:25,-1]
    y2_first_half = versicolor[:25,-1]
    y3_first_half = virginica[:25,-1]

    x1_second_half = setosa[25: , :-1]
    x2_second_half = versicolor[25: , :-1]
    x3_second_half = virginica[25: , :-1]
    y1_second_half = setosa[25:,-1]
    y2_second_half = versicolor[25:,-1]
    y3_second_half = virginica[25:,-1]

    #grid search
    C_list = [1, 5, 10, 50, 100, 500, 1000]
    sigma_power = list(range(-100, 101 ,5))
    sigma_base = 1.05
    sigma_list=[sigma_base ** power for power in sigma_power]
    sigma_index = []
    best_CR = 0
    CR=[]
    for i in range(len(sigma_list)):
        CR_row = []
        sigma = sigma_list[i]
        for j in range(len(C_list)):
            C = C_list[j]
            #fold 1 
            x_test = np.concatenate((x1_second_half, x2_second_half, x3_second_half),axis=0)
            y_test = np.concatenate((y1_second_half, y2_second_half, y3_second_half),axis=0)
            CR1 = SVM_one_against_one(x1_first_half, x2_first_half, x3_first_half, x_test, y_test, 3, C, sigma)
            #fold 2
            x_test = np.concatenate((x1_first_half, x2_first_half, x3_first_half),axis=0)
            y_test = np.concatenate((y1_first_half, y2_first_half, y3_first_half),axis=0)
            CR2 = SVM_one_against_one(x1_second_half, x2_second_half, x3_second_half, x_test, y_test, 3, C, sigma)

            if(np.isnan(CR1) or np.isnan(CR2)):
                CR_row.append(np.nan)
            else:
                CR_avg= ((CR1+CR2)/2)
                CR_row.append('{:.2f}'.format(CR_avg))
                
                if CR_avg > best_CR:
                    best_CR = CR_avg

        #change sigma into string representation
        sigma_index.append('{:.2f}^{:d}'.format(sigma_base,sigma_power[i]))
        CR.append(CR_row)
        
    df = pd.DataFrame(CR, columns=C_list, index=sigma_index)
    print(df)
    df.to_excel('output.xlsx')
    #find the optimal parameters and the corresponding CR
    best_CR_str = '{:.2f}'.format(best_CR)
    for i in range(len(CR)):
        for j in range(len(CR[i])):
            if(CR[i][j] == best_CR_str):
                best_C = C_list[j]
                best_sigma = '{:.2f}^{:d}'.format(sigma_base,sigma_power[i])
                print("Optimal parameters and CR are : [C = {:d}, sigma = {:s} , best CR = {:s}]".format(best_C,best_sigma,CR[i][j]))

    
if __name__ == '__main__':
    main()