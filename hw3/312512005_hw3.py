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

class linear_SVM:
    def __init__(self,C=1):
        self.C = C
        
    def train_model(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

        #use qpsolvers solve alpha
        P = []
        for i in range(len(self.x_train)):
            row=[]
            for j in range(len(self.x_train)):
                x_dot = self.x_train[i] @ self.x_train[j]
                element = self.y_train[i]*self.y_train[j]*x_dot
                row.append(element)
            P.append(row)
        
        P = np.array(P)
        q = -np.ones(len(self.x_train))
        A = self.y_train
        b = np.array([0])
        lb = np.zeros(len(self.x_train))
        ub = np.ones(len(self.x_train))*self.C
        alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")
        
        eps =  2.2204e-16
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
                    x_dot = self.x_train[j] @ self.x_train[i]
                    sum = sum + (self.alpha[j]*self.y_train[j]*x_dot)
                bias=(1/y_train[i])-sum
                b_list.append(bias)
        self.b_optimal = np.mean(np.array(b_list))

    def predictions(self,x_test):
        predictions = []
        #D(X)=W_T*X+b
        for i in range(len(x_test)):
            W_T_X = 0
            for j in range(len(self.x_train)):
                x_dot = self.x_train[j] @ x_test[i]
                W_T_X = W_T_X + (self.alpha[j]*self.y_train[j]*x_dot)

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
    species =['Setosa', 'Versicolor', 'Virginica']
    setosa=iris_arr[:50]
    versicolor=iris_arr[50:100]
    virginica=iris_arr[100:]
    
    #define training and test data 
    x1_train = versicolor[:25, 2:-1]
    x2_train = virginica[:25, 2:-1]
    x1_test = versicolor[25:, 2:-1]
    x2_test = virginica[25:, 2:-1]
    
    x_train = np.concatenate((x1_train,x2_train),axis=0)
    x_test = np.concatenate((x1_test,x2_test),axis=0)
    y_train = np.ones(len(x_train))
    y_train[25:] = -1
    y_test = np.ones(len(x_test))
    y_test[25:] = -1

    np.set_printoptions(suppress=True)
    #linear SVM C=1
    C = [1, 10, 100]
    bias= []
    CR=[]
    for i in range(len(C)):
        l_SVM = linear_SVM(C=C[i])
        l_SVM.train_model(x_train,y_train)
        print("linear_SVM(C={:d}) :".format(C[i]))
        print("alpha : ",np.round(l_SVM.alpha,4))
        print("total sum of alpha : ",l_SVM.alpha_sum,"\n")
        bias.append(l_SVM.b_optimal)
        CR.append(l_SVM.CR(x_test,y_test))
    linear_SVM_table(C,bias,CR)

    #nonlinear SVM RBF kernel
    C = [10, 10, 10, 10, 10]
    sigma = [5, 1, 0.5, 0.1, 0.05]
    bias= []
    CR=[]
    for i in range(len(C)):
        RBF_SVM = SVM(C=C[i],sigma=sigma[i],kernel='RBF')
        RBF_SVM.train_model(x_train,y_train)
        print("nonlinear_SVM_RBF(C=10, sigma={:.2f}) :".format(sigma[i]))
        print("alpha : ",np.round(RBF_SVM.alpha,4))
        print("total sum of alpha : ",RBF_SVM.alpha_sum,"\n")
        bias.append(RBF_SVM.b_optimal)
        CR.append(RBF_SVM.CR(x_test,y_test))
    nonlinear_SVM_table(C,bias,'RBF',sigma,CR)

    #nonlinear SVM polynomial kernel
    C = [10, 10, 10, 10, 10]
    p = [1, 2, 3, 4, 5]
    bias= []
    CR=[]
    for i in range(len(C)):
        p_SVM = SVM(C=C[i],p=p[i],kernel='polynomial')
        p_SVM.train_model(x_train,y_train)
        print("nonlinear_SVM_polynomial(C=10, p={:d}) :".format(p[i]))
        print("alpha : ",np.round(p_SVM.alpha,4))
        print("total sum of alpha : ",p_SVM.alpha_sum,"\n")
        bias.append(p_SVM.b_optimal)
        CR.append(p_SVM.CR(x_test,y_test))
    nonlinear_SVM_table(C,bias,'polynomial',p,CR)   

if __name__ == '__main__':
    main()