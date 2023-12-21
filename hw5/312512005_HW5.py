import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class LDA_classifier:
    def __init__(self,C1=1,C2=1):
        self.C1 = C1
        self.C2 = C2
        

    def train_model(self,positive_features, negative_features, positive_label, negative_label):
        self.pos_label = positive_label
        self.neg_label = negative_label
        m1=np.mean(positive_features,axis=0,keepdims=True)
        m2=np.mean(negative_features,axis=0,keepdims=True)
        n1 = len(positive_features)
        n2 = len(negative_features)
        p1=(n1/(n1+n2))
        p2=(n2/(n1+n2))

        feature_num= len(positive_features[0])
        
        sigma1 = np.zeros((feature_num, feature_num))
        for xi in positive_features:
            xi=np.array([xi])
            temp = np.matmul(((xi-m1).T),(xi-m1))
            sigma1 = sigma1 + temp
        
        sigma2 = np.zeros((feature_num, feature_num))
        for xj in negative_features:
            xj=np.array([xj])
            temp = np.matmul(((xj-m2).T),(xj-m2))
            sigma2 = sigma2 + temp

        sigma1 = sigma1/(n1-1)
        sigma2 = sigma2/(n2-1)
        sigma = p1*sigma1 + p2*sigma2
        sigma_inverse = np.linalg.inv(sigma)
        self.W_T = np.matmul((m1-m2),sigma_inverse)
        self.b = (-1/2)*(np.matmul(self.W_T,((m1+m2).T))) - math.log((self.C1*p2)/(self.C2*p1))

    def predictions(self,test_data_features):
        
        predictions = []
        #D(X)=W_T*X+b
        for x in test_data_features:
            x=np.array([x])
            D= np.matmul(self.W_T,x.T) + self.b
            #D>0 is positive, D<0 is negative , D=0 means that classification failed, store it as np.nan
            if D>0:
                predictions.append(self.pos_label)
            elif D<0:
                predictions.append(self.neg_label)
            else:
                predictions.append(np.nan)

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
    
def two_fold_LDA(x,y):

    LDA = LDA_classifier(C1=1, C2=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

    #fold 1 : x_train as training data, x_test as test data
    #seperate label 0 and label 1 data 
    x_label_0 = x_train[y_train==0]
    x_label_1 = x_train[y_train==1]

    LDA.train_model(x_label_0,x_label_1,0,1)
    CR1= LDA.CR(x_test,y_test)

    #fold 2 : x_test as training data, x_train as test data
    #seperate label 0 and label 1 data 
    x_label_0 = x_test[y_test==0]
    x_label_1 = x_test[y_test==1]

    LDA.train_model(x_label_0,x_label_1,0,1)
    CR2= LDA.CR(x_train,y_train)
    CR_avg= ((CR1+CR2)/2)
    return CR_avg


def SFS(data,y,feature_names):
    #number of the features 
    N = len(data[0])
    selected_features = []
    remaining_features =[i for i in range(N)]
    SFS_feature_subset = []
    SFS_feature_CR = []
    n=0
    while n<N:
        CR_max = 0
        for feature in remaining_features:
            temp_feature = selected_features + [feature]
            temp_feature.sort()
            x = data[ : , temp_feature]
            CR = two_fold_LDA(x,y)
            if(CR>CR_max):
                CR_max = CR
                final_selected_features = temp_feature
                corresponding_add_feature = feature

        selected_features = final_selected_features
        remaining_features.remove(corresponding_add_feature)
        SFS_feature_subset.append(tuple(selected_features))
        SFS_feature_CR.append(CR_max)
        n=len(selected_features)

    #find the best CR and corresponding feature subset
    
    max_index = np.argmax(np.array(SFS_feature_CR))
    best_CR = SFS_feature_CR[max_index]
    optimal_feature_number = len(SFS_feature_subset[max_index])
    optimal_features = feature_names[list(SFS_feature_subset[max_index])]
    for features,CR in zip(SFS_feature_subset,SFS_feature_CR):
        print("The number of features in this subset: ", len(features))
        print("feature subset:", features)
        print("Highest validated balanced accuracy: {:.2f}".format(CR))
        print()
    print("The number of features in the Optimal feature subset: {:d} ".format(optimal_feature_number))
    print("Optimal feature subset: ", SFS_feature_subset[max_index] )
    print("Optimal feature subset names: ", optimal_features)
    print("best CR by SFS: {:.2f}".format(best_CR))

def Fisher(data,y,feature_names):
    x_label_0 = data[y==0]
    x_label_1 = data[y==1]
    m1 = np.mean(x_label_0,axis=0,keepdims=True)
    m2 = np.mean(x_label_1,axis=0,keepdims=True)
    m = np.mean(data,axis=0,keepdims=True)
    n1 = len(x_label_0)
    n2 = len(x_label_1)
    p1=(n1/(n1+n2))
    p2=(n2/(n1+n2))
    feature_num = len(data[0])
    #calculate Sw
    sw1 = np.zeros((feature_num, feature_num))
    for xi in x_label_0:
        xi=np.array([xi])
        temp = np.matmul(((xi-m1).T),(xi-m1))
        sw1 = sw1 + temp
    
    sw2 = np.zeros((feature_num, feature_num))
    for xj in x_label_1:
        xj=np.array([xj])
        temp = np.matmul(((xj-m2).T),(xj-m2))
        sw2 = sw2 + temp

    sw1 = sw1/(n1)
    sw2 = sw2/(n2)
    Sw =  p1*sw1 + p2*sw2
    
    #calculate Sb
    sb1 = n1*(np.matmul(((m1-m).T),(m1-m)))
    sb2 = n2*(np.matmul(((m2-m).T),(m2-m)))
    Sb = sb1 + sb2

    #calculate fisher's scores
    fisher_scores = []
    for k in range(feature_num):
        f_score = Sw[k,k]/Sb[k,k]
        fisher_scores.append(f_score)

    #sort features by fisher's scores in descending order
    #get fisher's scores in ascending order index
    sort_ascending_index = np.argsort(np.array(fisher_scores))
    #reverse it into descending order
    sort_descending_index = sort_ascending_index[::-1]

    #calculate top n f-scores-ranked
    n = 0
    CR_list = []
    while n < feature_num:
        n = n + 1
        temp_feature = sort_descending_index[:n]
        x = data[ : , temp_feature]
        CR = two_fold_LDA(x,y)
        CR_list.append(CR)
    
    CR_max_index = np.argmax(np.array(CR_list))
    best_top_N = CR_max_index+1
    for i in range(feature_num):
        print("Top-{:d}-ranked features, CR: {:.2f} ".format(i+1,CR_list[i]))

    optimal_features = feature_names[sort_descending_index[:best_top_N]]
    best_CR = CR_list[CR_max_index]
    print("The number of features in the Optimal feature subset: {:d} ".format(best_top_N))
    print("Optimal feature subset: ", tuple(sort_descending_index[:best_top_N]))
    print("Optimal feature subset names: ", optimal_features)
    print("best CR by Fisher’s Criterion: {:.2f}".format(best_CR))

def main():
    breast_cancer_data = load_breast_cancer()
    x = breast_cancer_data.data
    y = breast_cancer_data.target
    feature_names = breast_cancer_data.feature_names
    #malignant=0,benign=1
    
    SFS(x,y,feature_names)
    print()
    Fisher(x,y,feature_names)
    
if __name__ == '__main__':
    main()