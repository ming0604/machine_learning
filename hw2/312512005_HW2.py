import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math


def read_to_array(filepath):
    iris_dataframe = pd.read_csv(filepath, header=None, sep='\s+')
    iris_data_array = iris_dataframe.to_numpy()

    return iris_data_array

def two_fold_LDA_two_classes(positive_data, negative_data):
    x1_first_half = positive_data[:25,2:-1]
    x2_first_half = negative_data[:25,2:-1]
    y1_first_half = positive_data[:25,-1]
    y2_first_half = negative_data[:25,-1]

    x1_second_half = positive_data[25: ,2:-1]
    x2_second_half = negative_data[25: ,2:-1]
    y1_second_half = positive_data[25:,-1]
    y2_second_half = negative_data[25:,-1]

    positive_label = np.unique(y1_first_half)[0]
    negative_label = np.unique(y2_first_half)[0]


    LDA = LDA_classifier(C1=1, C2=1)
    weight_vector=[]
    bias=[]
    CR=[]
    #count CR of the condition that first half as training,second half as testing
    LDA.train_model(x1_first_half,x2_first_half,positive_label,negative_label)
    weight_vector.append(LDA.W_T)
    bias.append(LDA.b)
    x_test = np.concatenate((x1_second_half,x2_second_half),axis=0)
    y_test = np.concatenate((y1_second_half,y2_second_half),axis=0)
    CR1= LDA.CR(x_test,y_test)
    CR.append(CR1)

    #count CR of the condition that second half as training,first half as testing
    LDA.train_model(x1_second_half,x2_second_half,positive_label,negative_label)
    weight_vector.append(LDA.W_T)
    bias.append(LDA.b)
    x_test = np.concatenate((x1_first_half,x2_first_half),axis=0)
    y_test = np.concatenate((y1_first_half,y2_first_half),axis=0)
    CR2= LDA.CR(x_test,y_test)
    CR.append(CR2)
    CR_avg= ((CR1+CR2)/2)
    CR.append(CR_avg)

    CR=["{:.2f}".format(CR_f) for CR_f in CR]

    #plot table
    data = {'weight_vector': weight_vector ,
            'bias' : bias,
            'CR(%)' : CR
            }
    index_list = ['fold1', 'fold2', 'average CR']
    
    df = pd.concat([pd.DataFrame({'weight_vector': weight_vector}), pd.DataFrame({'bias' : bias}), pd.DataFrame({'CR(%)' : CR})], axis=1)
    df.set_index(pd.Index(index_list), inplace=True)

    fig, ax = plt.subplots(figsize=(9,3))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,rowLabels=df.index, cellLoc='center', loc='center')
    table.scale(1,2)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    plt.title('LDA classifier of Versicolor and Virginica')
    plt.show()

def LDA_roc(positive_data, negative_data, title):
    x1_first_half = positive_data[:25,:-1]
    x2_first_half = negative_data[:25,:-1]
    y1_first_half = positive_data[:25,-1]
    y2_first_half = negative_data[:25,-1]

    x1_second_half = positive_data[25: ,:-1]
    x2_second_half = negative_data[25: ,:-1]
    y1_second_half = positive_data[25:,-1]
    y2_second_half = negative_data[25:,-1]

    positive_label = np.unique(y1_first_half)[0]
    negative_label = np.unique(y2_first_half)[0]

    #C1 = np.linspace(0.0001, 10000, 20000)
    #C2 = np.ones(20000)
    C1 = np.logspace(-4, 4, num=9)
    C2 = np.ones(9)
    ratio = C1/C2
    tpr1_arr=[]
    fpr1_arr=[]
    tpr2_arr=[]
    fpr2_arr=[]
    for i in range(len(C1)):
        #fold1
        LDA_1 = LDA_classifier(C1=C1[i], C2=C2[i])
        LDA_1.train_model(x1_first_half,x2_first_half,positive_label,negative_label)
        x_test = np.concatenate((x1_second_half,x2_second_half),axis=0)
        y_test = np.concatenate((y1_second_half,y2_second_half),axis=0)
        y_pred = LDA_1.predictions(x_test)
        cm = confusion_matrix(y_test, y_pred, labels=[3,2])
        tp = cm[0,0] 
        fn = cm[0,1]
        fp = cm[1,0]
        tn = cm[1,1]
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tpr1_arr.append(tpr)
        fpr1_arr.append(fpr)
    
        #fold 2
        LDA_2 = LDA_classifier(C1=C1[i], C2=C2[i])
        LDA_2.train_model(x1_second_half,x2_second_half,positive_label,negative_label)
        x_test = np.concatenate((x1_first_half,x2_first_half),axis=0)
        y_test = np.concatenate((y1_first_half,y2_first_half),axis=0)
        y_pred = LDA_2.predictions(x_test)
        cm = confusion_matrix(y_test, y_pred, labels=[3,2])
        tp = cm[0,0] 
        fn = cm[0,1]
        fp = cm[1,0]
        tn = cm[1,1]
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tpr2_arr.append(tpr)
        fpr2_arr.append(fpr)

    tpr1_arr=np.array(tpr1_arr)
    fpr1_arr=np.array(fpr1_arr)
    tpr2_arr=np.array(tpr2_arr)
    fpr2_arr=np.array(fpr2_arr)
    tpr_arr=(tpr1_arr+tpr2_arr)/2
    fpr_arr=(fpr1_arr+fpr2_arr)/2


    aoc=np.abs(np.trapz(tpr_arr,fpr_arr))
    fig = plt.figure()
    plt.plot(fpr_arr, tpr_arr, label='ROC curve, AOC={:.2f}'.format(aoc),marker = 'o')
    for i in range(len(tpr_arr)):
        plt.text(fpr_arr[i]+0.005, tpr_arr[i]+0.01, 'C1/C2={:.4f}'.format(ratio[i]),fontsize=5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()

def LDA_multiclass(class1_data,class2_data,class3_data):
    x1_first_half = class1_data[:25,2:-1]
    x2_first_half = class2_data[:25,2:-1]
    x3_first_half = class3_data[:25,2:-1]
    y1_first_half = class1_data[:25,-1]
    y2_first_half = class2_data[:25,-1]
    y3_first_half = class3_data[:25,-1]

    x1_second_half = class1_data[25: ,2:-1]
    x2_second_half = class2_data[25: ,2:-1]
    x3_second_half = class3_data[25: ,2:-1]
    y1_second_half = class1_data[25:,-1]
    y2_second_half = class2_data[25:,-1]
    y3_second_half = class3_data[25:,-1]
    class_type=3
    CR=[]
    #fold 1 that first half as training,second half as testing
    #training multiclass LDA 
    LDA_1_2 = LDA_classifier(C1=1, C2=1)
    positive_label = np.unique(y1_first_half)[0]
    negative_label = np.unique(y2_first_half)[0]
    LDA_1_2.train_model(x1_first_half,x2_first_half,positive_label,negative_label)

    LDA_1_3 = LDA_classifier(C1=1, C2=1)
    positive_label = np.unique(y1_first_half)[0]
    negative_label = np.unique(y3_first_half)[0]
    LDA_1_3.train_model(x1_first_half,x3_first_half,positive_label,negative_label)

    LDA_2_3 = LDA_classifier(C1=1, C2=1)
    positive_label = np.unique(y2_first_half)[0]
    negative_label = np.unique(y3_first_half)[0]
    LDA_2_3.train_model(x2_first_half,x3_first_half,positive_label,negative_label)

    #pridiction of each classifier
    x_test = np.concatenate((x1_second_half,x2_second_half,x3_second_half),axis=0)
    y_test = np.concatenate((y1_second_half,y2_second_half,y3_second_half),axis=0)
    pred_1_2 = LDA_1_2.predictions(x_test)
    pred_1_3 = LDA_1_3.predictions(x_test)
    pred_2_3 = LDA_2_3.predictions(x_test)
    predictions=np.stack((pred_1_2,pred_1_3,pred_2_3),axis=0)

    #find final pridiction
    final_prediction = []
    for i in range(len(y_test)):
        class_arr=np.zeros(class_type)
        for j in range(class_type):
            if(predictions[j][i]!=1 and predictions[j][i]!=2 and predictions[j][i]!=3):
                pass
            else:
                temp=int(predictions[j][i])
                class_arr[temp-1] +=1
        
        max_value = np.max(class_arr)
        max_indices = np.where(class_arr == max_value)[0]
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
    CR1= (num_correct_pred/len(y_test))*100
    CR.append(CR1)

    #fold 2 that second half as training,first half as testing
    #training multiclass LDA 
    LDA_1_2 = LDA_classifier(C1=1, C2=1)
    positive_label = np.unique(y1_second_half)[0]
    negative_label = np.unique(y2_second_half)[0]
    LDA_1_2.train_model(x1_second_half,x2_second_half,positive_label,negative_label)

    LDA_1_3 = LDA_classifier(C1=1, C2=1)
    positive_label = np.unique(y1_second_half)[0]
    negative_label = np.unique(y3_second_half)[0]
    LDA_1_3.train_model(x1_second_half,x3_second_half,positive_label,negative_label)

    LDA_2_3 = LDA_classifier(C1=1, C2=1)
    positive_label = np.unique(y2_second_half)[0]
    negative_label = np.unique(y3_second_half)[0]
    LDA_2_3.train_model(x2_second_half,x3_second_half,positive_label,negative_label)

    #pridiction of each classifier
    x_test = np.concatenate((x1_first_half,x2_first_half,x3_first_half),axis=0)
    y_test = np.concatenate((y1_first_half,y2_first_half,y3_first_half),axis=0)
    pred_1_2 = LDA_1_2.predictions(x_test)
    pred_1_3 = LDA_1_3.predictions(x_test)
    pred_2_3 = LDA_2_3.predictions(x_test)
    predictions=np.stack((pred_1_2,pred_1_3,pred_2_3),axis=0)

    #find final pridiction
    final_prediction = []
    for i in range(len(y_test)):
        class_arr=np.zeros(class_type)
        for j in range(class_type):
            if(predictions[j][i]!=1 and predictions[j][i]!=2 and predictions[j][i]!=3):
                pass
            else:
                temp=int(predictions[j][i])
                class_arr[temp-1] +=1
        
        max_value = np.max(class_arr)
        max_indices = np.where(class_arr == max_value)[0]
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
    CR2= (num_correct_pred/len(y_test))*100
    CR.append(CR2)

    #average CR
    CR_avg= ((CR1+CR2)/2)
    CR.append(CR_avg)
    CR=["{:.2f}".format(CR_f) for CR_f in CR]
    
    #plot table
    data = {'CR(%)' : CR}
    index_list = ['fold1', 'fold2', 'average CR']
    df = pd.DataFrame(data,index=index_list)
    fig, ax = plt.subplots(figsize=(9,3))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,rowLabels=df.index, cellLoc='center', loc='center')
    table.scale(1,2)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    plt.title('multi-class LDA classifier')
    plt.show()
    
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


def main():
    iris_arr=read_to_array("iris.txt")
    features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    species =['Setosa', 'Versicolor', 'Virginica']
    

    #Split the data into the first half and the second half
    setosa=iris_arr[:50]
    versicolor=iris_arr[50:100]
    virginica=iris_arr[100:]
    two_fold_LDA_two_classes(versicolor,virginica)
    LDA_roc(virginica, versicolor,"ROC curve (four features)")
    LDA_roc(virginica[:, [0, 1, 4]], versicolor[:, [0, 1, 4]],"ROC curve (1st & 2nd features)")
    LDA_roc(virginica[:,2:], versicolor[:,2:],"ROC curve (3rd & 4th features)")
    LDA_multiclass(setosa,versicolor,virginica)
if __name__ == '__main__':
    main()