import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_to_array(filepath):
    iris_dataframe = pd.read_csv(filepath, header=None, sep='\s+')
    iris_data_array = iris_dataframe.to_numpy()

    return iris_data_array

def scatter_plot(data_arr, features, species):
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            #plot label 1
            x1 = data_arr[ :50, i]
            y1 = data_arr[ :50, j]
            plt.scatter(x1, y1, label=species[0], color='blue', marker='o', s=20)

            #plot label 2
            x2 = data_arr[50:100, i]
            y2 = data_arr[50:100, j]
            plt.scatter(x2, y2, label=species[1], color='red', marker='o', s=20)

            #plot label 3
            x3 = data_arr[100:, i]
            y3 = data_arr[100:, j]
            plt.scatter(x3, y3, label=species[2], color='green', marker='o', s=20)

            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.title('Scatter plot of '+features[i]+' and '+features[j])
            plt.legend()
            plt.show()

def KNN_CR(k_neighbors,first_half_f,first_half_l,second_half_f,second_half_l):
    num_features = np.size(first_half_f,axis=1)

    CR=[]
    fea_comb =[]
    #choose 1 feature
    for i in range(num_features):
        KNN_1= KNN(k=k_neighbors)

        #count CR of the condition that first half as training,second half as testing
        KNN_1.train_model(first_half_f[:,i],first_half_l)
        CR1= KNN_1.CR(second_half_f[:,i],second_half_l)
        #count CR of the condition that second half as training,first half as testing
        KNN_1.train_model(second_half_f[:,i],second_half_l)
        CR2= KNN_1.CR(first_half_f[:,i],first_half_l)

        CR_avg= ((CR1+CR2)/2)
        CR.append(CR_avg)
        fea_comb.append([i])

    #choose 2 feature
    for i in range(num_features):
        for j in range(i+1,num_features):
            KNN_1= KNN(k=k_neighbors)
            x1=np.stack((first_half_f[:,i],first_half_f[:,j]),axis=1)
            x2=np.stack((second_half_f[:,i],second_half_f[:,j]),axis=1)

            #count CR of the condition that first half as training,second half as testing
            KNN_1.train_model(x1,first_half_l)
            CR1= KNN_1.CR(x2,second_half_l)
            #count CR of the condition that second half as training,first half as testing
            KNN_1.train_model(x2,second_half_l)
            CR2= KNN_1.CR(x1,first_half_l)

            CR_avg= ((CR1+CR2)/2)
            CR.append(CR_avg)
            fea_comb.append([i,j])

    #choose 3 feature         
    for i in range(num_features):
        for j in range(i+1,num_features):
            for k in range(j+1,num_features):
                KNN_1= KNN(k=k_neighbors)
                x1=np.stack((first_half_f[:,i],first_half_f[:,j],first_half_f[:,k]),axis=1)
                x2=np.stack((second_half_f[:,i],second_half_f[:,j],second_half_f[:,k]),axis=1)

                #count CR of the condition that first half as training,second half as testing
                KNN_1.train_model(x1,first_half_l)
                CR1= KNN_1.CR(x2,second_half_l)
                #count CR of the condition that second half as training,first half as testing
                KNN_1.train_model(x2,second_half_l)
                CR2= KNN_1.CR(x1,first_half_l)

                CR_avg= ((CR1+CR2)/2)
                CR.append(CR_avg)
                fea_comb.append([i,j,k])
    
    #choose 4 feature         
    #count CR of the condition that first half as training,second half as testing
    KNN_1= KNN(k=k_neighbors)
    KNN_1.train_model(first_half_f,first_half_l)
    CR1= KNN_1.CR(second_half_f,second_half_l)
    #count CR of the condition that second half as training,first half as testing
    KNN_1.train_model(second_half_f,second_half_l)
    CR2= KNN_1.CR(first_half_f,first_half_l)

    CR_avg= ((CR1+CR2)/2)
    CR.append(CR_avg)
    fea_comb.append([0,1,2,3])

    return CR,fea_comb

def CR_table(k,CR,fea_comb,features):
    #transform the features combinations into string type
    fc=[]
    for fea in fea_comb:
        temp=[]
        for i in fea:
            temp.append(features[i])
        fc.append(' + '.join(temp))    

    CR=["{:.2f}".format(CR_f) for CR_f in CR]

    data = {'Feature Combinations': fc,
            'CR(%)' : CR}
    
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.scale(1.2,1.3)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    plt.title('k={:d}'.format(k))
    plt.show()
    
class KNN:
    def __init__(self,k):
        self.k_neighbors = k

    def train_model(self,train_data_features,train_data_labels):
        self.training_features = train_data_features
        self.training_label = train_data_labels
        labels=np.unique(train_data_labels)
        self.label_num=len(labels)

    def predictions(self,test_data_features):
        
        predictions = []
        #count the distance
        for test in test_data_features:

            distance_list=[]
            for train in self.training_features:
                sub = np.subtract(test, train)
                square = np.power(sub,2)
                distance = np.sqrt(np.sum(square))
                distance_list.append(distance)

            distance_arr=np.array(distance_list)
            
            #use argsort to sort the distance from small to large
            distance_with_label= np.stack((distance_arr,self.training_label),axis=1)
            sort_distance_index =np.argsort(distance_with_label[:,0],kind='mergesort')
            distance_sort=np.take(distance_with_label,sort_distance_index,axis=0)
            
            #classify the label depends on k 
            labels_arr=np.zeros(self.label_num)
            for i in range(self.k_neighbors):
                label_temp=int(distance_sort[i,1])
                labels_arr[label_temp-1]+=1
            '''
            #use argmin to find smallest distance index
            labels_arr=np.zeros(self.label_num)
            for i in range(self.k_neighbors):
               min_index=np.argmin(distance_arr)
               label_temp=self.training_label[min_index]
               labels_arr[label_temp-1]+=1
               distance_arr=np.delete(distance_arr,min_index)
            '''
            predictions.append((np.argmax(labels_arr))+1)
            
        return np.array(predictions)
    
    def CR(self,test_data_features,test_data_labels):
        #number of data
        num_all=len(test_data_labels)

        #count number of correct prediction
        test_pred=self.predictions(test_data_features)
        equal_bool= (test_pred==test_data_labels)
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
    scatter_plot(iris_arr,features, species)

    #Split the data into the first half and the second half
    label1_first_half = iris_arr[ :25]
    label1_second_half = iris_arr[25:50]
    label2_first_half = iris_arr[50:75]
    label2_second_half = iris_arr[75:100]
    label3_first_half = iris_arr[100:125]
    label3_second_half = iris_arr[125:]

    iris_first_half = np.concatenate((label1_first_half, label2_first_half,label3_first_half),axis=0)
    iris_second_half = np.concatenate((label1_second_half,label2_second_half,label3_second_half),axis=0)

    iris_first_half_features = iris_first_half[:,:-1]
    iris_first_half_labels = (iris_first_half[:,-1]).astype(int)
    iris_second_half_features = iris_second_half[:,:-1]
    iris_second_half_labels = (iris_second_half[:,-1]).astype(int)

    #k=1
    CR,feature_combinations = KNN_CR(1,iris_first_half_features, iris_first_half_labels, iris_second_half_features, iris_second_half_labels)
    CR_table(1,CR,feature_combinations,features)
    #k=3
    CR,feature_combinations = KNN_CR(3,iris_first_half_features, iris_first_half_labels, iris_second_half_features, iris_second_half_labels)
    CR_table(3,CR,feature_combinations,features)

if __name__ == '__main__':
    main()