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



def main():
    iris_arr=read_to_array("iris.txt")
    features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    species =['Setosa', 'Versicolor', 'Virginica']
    scatter_plot(iris_arr,features, species)

if __name__ == '__main__':
    main()