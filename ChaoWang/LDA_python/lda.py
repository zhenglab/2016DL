import pandas as pd   
import numpy as np
from matplotlib import pyplot as plt
import math

#python dict
feature_dict = {i:label for i, label in zip(
                range(4),
                    ('sepal length in cm',
                     'speal width in cm',
                     'petal length in cm',
                     'petal width in cm',))}
label_dict = {1: 'Iris-Setosa', 2: 'Iris-Versicolor', 3: 'Iris-Virginica'}
df = pd.io.parsers.read_csv(
    filepath_or_buffer='/home/if/ChallengeAll/machine_learning/LDA_python/datasets.csv',
    header=None,
    sep=',',
    )
df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
df.dropna(how='all', inplace=True) #drop the empty line at file end

df.tail()

#LDA
# step1: compute the d-dimentional mean vector
x = df[[0,1,2,3]].values
y = df['class label'].values

np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(x[y==cl], axis=0))
    print('mean vector class %s: %s\n' %(cl, mean_vectors[cl-1]))

#step2: compute the scattrer matrix

#2.1 within class scatter matrix
S_W = np.zeros((4,4)) #according the formula, the final result of the sw is a 4*4 matrix, for 4 features of one sample   
for cl,mv in zip(range(1,4), mean_vectors):
    class_sc_mat = np.zeros((4,4))
    for row in x[y == cl]:
        row, mv = row.reshape(4,1), mv.reshape(4,1) # get the col vector
        class_sc_mat += (row-mv).dot((row-mv).T) # the formula to compute the sw
    S_W += class_sc_mat

print('in Scatter Matrix:')
print(S_W)

#2.2 between class scatter matrix

overall_mean = np.mean(x, axis=0)

S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):
    n = x[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(4,1)
    overall_mean = overall_mean.reshape(4,1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between Scatter Matrix:')
print(S_B)

# step3: compute the eigenvalue(tezhengzhi) of matrix
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)
    print('eigenvector: {}: {} '.format(i+1, eigvec_sc.real))  #print in 
    print('eigenvalue: {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# check caculation

#step4: select the linear discriminants for new space
#sort eigenvectors
# make a list
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:')
for i in eig_pairs:
    print(i[0])

#acording to var to decide which to choose
print('Variance:')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('W: ', W.real)

#step5: transform the samples into the new sapce

x_lda = x.dot(W)
assert x_lda.shape == (150,2)
#define a function to show the results
def plot_step_lda():

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('o', 'o', 'o'),('red', 'green', 'blue')):

        plt.scatter(x=x_lda[:,0].real[y == label],
                    y=x_lda[:,1].real[y==label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

    plt.xlabel('LD1')
    plt.ylabel('LD2')


    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA')

    plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on",
                    left="off", right="off", labelleft="on")
    plt.grid()
    plt.show()

plot_step_lda()































    






















  
