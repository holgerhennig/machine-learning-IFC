"""
machine learning for cell cycle assay
"""

from __future__ import division
import numpy
import os
import pandas
import sklearn
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from operator import itemgetter
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

working_directory = '/Users/holgerh/Dropbox/holger/work/github_repositories/machine-learning-IFC/examples/cell_cycle'
os.chdir(working_directory)
input_directory = "Step3_AllData/"
output_directory = "Step4_ML_output/"

#class_names = {'Anaphase','G1','G2','Metaphase','Prophase','S','Telophase'};
#class_labs  = [    4,       1,   1,      3,         2,      1,      5     ];

brightfield_filename = 'BF_cells_on_grid.txt';
darkfield_filename  = 'SSC.txt';

if not os.path.exists(input_directory):
    print("Input directory does not exist. The input directory is the folder where the image montages are located")
    
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

print("Loading data")
brightfield = pandas.read_csv(os.path.join(input_directory, brightfield_filename), sep='\t', )
darkfield = pandas.read_csv(os.path.join(input_directory, darkfield_filename), sep='\t',)

#%% Preprocessing ###

#exclude features
exclude_featuresBF = [0,1,3,4,7,17,19]
exclude_featuresBF.extend(list(range(70,77)))
exclude_featuresDF = list(range(0,50))
exclude_featuresDF.extend(list(range(70,77)))

brightfield.drop(brightfield.columns[exclude_featuresBF],axis=1,inplace=True)
darkfield.drop(darkfield.columns[exclude_featuresDF],axis=1,inplace=True)

#build ground truth
ground_truth_list = [4] * 225
ground_truth_list.extend([1] * (103 * 225))
ground_truth_list.extend([3] * 225)
ground_truth_list.extend([2]*(3*225))
ground_truth_list.extend([1] * (39*225))
ground_truth_list.extend([5] * 225)

ground_truth = pandas.DataFrame({'ground_truth': ground_truth_list})


#combine bf and df
data = pandas.concat([brightfield, darkfield, ground_truth], axis=1)

#drop nan
data = data.dropna()

#split data and ground truth
ground_truth = data['ground_truth']
data.drop('ground_truth', axis = 1, inplace=True)

#get a list with all the features names
all_features_names =list(data.columns.values)

# remove low variance features
selector = VarianceThreshold() #.8 * (1 - .8) 
data = selector.fit_transform(data)

#remove highly correlated features
#skip this for now

### The plot function ###
def plot(classifier_name, cm_diag):
    bar_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    y_pos = numpy.arange(len(bar_labels))

    plt.bar(y_pos,
        cm_diag,
        align='center',
        color='blue')

    plt.ylabel('Percent of cells correctly classifed')
    plt.xticks(y_pos, bar_labels)
    plt.title('Cell Classes, ' + classifier_name)
    plt_name = classifier_name + '_plt.png'
    plt.savefig(os.path.join(output_directory, plt_name))
    plt.clf()

#%% Machine Learning ###
names_classifiers = []
#names_classifiers.append(('RandomForest', RandomForestClassifier()))
names_classifiers.append(('NaiveBayes', GaussianNB()))


for name, classifier in names_classifiers:
    #cross validation
    y_pred = sklearn.model_selection.cross_val_predict(classifier, data, ground_truth, cv=10)
    cm = confusion_matrix(ground_truth, y_pred, labels = [1,2,3,4,5])
    
    #normalize confusion matrix
    row_sums = cm.sum(axis=1)
    normalized_cm = cm / row_sums[:, numpy.newaxis]
    numpy.set_printoptions(precision=3, suppress=True)
    cm_diag = normalized_cm.diagonal()
    cm_file_name = name + '_cm.txt'
    cm_file = open(os.path.join(output_directory, cm_file_name), 'w+')
    cm_file.write(str(normalized_cm))
    cm_file.close()
    #print(normalized_cm)
    plot(name, cm_diag)





#%% Feature selection ###
names_classifiers = []
names_classifiers.append(('RandomForest', RandomForestClassifier()))

for name, classifier in names_classifiers:
    selector = RFECV(classifier, step=20, cv=10)
    selector = selector.fit(data, ground_truth)
    y_pred = selector.predict(data)
    cm = confusion_matrix(ground_truth, y_pred, labels = [1,2,3,4,5])
    
    #normalize confusion matrix
    row_sums = cm.sum(axis=1)
    normalized_cm = cm / row_sums[:, numpy.newaxis]
    numpy.set_printoptions(precision=3, suppress=True)
    cm_diag = normalized_cm.diagonal()
    
    selected_features_indices = selector.get_support(True)
    selected_features = []
    for i in selected_features_indices:
        selected_features.append(all_features_names[i])
    
    output_file_name = name + '.txt'
    output_file = open(output_file_name, 'w+')
    output_file.write(str(selected_features))
    output_file.write("\n\n")
    output_file.write(str(normalized_cm))
    output_file.close()
    plot(name, cm_diag)