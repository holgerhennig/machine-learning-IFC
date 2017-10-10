"""
machine learning for cell cycle assay
"""

from __future__ import division
import numpy as np
import os
import pandas
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

working_directory = '/Users/holgerh/Dropbox/holger/work/Academia/presenting/dissemination_workshops/machine-learning-IFC_tutorial/examples/cell_cycle'
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

# build ground truth
# img# | class# | phase
#  1		    4	     ana	
# 2-65		  1	     G1
# 66-104	  1	     G2
# 105		  3     	meta
# 106-108  2	     pro
# 109-147  1       S
# 148	   	5	     telo

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

# feature scaling (normalize features to mean 0 and std 1, i.e. Z-score normalization)
data = preprocessing.scale(data, axis=0)

# revert back to pandas data frame
data = pandas.DataFrame(data)
data.columns = all_features_names

# remove highly correlated features
# skip this for now

### plot function ###
def plot(classifier_name, cm_diag):
    bar_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    y_pos = np.arange(len(bar_labels))

    plt.bar(y_pos,
        cm_diag,
        align='center',
        color='blue')

    plt.ylabel('Prediction rate in percent')
    plt.xticks(y_pos, bar_labels)
    plt.title('Cell Classes, ' + classifier_name)
    plt_name = classifier_name + '_plt.png'
    plt.savefig(os.path.join(output_directory, plt_name))
    plt.clf()

#%% Machine Learning ###
names_classifiers = []

names_classifiers.append(('NaiveBayes', GaussianNB()))
names_classifiers.append(('RandomForest', RandomForestClassifier()))

# GradientBoosting takes longer to run, if you want to quickly try out this machine learning script, use NaiveBayes
# names_classifiers.append(('GradientBoosting', GradientBoostingClassifier()))

for name, classifier in names_classifiers:
    print("Begin classification with %s\n" %name)
    
    #cross validation
    y_pred = sklearn.model_selection.cross_val_predict(classifier, data, ground_truth, cv=10)
    cm = confusion_matrix(ground_truth, y_pred, labels = use_classes)
    
    #normalize confusion matrix
    row_sums = cm.sum(axis=1)
    normalized_cm = cm / row_sums[:, np.newaxis] *100
    np.set_printoptions(precision=1, suppress=True)
    cm_diag = normalized_cm.diagonal()
    cm_file_name = name + '_cm.txt'
    cm_file = open(os.path.join(output_directory, cm_file_name), 'w+')
    cm_file.write(str(normalized_cm)) # only works properly under python 3.x, not under 2.x
    cm_file.close()
    plot(name, cm_diag)
        
    # list most important features
    if (name == 'RandomForest') or (name == 'GradientBoosting'):
        print("Extracting most important features with %s\n" %name)
        
        # calculate feature importance
        classifier.fit(data, ground_truth)
        feature_importance = classifier.feature_importances_
        
        # feature importance in percent
        feature_importance = 100.0 * (feature_importance / feature_importance.max()) 
        
        # sort features (from high to low importance)
        sorted_idx = np.argsort(-feature_importance)
   
        # write TOP 20 most important features to file
        with open(os.path.join(output_directory, 'feature_selection.txt'), 'a') as file:
            file.write('\n *** Feature importance from %s in percent *** \n' % name)
            for k in np.arange(0,20):
                file.write("%s.  %s, %s\n" % (k+1, data.columns[sorted_idx[k]], feature_importance[sorted_idx[k]]))
            file.close()
