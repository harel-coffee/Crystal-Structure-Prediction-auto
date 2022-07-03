# General:
import sys, os
from IPython.display import Markdown, display, clear_output
import time
import datetime
import pandas as pd
from pandas import DataFrame as df
import random
import math
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
from google.colab import drive, files
drive.mount('/content/drive')

# Resampling Algorithms:
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, TomekLinks, NearMiss, OneSidedSelection
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE

# ML Algorithms:
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Other:
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import StandardScaler, normalize, LabelBinarizer
from imblearn.pipeline import Pipeline

#____________________________________________________________________________________
dir = "/content/drive/MyDrive/Crystal Prediction/Ternary Materials Point Group Prediction/Data/"

data = pd.read_pickle(dir+"NOMAD_2/Classification_Data_8_binarized.pkl")
#____________________________________________________________________________________

# Define the techniques:
scores_functions = {'Balanced_Accuracy': balanced_accuracy_score, 'Matthews_CorrCoef': matthews_corrcoef, 
                    'F1': f1_score, 'Precision': precision_score, 'Recall': recall_score, 'Accuracy': accuracy_score}
scores_names = list(scores_functions.keys())

# Define the classifiers:
classifiers = {'RF': RandomForestClassifier}

resampling_techniques = {'ROS': RandomOverSampler, 'SMOTE': SMOTE, 'ADASYN': ADASYN, 'RUS': RandomUnderSampler, 
                         'ENN': EditedNearestNeighbours, 'NM': NearMiss, 'SMOTETOMEK': SMOTETomek, 'SMOTEENN': SMOTEENN,
                         'BLSMOTE': BorderlineSMOTE, 'OSS': OneSidedSelection, 'TL': TomekLinks, 'IHT': InstanceHardnessThreshold}

classes_nums = 32

# Read the labels:
label_start = data.columns.get_loc(0)


y = df.to_numpy(data.iloc[:, label_start:label_start+classes_nums])


# Read the Features:
X = df.to_numpy(data.loc[:,['Coefficient 1', 'Coefficient 2', 'Coefficient 3',\
                              'IonizationPot1st_1', 'IonizationPot1st_2', 'IonizationPot1st_3',\
                              'Oxidation 1', 'Oxidation 2', 'Oxidation 3',\
                              'IonicRadius_1', 'IonicRadius_2', 'IonicRadius_3']])
classes_count = sum(y)
total_ones = sum(classes_count)
classifier_loop = len(classifiers)

micro_weights = np.divide(classes_count, total_ones)

print(total_ones)
#____________________________________________________________________________________
res_tech = 'SMOTEENN'

high_count_threshold = 54184 # allows for the sampling_strategy to be as low as 0.2

sampling_strategy = np.linspace(0.2, 1, num=21) # Degree of Balancing (DoB) = # Minority / # Majority


for DoB in sampling_strategy:
  for cls in classifiers: # loop over classifiers
    try:
      scores = pd.read_pickle("/content/drive/MyDrive/Crystal Prediction/Ternary Materials Point Group Prediction/Data/ResamplingStudy/Classification_Data_8_V3/"+res_tech+"mode_"+cls+"_"+str(round(DoB,2))+"DoB").transpose().to_dict()
      to_skip = len(scores['F1']['Max'])
    except:
      # Define the major scores:
      scores = {scores_names[0]: 0, scores_names[1]: 0, scores_names[2]: 0, 
                scores_names[3]: 0, scores_names[4]: 0, scores_names[5]: 0}
      for key in scores:
        scores[key] = {'Max': [], 'Min': [], 'Mean': []}
      to_skip = 0

    for PG in range(classes_nums-to_skip): # Loop over point groups
      used_PG = PG + to_skip
      clear_output()
      print("Total number of considered labels: ", total_ones)
      print('Currently Processing PG #',used_PG+1, "\n classifier: "+cls)
      # Define the kfold scores:
      scores_kfold = {scores_names[0]: [], scores_names[1]: [], scores_names[2]: [],
                      scores_names[3]: [], scores_names[4]: [], scores_names[5]: []}

      if classes_count[used_PG] < high_count_threshold:
        used_DoB = DoB
      else:
        used_DoB = 1

      for train_index, test_index in RepeatedStratifiedKFold(n_splits=5, n_repeats=1).split(X, y[:,used_PG]):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[:,used_PG][train_index], y[:,used_PG][test_index]

        pipeline = Pipeline([('resampling', resampling_techniques[res_tech](sampling_strategy=used_DoB, smote=SMOTE(k_neighbors=7), enn=EditedNearestNeighbours(n_neighbors=7, kind_sel='mode'))), ('model', classifiers[cls]())]).fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        i=0
        for key in scores_kfold:
          scores_kfold[key].append(scores_functions[scores_names[i]](y_test, y_pred)*100)
          i+=1

      for key in scores:
        scores[key]['Max'].append(round(max(scores_kfold[key]),3))
        scores[key]['Min'].append(round(min(scores_kfold[key]),3))
        scores[key]['Mean'].append(round(mean(scores_kfold[key]),3))
      scores_df = pd.DataFrame.from_dict(scores, orient='index')
      scores_df.to_pickle("/content/drive/MyDrive/Crystal Prediction/Ternary Materials Point Group Prediction/Data/ResamplingStudy/Classification_Data_8_V3/"+res_tech+"mode_"+cls+"_"+str(round(DoB,2))+"DoB")
