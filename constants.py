import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
from sklearn.model_selection import GridSearchCV
import gc
import itertools
from sklearn.utils import resample
import ast
import json
import re

import utils 
import model_train
from constants import *
import particle_swarm


# German Credit Dataset
# Path to the dataset
path = '/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Third Year Paper/experiments/German Credit dataset.csv'
# Numeric Quasi-identifiers in the dataset
NQIs = ['age']
# Categorical Quasi-identifiers in the dataset
CQIs = ['personal_status','job']
# Sensitive Attribute in the dataset
SA = ['checking_status', 'savings_status']

# # Adult Dataset
# # Path to the dataset
# path = '/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Third Year Paper/experiments/adult.csv'
# # Numeric Quasi-identifiers in the dataset
# NQIs = ['age']
# # Categorical Quasi-identifiers in the dataset
# CQIs = ['race', 'sex', 'marital_status']
# # Sensitive Attribute in the dataset
# SA = ['occupation']

# # Sepsis Patient Dataset
# # Path to the dataset
# path = '/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Third Year Paper/experiments/PSM-SepsisPatient.csv'
# # Numeric Quasi-identifiers in the dataset
# NQIs = ['AgeCategory',	'LOSDays',	'NumberofVisits']
# # Categorical Quasi-identifiers in the dataset
# CQIs = ['GenderDescription', 'RaceDescription',	'EthnicGroupDescription']
# # Sensitive Attribute in the dataset
# SA = ['HX_AIDS', 'HX_ALCOHOL', 'HX_ANEMDEF', 'HX_ARTH',	'HX_BLDLOSS',	 'HX_CHF',	'HX_CAD',	'HX_CHRNLUNG',	'HX_COAG',	'HX_DEPRESS',	'HX_DM',	'HX_DMCX',	'HX_DRUG',	'HX_HTN',	'HX_HYPOTHY',	'HX_LIVER',	'HX_LYMPH',	'HX_LYTES',	'HX_METS',	'HX_NEURO',	'HX_OBESE',	'HX_PARA',	'HX_PERIVASC',	'HX_PSYCH',	'HX_PULMCIRC',	'HX_RENLFAIL',	'HX_TUMOR',	'HX_ULCER',	'HX_VALVE',	'HX_WGHTLOSS']
