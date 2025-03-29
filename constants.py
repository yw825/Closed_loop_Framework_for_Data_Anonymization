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


# NQIs = ['AgeCategory', 'LOSDays', 'NumberofVisits']
# CQIs = ['GenderDescription','RaceDescription','EthnicGroupDescription']
# SA = []

# NQIs = ['capital_gain','age', 'education_level']
# CQIs = ['sex','race','marital_status']
# # SA = []

NQIs = ['age']
CQIs = ['personal_status','job']
# # SA = []

