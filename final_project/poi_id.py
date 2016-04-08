#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi',
				 'bonus',
				 'exercised_stock_options',
				 'salary']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#pprint.pprint(data_dict)

### Data exploration
### Total number of data points
### Allocation across classes (POI/non-POI)
### Number of features
### Features with missing values?
### Exploratory plots to look for outliers?

### Task 2: Remove outliers

# Remove from data set
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for key in my_dataset:
	if (my_dataset[key]['bonus']) == 'NaN' or (my_dataset[key]['salary']) == 'NaN':
		my_dataset[key]['ratio'] = 0.0
	else:
		my_dataset[key]['ratio'] = float(my_dataset[key]['bonus']) / float(my_dataset[key]['salary'])
	
	if (my_dataset[key]['from_messages']) == 'NaN' or (my_dataset[key]['to_messages']) == 'NaN':
		my_dataset[key]['ratio_from_to'] = 0.0
	else:
		my_dataset[key]['ratio_from_to'] = float(my_dataset[key]['from_messages']) / float(my_dataset[key]['to_messages'])
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#pprint.pprint(data[0:10])
#pprint.pprint(labels)
#pprint.pprint(features)

#from sklearn.preprocessing import MinMaxScaler
#pprint.pprint(min(data[:,1]))
#pprint.pprint(max(data[:,1]))
#pprint.pprint(min(data[:,2]))
#pprint.pprint(max(data[:,2]))
#min_max_scaler = MinMaxScaler()
#data[:,1:2] = min_max_scaler.fit_transform(data[:,1:2])
#data = min_max_scaler.fit_transform(data)
#data[:,1] = min_max_scaler.fit_transform(data[:,1])
#data[:,2] = min_max_scaler.fit_transform(data[:,2])
#pprint.pprint(data[0:10])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.



from sklearn.grid_search import GridSearchCV
# use_classifier can be 'GaussianNB', 'Tree', 'SVC', 'Neighbor', 'RForest',
# 'AdaBoost', 'LogReg', 'Voting', 'GradBoost'
use_classifier = 'SVC'

if use_classifier == 'Tree':
	from sklearn.tree import DecisionTreeClassifier
	svr = DecisionTreeClassifier()
	parameters = {'criterion':('gini', 'entropy'), 
			  'splitter':('best', 'random'),
			  'max_features':('auto','sqrt','log2', None),
			  'max_depth':(1,2,3,4,5,None),
			  'class_weight':('balanced', None),
			  'presort': (True, False),
			  'random_state': range(2),
			  'min_samples_leaf': (1,2)}
	clf = GridSearchCV(svr, parameters)
	
elif use_classifier == 'SVC':
	from sklearn.svm import SVC
	parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 
				  'C':[1, 10, 25, 50, 100],
				  'degree':[2, 3, 4, 5],
				  'gamma': [2, 3, 4, 'auto']}
	svr = SVC()
	clf = GridSearchCV(svr, parameters)
	
elif use_classifier == 'GaussianNB':
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	
elif use_classifier == 'Neighbor':
	from sklearn.neighbors import KNeighborsClassifier
	svr = KNeighborsClassifier()
	parameters = {'n_neighbors': range(1,30),
				  'weights':['uniform', 'distance'],
				  'algorithm':['auto','ball_tree','kd_tree','brute'],
				  'leaf_size':range(5,50,5),
				  'metric':['euclidean','manhattan','chebyshev','minkowski'],
				  'p':range(3,10)}
	clf = GridSearchCV(svr, parameters)
	
elif use_classifier == 'RForest':
	from sklearn.ensemble import RandomForestClassifier
	svr = RandomForestClassifier()
	parameters = {'n_estimators': range(5,50,5),
				  'random_state': range(2),
				  'criterion':('gini', 'entropy'),
				  'max_features':('auto','sqrt','log2', None),
				  'max_depth':(1,2,3,4,5,None),
				  'bootstrap':[True, False]}
	clf = GridSearchCV(svr, parameters)
	
elif use_classifier == 'AdaBoost':
	from sklearn.ensemble import AdaBoostClassifier
	svr = AdaBoostClassifier()
	parameters = {'algorithm':['SAMME', 'SAMME.R'],
				  'n_estimators':range(5,100,5),
				  'random_state': range(2)}
	clf = GridSearchCV(svr, parameters)

elif use_classifier == 'LogReg':
	from sklearn.linear_model import LogisticRegression
	svr = LogisticRegression()
	parameters = {'penalty': ['l1','l2'],
				  'C':[0.1, 0.25, 0.5, 0.75, 1.0],
				  'random_state': range(2),
				  'class_weight':['balanced', None],
				  'fit_intercept':[True, False]}
	clf = GridSearchCV(svr, parameters)

elif use_classifier == 'GradBoost':
	from sklearn.ensemble import GradientBoostingClassifier
	svr = GradientBoostingClassifier()
	parameters = {'loss': ['deviance', 'exponential'],
				  'n_estimators':range(5,100,5),
				  'max_depth': (1,2,3,4,5,None),
				  'min_samples_leaf': (1,2),
				  'max_features':('auto','sqrt','log2', None),
				  'random_state': range(2),
				  'presort': (True, False, 'auto')}
	clf = GridSearchCV(svr, parameters)
	
elif use_classifier == 'Voting':
	from sklearn.ensemble import VotingClassifier
	from sklearn.svm import SVC
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import GradientBoostingClassifier
	clf1 = SVC(random_state=1)
	clf2 = RandomForestClassifier(bootstrap = False,
								  n_estimators = 20,
								  random_state = 0,
								  criterion = 'entropy',
								  max_features = 'auto',
								  max_depth = 5)
	clf3 = AdaBoostClassifier(n_estimators = 20,
							  random_state = 0,
							  algorithm = 'SAMME.R')
	clf4 = KNeighborsClassifier(n_neighbors = 3,
							    algorithm = 'auto',
								metric = 'manhattan',
								p = 3,
								weights = 'uniform',
								leaf_size = 5)
	clf5 = GaussianNB()
	clf6 = DecisionTreeClassifier(presort = True, splitter = 'best',
								  min_samples_leaf = 1,
								  random_state = 0,
								  criterion = 'entropy',
								  max_features = 'auto',
								  max_depth = 5,
								  class_weight = None)
	clf7 = GradientBoostingClassifier(presort = True,
									  loss = 'deviance',
									  min_samples_leaf = 1,
									  n_estimators = 25,
									  random_state = 1,
									  max_features = 'sqrt',
									  max_depth = 4)
	clf = VotingClassifier(estimators=[('sv', clf1), ('rf', clf2),
							           ('ab', clf3), ('knn', clf4),
									   ('gnb',clf5), ('dt', clf6),
									   ('gb', clf7)],
						   voting='hard')
	
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)


if use_classifier == 'Tree':
	print clf.best_params_
	best = clf.best_params_
	clf = DecisionTreeClassifier(presort = best['presort'],
							 splitter = best['splitter'],
							 criterion = best['criterion'],
							 max_features = best['max_features'],
							 max_depth = best['max_depth'],
							 class_weight = best['class_weight'],
							 random_state = best['random_state'],
							 min_samples_leaf = best['min_samples_leaf'])
							 
elif use_classifier == 'SVC':
	print clf.best_params_
	best = clf.best_params_
	clf = SVC(kernel = best['kernel'],
			  C = best['C'],
			  degree = best['degree'],
			  gamma = best['gamma'])
			  
elif use_classifier == "Neighbor":
	print clf.best_params_
	best = clf.best_params_
	clf = KNeighborsClassifier(n_neighbors = best['n_neighbors'],
							   weights = best['weights'],
							   algorithm = best['algorithm'],
							   leaf_size = best['leaf_size'],
							   metric = best['metric'],
							   p = best['p'])
elif use_classifier == 'RForest':
	print clf.best_params_
	best = clf.best_params_
	clf = RandomForestClassifier(n_estimators = best['n_estimators'],
								 random_state = best['random_state'],
								 criterion = best['criterion'],
								 max_features = best['max_features'],
								 max_depth = best['max_depth'],
								 bootstrap = best['bootstrap'])
elif use_classifier == 'AdaBoost':
	print clf.best_params_
	best = clf.best_params_
	clf = AdaBoostClassifier(algorithm = best['algorithm'],
							 n_estimators = best['n_estimators'],
							 random_state = best['random_state'])
							 
elif use_classifier == 'LogReg':
	print clf.best_params_
	best = clf.best_params_
	clf = LogisticRegression(penalty = best['penalty'],
							 C = best['C'],
							 random_state = best['random_state'],
							 class_weight = best['class_weight'],
							 fit_intercept = best['fit_intercept'])
							 
elif use_classifier == 'Voting':
	pass;

elif use_classifier == 'GradBoost':
	print clf.best_params_
	best = clf.best_params_
	clf = GradientBoostingClassifier(loss = best['loss'],
									 n_estimators = best['n_estimators'],
									 max_depth = best['max_depth'],
									 min_samples_leaf = best['min_samples_leaf'],
									 max_features = best['max_features'],
									 random_state = best['random_state'],
									 presort = best['presort'])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)