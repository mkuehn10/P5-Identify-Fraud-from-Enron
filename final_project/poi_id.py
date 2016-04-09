#!/usr/bin/python
import sys
import pickle
import pprint
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Complete list of features
features_list = ['poi', 'bonus', 'ratio', 'ratio_from_to',
 'deferral_payments',
 'deferred_income',
 'director_fees',
 'exercised_stock_options',
 'expenses',
 'from_messages',
 'from_poi_to_this_person',
 'from_this_person_to_poi',
 'loan_advances',
 'long_term_incentive',
 'other',
 'restricted_stock',
 'restricted_stock_deferred',
 'salary',
 'shared_receipt_with_poi',
 'to_messages',
 'total_payments',
 'total_stock_value']
 
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

### Task 2: Remove outliers
# Total and Travel Agency are not people
# Lockhart had no values in any column
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Two new features
### Ratio of bonus to salary
### Ratio of from_messages to to_messages
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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Provided to give you a starting point. Try a variety of classifiers.
### Use GridSearchCV to tune each algorithm separately and then pass the
### optimized parameters to VotingClassifier for each algorithm
 
from sklearn.grid_search import GridSearchCV
### use_classifier can be 'GaussianNB', 'Tree', 'SVC', 'Neighbor', 'RForest',
### 'AdaBoost', 'Voting', 'GradBoost'
use_classifier = 'Voting'

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
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	clf2 = RandomForestClassifier(bootstrap = False,
								  n_estimators = 20,
								  random_state = 0,
								  criterion = 'entropy',
								  max_features = 'auto',
								  max_depth = 5)
	clf3 = AdaBoostClassifier(n_estimators = 20,
							  random_state = 0,
							  algorithm = 'SAMME.R')
	#clf4 = KNeighborsClassifier(n_neighbors = 3,
	#						    algorithm = 'auto',
	#							metric = 'manhattan',
	#							p = 3,
	#							weights = 'uniform',
	#							leaf_size = 5)
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
	clf = VotingClassifier(estimators=[('rf', clf2),
							           ('ab', clf3),
									   ('gnb',clf5), ('dt', clf6),
									   ('gb', clf7)],
						   voting='hard')
	


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
# selector = SelectKBest(f_classif, k = 'all')
# selector.fit_transform(features_train, labels_train)
# print selector.scores_
# print selector.pvalues_

clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)

### Pass the optimized version of the classifier to tester
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
	clf.fit(features_train, labels_train)
	features = sorted(zip(features_list[1:],clf.feature_importances_), key = lambda x: x[1])
	pprint.pprint(features)
							 
	
			  
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
	clf.fit(features_train, labels_train)
	features = sorted(zip(features_list[1:],clf.feature_importances_), key = lambda x: x[1])
	pprint.pprint(features)
								 
elif use_classifier == 'AdaBoost':
	print clf.best_params_
	best = clf.best_params_
	clf = AdaBoostClassifier(algorithm = best['algorithm'],
							 n_estimators = best['n_estimators'],
							 random_state = best['random_state'])
	clf.fit(features_train, labels_train)
	features = sorted(zip(features_list[1:],clf.feature_importances_), key = lambda x: x[1])
	pprint.pprint(features)
							
							 
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
	clf.fit(features_train, labels_train)
	features = sorted(zip(features_list[1:],clf.feature_importances_), key = lambda x: x[1])
	pprint.pprint(features)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)