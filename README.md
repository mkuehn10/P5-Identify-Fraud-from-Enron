P5 Identify Fraud from Enron Project
==============


#### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The bankruptcy of Enron was due to its complex business model and unethical practices such as using accounting statements to misrepresent earnings and indicate favorable performances.  Several people who worked at Enron were found to be persons of interest (POI) who were involved with this scandal.  Data on several features such as bonuses, salaries, stock options, etc. is available for Enron employees.  The goal is to see if a combination of these features is correlated with whether or not someone is a POI.  This is a classification problem in which several data points are used to classify someone as a POI or non-POI.  Machine learning can examine the data in order to create a classifier that could potentially look at new data to identify POIs before a financial collapse occurs.  This would be greatly beneficial in order to be proactive rather than reactive.  

The data set originally contained data on 146 "people" and 21 features.  18 of the people identified in the data set are considered POIs.  Upon inspection of the data, three "people" were removed from the data set.  One was a row that was read in that represented a total for the columns.  Another row represented a travel agency company.  There was also one individual (Eugene Lockhart) who did not have any values for any of the features.  After removal, this left 143 rows of data.  Missing values were replaced with 0.  This makes sense since this means that someone may have received a 0 dollar bonus.  Any statistical outliers (such as Kenneth Lay's total payments of over 103 million) were left in the data.  The reasoning is that values that are exorbitantly large may actually help to identify POIs.  Removing these outliers would potentially remove some of the POIs from the data set which would make prediction more difficult when creating a machine learning classifier.  Since the data contains such a small number of POIs, removing one would severaly hinder the predictive power of any models created.

#### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

Python was used to explore the data set [here](/final_project/exploratory/README.md).  Exploratory box plots were created for each of the features in the data set.  The goal was to examine the distributions of each of these features for POIs and non-POIs.  Upon viewing the box plots, features were selected on the basis of whether or not there appeared to be a difference between the distributions.  The initial features that appeared to show the strongest differences were bonus, exercised_stock_options, salary, shared_receipt_with_poi, and total_stock_value.  Several other features were identified as having somewhat of a difference.  Based on these insights, different combinations of features were tested until a suitable algorithm performance (greater than 0.3 precision and recall) was obtained.  The features that were selected were bonus, exercised_stock_options, and salary.  The box plots for these features are shown below for reference.
<table>
<tr>
<td><img src="/final_project/exploratory/output_7_0.png" height="300" width="300">
</td>
<td><img src="/final_project/exploratory/output_7_4.png" height="300" width="300">
</td>
<td><img src="/final_project/exploratory/output_7_14.png" height="300" width="300">
</td>
</tr>
</table>

Two features were created from the original features: ratios of bonus to salary and from_messages to to_messages.  The thought was that perhaps POIs tended to have large bonuses compared to their salaries.  Another thought was that the ratio for from_messages to to_messages would be smaller for POIs since they may receive more emails than they send.  The two ratios created did not have a discernible impact on algorithm performance and were not included in the final features used.

Scaling was not conducted for any of the features since regression, support vector machines, and clustering type algorithms were not used.

The following are the feature importances for the relevant algorithms:
<table>
<tr>
<td><b><u>Algorithm</u></b>
</td>
<td><b><u>bonus</u></b>
</td>
<td><b><u>exercised_stock_options</u></b>
</td>
<td><b><u>salary</u></b>
</td>
</tr>
<tr>
<td>DecisionTreeClassifier
</td>
<td>0.12
</td>
<td>0.42
</td>
<td>0.46
</td>
</tr>
<tr>
<td>RandomForestClassifier
</td>
<td>0.24
</td>
<td>0.44
</td>
<td>0.32
</td>
</tr>
<tr>
<td>AdaBoostClassifier
</td>
<td>0.15
</td>
<td>0.50
</td>
<td>0.35
</td>
</tr>
<tr>
<td>GradientBoostingClassifier
</td>
<td>0.26
</td>
<td>0.35
</td>
<td>0.39
</td>
</tr>
</table>
#### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I tested several algorithms (DecisionTreeClassifier, GaussianNB, RandomForestClassifier, AdaBoostClassifier, and GradientBoostingClassifier).  I also created a VotingClassifier that used each of the tuned classifiers.  The following table summarizes the performance of each of these algorithms when run alone.
<table>
<tr>
<td><b><u>Algorithm</u></b>
</td>
<td><b><u>Accuracy</u></b>
</td>
<td><b><u>Precision</u></b>
</td>
<td><b><u>Recall</u></b>
</td>
</tr>
<tr>
<td>DecisionTreeClassifier
</td>
<td>0.85
</td>
<td>0.51
</td>
<td>0.35
</td>
</tr>
<td>GaussianNB
</td>
<td>0.84
</td>
<td>0.48
</td>
<td>0.31
</td>
</tr>
<tr>
<td>RandomForestClassifier
</td>
<td>0.86
</td>
<td>0.58
</td>
<td>0.33
</td>
</tr>
<tr>
<td>AdaBoostClassifier
</td>
<td>0.82
</td>
<td>0.40
</td>
<td>0.32
</td>
</tr>
<tr>
<td>GradientBoostingClassifier
</td>
<td>0.86
</td>
<td>0.57
</td>
<td>0.36
</td>
</tr>
<tr>
<td>VotingClassifier
</td>
<td>0.87
</td>
<td>0.61
</td>
<td>0.34
</td>
</tr>
</table>


#### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

Tuning the parameters of an algorithm means to make minor changes to the various options to see how these changes affect the prediction power of the algorithm.  If this is not done correctly, the algorithm will not be as effective as it could be.  In order to tune the parameters of the various algorithms I tested, I used GridSearchCV to automatically find the best parameter options for each of the algorithms.  Once I had finalized my prediction features, each of the models was run using GridSearchCV to find the optimal parameters.  These parameters were then used for each of the models in the VotingClassifier.

#### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

Model validation is making sure the model actually works on unseen data.  This involves splitting up known data into a training set and a testing set.  A classic mistake is to use all of the data to train the model which will most likely cause overfitting.  The model will not do a good job of making predictions with unseen data since the model has not been validated against unseen data.  For feature selection and parameter tuning, a simple 70%/30% training/test split was used.  For the testing script, stratified shuffle split validation was used which basically means that the data was split into sets that contained approximately the same percentage of POIs and non-POIs that were in the original data set.  1000 folds were used to create 1000 different testing and training sets.

#### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The evaluation metrics I used to assess performance were accuracy, precision, and recall.  The metrics are summarized in the table above.  The accuracy describes the overall percentage of predictions that were correct.  The precision identifies what percentage of the positive predictions were correct.  The recall identifies what percentage of the actual positive cases were identified. In the context of the Enron data:

* Accuracy: How many of the POIs and non-POIs were accurately identified as POIs and non-POIs?
* Precision: What percentage of the predicted POIs were actually POIs?
* Recall: What percentage of the true POIs were predicted as POIs?

<table>
<tr>
<td>
</td>
<td><b><u>Predicted POI</u></b> 
</td>
<td><b><u>Predicted non-POI</u></b>
</td>
</tr>
<tr>
<td><b><u>Actual POI</u></b>
</td>
<td>True Positive: 681
</td>
<td>False Negative: 1319
</td>
</tr>
<tr>
<td><b><u>Actual non-POI</u></b>
</td>
<td>False Positive: 427
</td>
<td>True Negative: 10573
</td>
</tr>
</table>

The VotingClassifier algorithm was more precise than any of the other
algorithms meaning that there is more confidence that someone is a POI if
the alogithm identifies that person as a POI.  The algorithm predicted a total of 1108 POIs of which 681 were actually POIs. This is where the precision of 0.61 comes from.  The recall was similar to all
of the individual models and is somewhat low.  Out of the actual 2000 POIs in the data, the algoirthm was only able to identify 681 which is where the recall of 0.34 comes from.  Overall, the algorithm does not
do a good job of picking out the POIs from the data.

The overall accuracy of the model comes from the number of true positives and true negatives that were correctly identified.  In this case, 11254 out of the 13000 data points were correctly identified.

#### References
Information on the Enron scandal taken from https://en.wikipedia.org/wiki/Enron_scandal

Information to help explain accuracy, precision, and recall taken from 
<a href="http://www.kdnuggets.com/faq/precision-recall.html" target="blank">KD Nuggets
FAQ: How Are Precision and Recall Calculated?</a>