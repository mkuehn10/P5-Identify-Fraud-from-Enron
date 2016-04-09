P5 Identify Fraud from Enron Project
==============


* Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

* What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]


* What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I tested several algorithms (DecisionTreeClassifier, KNeighborsClassifier, SVC, GaussianNB, RandomForestClassifier, AdaBoostClassifier, and GradientBoostingClassifier).  I also created a VotingClassifier that used each of the tuned classifiers.  The following table summarizes the performance of each of these algorithms when run alone.
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
<tr>
<td>KNeighborsClassifier
</td>
<td>0.88
</td>
<td>0.69
</td>
<td>0.40
</td>
</tr>
<tr>
<td>SVC
</td>
<td>
</td>
<td>
</td>
<td>
</td>
</tr>
<tr>
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
<td>0
</td>
<td>
</td>
<td>
</td>
</tr>
</table>


* What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

* What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

* Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
