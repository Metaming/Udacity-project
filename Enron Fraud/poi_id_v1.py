#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#%%
################# Task 1: Convert the dictionary to pandas Dataframe

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# convert the dictionary to pandas dataframe
dataset = pd.DataFrame.from_dict(data_dict, orient='index')


# columns of dataframe
features_list = list(dataset.columns)

# convert string type NaN to numpy.NaN
for column in dataset.columns:
    dataset[column] = dataset[column].apply(lambda x: np.NaN if x == 'NaN' else x)
    
#print dataset.info()

dataset.fillna(0,inplace=True)

#print dataset.head()


######################## Task 2: Remove outliers

#%%
 
#check data for outliers

plt.figure()
plt.scatter(dataset['salary'],dataset['total_payments'])
plt.show(block=False)

# remove the max that corresponds to the "total"
index_max = dataset['salary'].argmax()
dataset=dataset.drop(index_max)


# remove non-numerical email_address
features_array=dataset.drop('email_address',axis=1)
# convert poi to numerics
features_array['poi']=features_array['poi']*1.


############################# Task 3: Create new feature(s)


# method 1: generate new features based on rational thinking: the ratio for email communication among POI compared to Non-POI

for index in features_array.index:
    if features_array.loc[index,'from_this_person_to_poi']== 'NaN' or features_array.loc[index,'from_messages']== 'NaN'or features_array.loc[index,'from_messages']== 0:
        features_array.loc[index,'ratio_from_message']=0
    else:
        features_array.loc[index,'ratio_from_message'] = float(features_array.loc[index,'from_this_person_to_poi'])/ features_array.loc[index,'from_messages']

    if features_array.loc[index,'from_poi_to_this_person']== 'NaN' or features_array.loc[index,'to_messages']== 'NaN'or features_array.loc[index,'to_messages']== 0:
        features_array.loc[index,'ratio_to_message']=0
    else:
        features_array.loc[index,'ratio_to_message'] = float(features_array.loc[index,'from_poi_to_this_person'])/ features_array.loc[index,'to_messages']

# plot selected features for POI and Non-POI    
poi_index = (features_array['poi']== True)
npoi_index = (features_array['poi']== False)  


plt.figure(tight_layout = True,figsize=(10,8))
plt.subplot(2,2,1)
plt.scatter(features_array['salary'][(poi_index)],features_array['bonus'][(poi_index)],c='r',s=50)
plt.scatter(features_array['salary'][(npoi_index)],features_array['bonus'][(npoi_index)])
plt.legend(['poi','non-poi'])
plt.xlabel('salary')
plt.ylabel('bonus')

plt.subplot(2,2,2)
plt.scatter(features_array['from_this_person_to_poi'][(poi_index)],features_array['from_poi_to_this_person'][(poi_index)],c='r',s=50)
plt.scatter(features_array['from_this_person_to_poi'][(npoi_index)],features_array['from_poi_to_this_person'][(npoi_index)])
plt.legend(['poi','non-poi'])
plt.xlabel('from_this_person_to_poi')
plt.ylabel('from_poi_to_this_person')

plt.subplot(2,2,3)
plt.scatter(features_array['ratio_from_message'][(poi_index)],features_array['ratio_to_message'][(poi_index)],c='r',s=50)
plt.scatter(features_array['ratio_from_message'][(npoi_index)],features_array['ratio_to_message'][(npoi_index)])
plt.legend(['poi','non-poi'])
plt.xlabel('ratio_from_message')
plt.ylabel('ratio_to_message')

plt.subplot(2,2,4)
plt.scatter(features_array['total_payments'][(poi_index)],features_array['shared_receipt_with_poi'][(poi_index)],c='r',s=50)
plt.scatter(features_array['total_payments'][(npoi_index)],features_array['shared_receipt_with_poi'][(npoi_index)])
plt.legend(['poi','non-poi'])
plt.xlabel('total_payments')
plt.ylabel('shared_receipt_with_poi')

plt.show()

#%%
# calculate the correlation among different features to see which ones are more correlated with the POI

allfeatures = ['poi','salary','bonus', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi','loan_advances',
               'total_payments', 'expenses', 'total_stock_value', 'deferred_income', 'long_term_incentive',
               'to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'ratio_from_message','ratio_to_message','other']


feature_corr = features_array[allfeatures].corr()

fig,plots = plt.subplots(figsize=(12,10))
cbar_kws = {'orientation':"vertical", 'pad':0.025, 'aspect':70}
sns.heatmap(feature_corr, annot=True, vmin=-0.8, vmax=0.8 , ax=plots,linewidths=0.3, fmt='.2f',cmap="PiYG",cbar_kws=cbar_kws)
plt.show()



#%%

# get the correlation of poi with other features
corr_poi=feature_corr['poi']
X=range(len(corr_poi))

# sort the correlation
sorted_corr = np.sort(corr_poi)
# get the index of sorting,
# Attention: np.argsort does not give the sorting index but the position of the sorted 
# element in the original array. Need to use twice np.argsort to get the sorting index 
# of the element in the original array

sorted_index = np.argsort(np.argsort(corr_poi))
#sorted_index = np.argsort(corr_poi.values)

plt.figure(figsize=(8,6))
plt.bar(X,sorted_corr)
plt.xticks(sorted_index,sorted_index.index,rotation=90)
    
plt.show()
 
 
#%%
 
############# Task 4: feature selection and principle component analysis

# Since the number of poi is small, it is necessary to reduce the number of features to avoid overfitting

# The first handpicked approach is to keep the features that have a correlation coefficient higher than certain value
features_kept=pd.DataFrame()

for col in allfeatures:
    
    if abs(corr_poi[col])>=0.2:
        features_kept[col] = features_array[col]
        
print "features_kept based on correlations:\n",list(features_kept.columns)

#%%
# The second approach to reduce feature dimension is via univariant feature selection
from sklearn.feature_selection import SelectPercentile, f_classif

# choose all features except 'poi'
selectorfeatures = features_array.drop('poi',axis=1)
# choose features 'poi' as labels
selectorlabels = features_array['poi']

# use SelectPercentile to choose the top X percent features that return the highest p-values in the F-test
# f_classif is the classification based on f-test that compute the ANOVA (Analysis of variance) F-value, which then gives the p-value
selector = SelectPercentile(f_classif,percentile=30)
selector.fit(selectorfeatures,selectorlabels)

selected_features = selector.fit_transform(selectorfeatures,selectorlabels)

#print selectorfeatures.shape
#print selected_features.shape


scores = -np.log10(selector.pvalues_)
scores/=scores.max()

plt.figure(num=5)
X_feat = range(len(selectorfeatures.columns))

# sort the scores in decending order
sorted_scores = np.flipud(np.sort(scores))
sorted_ind = np.flipud(np.argsort(scores))
#sorted_ind = np.flipud(np.argsort(scores))

plt.figure(tight_layout = True,figsize = (8,10))
plt.subplot(2,1,1)
plt.bar(X_feat,scores)
plt.xticks(X_feat,list(selectorfeatures.columns),rotation=90)
plt.ylabel('Norm. log (p values)')

plt.subplot(2,1,2)
plt.bar(X_feat,sorted_scores)
plt.xticks(X_feat,list(selectorfeatures.columns[sorted_ind]),rotation=90)
plt.ylabel('Norm. log (p values)')
plt.show()

#%%
features_kept2 = pd.DataFrame()
features_kept2['poi']=features_array['poi']

for ind,pvalue in enumerate(scores):
    
    if pvalue>=0.5:
        index = selectorfeatures.columns[ind]
        #print index,pvalue
        features_kept2[index] = features_array[index]
        
print "features_kept based on univariant feature selection (p value):\n",list(features_kept2.columns) 

#%%  Preparation of train and test data sets

### Extract features and labels from dataset for local testing
#features_list = ['poi','salary','bonus','total_payments']#,'from_this_person_to_poi','from_poi_to_this_person','ratio_from_message','ratio_to_message','shared_receipt_with_poi']
#features_list = ['poi','from_this_person_to_poi','from_poi_to_this_person','ratio_from_message','ratio_to_message']#,'shared_receipt_with_poi']

features_list = list(features_kept2.columns)

# convert the dataframe back to nest-dictionary
# and Store to my_dataset for easy export below.

my_dataset= features_array.to_dict(orient='index')

data = featureFormat(my_dataset, features_list, sort_keys = False)

# split data to features and labels
labels, features = targetFeatureSplit(data)


# rescale feature data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)

#%%
################## Taks 5: Machine learning model testing and tuning

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

 
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#%%
#  cross_validation via train_test_split, note that the split is defined by the random_state number and test_size
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=100)

#clf_tree = tree.DecisionTreeClassifier()
#clf_NB = GaussianNB()
#clf_SVC = SVC(kernel="rbf",C=1000)

# split features and labels to train and test sets

target_names=['Non-POI','POI']


scoring = 'f1'
scoring = 'recall'
scoring = 'precision'

#%%  SVC model
# define the parameter grid for svc model tuning
pm_grid_SVC = {
         'C': [5e2,1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }

# define the stratified shuffle split
cv = StratifiedShuffleSplit(labels_train, n_iter=5, test_size=0.3, random_state=50)

# define the gridsearch cross validation for the SVC classification
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto', scoring defines the metrics used for evaluation 
clf_SVC = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid = pm_grid_SVC, cv = cv, scoring=scoring,error_score=0)
clf_SVC.fit(features_train, labels_train)

print '________________________________________________'
print "Best estimator found by grid search for SVC model:"
print clf_SVC.best_estimator_

print '________________________________________________'
pred_train = clf_SVC.predict(features_train)
print 'classification_report for SVC model: train sets'

print classification_report(labels_train, pred_train, target_names=target_names)
print '________________________________________________'
pred_test = clf_SVC.predict(features_test)
print 'classification_report for SVC model: test sets'
print classification_report(labels_test, pred_test, target_names=target_names)
#%%  Decision Tree model
# define the parameter grid for svc model tuning
pm_grid_tree = {
         'min_samples_split': [2,6, 10, 15, 20, 25],         
          }
 
clf_tree = GridSearchCV(tree.DecisionTreeClassifier(criterion='gini', splitter='best'), pm_grid_tree)
clf_tree = clf_tree.fit(features_train, labels_train)

print '________________________________________________'
print "Best estimator found by grid search for DecisionTree model:"
print clf_tree.best_estimator_

print '________________________________________________'
print "Score of parameters by grid search for DecisionTree model:"

print clf_tree.best_score_


print '________________________________________________'
pred_train = clf_tree.predict(features_train)
print 'classification_report for DecisionTree model: train sets'
print classification_report(labels_train, pred_train, target_names=target_names)
print '________________________________________________'
pred_test = clf_tree.predict(features_test)
print 'classification_report for DecisionTree model: test sets'
print classification_report(labels_test, pred_test, target_names=target_names)

#print confusion_matrix(labels_test, pred_test, labels=range(len(target_names)))


#%%
################# Principle component analysis and data pipeline
### Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA

SVC = SVC(kernel="rbf")
PCA = RandomizedPCA()

pipe = Pipeline(steps=[('PCA',PCA),('SVC',SVC)])

features_full = features_array.drop('poi',axis=1).values
features_full = scaler.fit_transform(features_full)
print features_full[0] 

PCA.fit(features_full)

plt.figure()
plt.plot(PCA.explained_variance_)
plt.axis('tight')
plt.show()

#%% 

features_train, features_test, labels_train, labels_test = train_test_split(features_full, labels, test_size=0.30, random_state=100)


n_components = [5,10]
C_SVC =  [5e2,1e3, 5e3, 1e4, 5e4]   # large C is more accurate in fitting but could overfit
gamma_SVC =  [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05] # large gamma takes more weight on the nearest datapoints around the boundary

# define the stratified shuffle split
cv = StratifiedShuffleSplit(labels_train, n_iter=10, test_size=0.3, random_state=5)

scoring = 'f1'
clf_PCA_SVC = GridSearchCV(pipe, dict(PCA__n_components = n_components,
                                     SVC__C=C_SVC, SVC__gamma=gamma_SVC),scoring=scoring,cv=cv)

clf_PCA_SVC.fit(features_train,labels_train)


print '________________________________________________'
print 'best estimator:\n',clf_PCA_SVC.best_estimator_


print '________________________________________________'
pred_train = clf_PCA_SVC.predict(features_train)
print 'classification_report for PCA-SVC model: train sets'

print classification_report(labels_train, pred_train, target_names=target_names)
print '________________________________________________'
pred_test = clf_PCA_SVC.predict(features_test)
print 'classification_report for PCA-SVC model: test sets'
print classification_report(labels_test, pred_test, target_names=target_names)

#%%
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

 
