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


### Task 1: Convert the dictionary to pandas Dataframe

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


### Task 2: Remove outliers

# check data for outliers

#plt.figure()
#plt.scatter(dataset['salary'],dataset['total_payments'])
#plt.show(block=False)


# remove the max that corresponds to the "total"
index_max = dataset['salary'].argmax()
dataset=dataset.drop(index_max)


# remove non-numerical email_address
features_array=dataset.drop('email_address',axis=1)
# convert poi to numerics
features_array['poi']=features_array['poi']*1.


### Task 3: Create new feature(s)

# generate new features of ratio for email communication

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


plt.figure(tight_layout = True)
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


# calculate the correlation among different features to see which ones are more correlated with the POI

allfeatures = ['poi','salary','bonus']#, 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi',
               #'total_payments', 'expenses', 'total_stock_value', 'deferred_income', 'long_term_incentive',
               #'to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'ratio_from_message','ratio_to_message','other']


feature_corr = features_array[allfeatures].corr()

fig,plots = plt.subplots()
cbar_kws = {'orientation':"vertical", 'pad':0.025, 'aspect':70}
sns.heatmap(feature_corr, vmin=-0.5, vmax=0.5 ,annot=True, ax=plots,linewidths=0.3, fmt='.2f',cmap="PiYG",cbar_kws=cbar_kws)
plt.show()

#data_dict[key]['ratio_to_message']

### Store to my_dataset for easy export below.



### Extract features and labels from dataset for local testing
#features_list = ['poi','salary','bonus','total_payments']#,'from_this_person_to_poi','from_poi_to_this_person','ratio_from_message','ratio_to_message','shared_receipt_with_poi']
features_list = ['poi','from_this_person_to_poi','from_poi_to_this_person','ratio_from_message','ratio_to_message']#,'shared_receipt_with_poi']

# convert the dataframe back to nest-dictionary
my_dataset= features_array.to_dict(orient='index')#data_dict

data = featureFormat(my_dataset, features_list, sort_keys = False)

labels, features = targetFeatureSplit(data)

from sklearn.preprocessing import MinMaxScaler
### rescale feature data
scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree


clf_tree = tree.DecisionTreeClassifier()
clf_NB = GaussianNB()
clf_SVC = SVC(kernel="rbf",C=1000)

clf = clf_SVC
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
### cross_validation via train_test_split
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=40)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

clf.fit(features_train,labels_train)
pred_test = clf.predict(features_test)
pred_train = clf.predict(features_train)

### accuracy and precision of test samples
acc_test = accuracy_score(labels_test,pred_test)
prec_test= precision_score(labels_test,pred_test)

### accuracy and precision of train samples
acc_train = accuracy_score(labels_train,pred_train)
prec_train= precision_score(labels_train,pred_train)

print ("accuracy_train:",acc_train)
print ("precision_train:",prec_train)

print ("accuracy_test:",acc_test)
print ("precision_test:",prec_test)

print ("labels_test:",labels_test)
print ("labels_pred:",pred_test)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

 
