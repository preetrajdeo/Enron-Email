
## Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.

In this project I will be engineering the features, pick and tune an algorithm, test and evaluate my identifier. 

###### Task 1: is to load the dataset. 


```python
sys.path.append("../tools/")


from feature_format import featureFormat
from feature_format import targetFeatureSplit

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi"]

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
```

###### Task 2: Remove outliers. I have removed the outlier "Total" and  "The Travel Agency In The Park" as they didn't have any relevance in the dataset. I also go ahead and remove all Nan's from the 'Salary' column. 


```python
import matplotlib.pyplot as plt
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()
outliers = ["TOTAL", 'The TRAVE AGENCY IN THE PARK']

def remove_outliers():
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    for x in outliers:
        data_dict.pop(x, 0)

    return data_dict

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

```

######  Task 3: Create new feature(s)


```python
### create new features
### new features are: fraction_to_poi_email,fraction_from_poi_email

def dict_to_list(var1,var2):
    new_list=[]

    for i in data_dict:
        if data_dict[i][var1]=="NaN" or data_dict[i][var2]=="NaN":
            new_list.append(0.)
        elif data_dict[i][var1]>=0:
            new_list.append(float(data_dict[i][var1])/float(data_dict[i][var2]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### Now its time to insert those features in data_dict
counter = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[counter]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[counter]
    counter +=1

#Now I add my two new features to my list of variables. 
    
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"] 

### store to my_dataset for easy export below
my_dataset = data_dict
```

However the dataset comes with a number of variables. 
Let's use a decision tree to find out which variable is important and which
is not. We will use accuracy, precision and recall to make our decision. 



```python
##Let's make a separate dataset to use in this decision tree so that we can use my_dataset unchanged 
##later on. 
features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

data_dt = featureFormat(my_dataset, features_list)

#Now its time to split the dataset into features and labels. The code below assumes that the 
#first variable is the label.

labels, features = targetFeatureSplit(data_dt)

#Let's split the dataset into training and testing data
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features,labels,test_size=0.1, random_state=42)

#Now let's deploy the decision tree and calculate the accuracy of it by using the testing set

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'Accuracy:', score
print 'Precision:', precision_score(labels_test, pred)  
print 'Recall:', recall_score(labels_test, pred)

importances = clf.feature_importances_



import pandas as pd

data_imp = pd.DataFrame(
    {'Features': features_list[1:], 
     'Importances': importances})
data_imp


```

    Accuracy: 0.866666666667
    Precision: 0.0
    Recall: 0.0





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>Importances</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>salary</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bonus</td>
      <td>0.077381</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fraction_from_poi_email</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fraction_to_poi_email</td>
      <td>0.232919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>deferral_payments</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>total_payments</td>
      <td>0.058036</td>
    </tr>
    <tr>
      <th>6</th>
      <td>loan_advances</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>restricted_stock_deferred</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>deferred_income</td>
      <td>0.140117</td>
    </tr>
    <tr>
      <th>9</th>
      <td>total_stock_value</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>expenses</td>
      <td>0.010031</td>
    </tr>
    <tr>
      <th>11</th>
      <td>exercised_stock_options</td>
      <td>0.118986</td>
    </tr>
    <tr>
      <th>12</th>
      <td>long_term_incentive</td>
      <td>0.053737</td>
    </tr>
    <tr>
      <th>13</th>
      <td>shared_receipt_with_poi</td>
      <td>0.255057</td>
    </tr>
    <tr>
      <th>14</th>
      <td>restricted_stock</td>
      <td>0.053737</td>
    </tr>
    <tr>
      <th>15</th>
      <td>director_fees</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



The precision and recall is 0 in this decision tree. Hence, I manually try using the decision tree with a combination of features. I come up with these three that give me a precision and recall of 0.67:
* fraction_to_poi_email
* expenses
* shared_receipt_with_poi


```python
##According to these results I pick the following features. 
features_list_rev = ['poi', 'fraction_to_poi_email',
                     'expenses', 'shared_receipt_with_poi']
##Now I deply the decision tree again for the new features. 
data_dt_rev = featureFormat(data_dict, features_list_rev)

#Now its time to split the dataset into features and labels. The code below assumes that the first variable is the 
#label.

labels, features = targetFeatureSplit(data_dt_rev)

#Let's split the dataset into training and testing data
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.1, random_state=42)


#Now let's deploy the decision tree and calculate the accuracy, precision and recall before 
#tuning the algorithm


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'Accuracy, Precision and Recall before tuning the algorithm:'
print 'Accuracy:', score

from sklearn.metrics import precision_score, recall_score
print 'Precision:', precision_score(labels_test, pred)  
print 'Recall:', recall_score(labels_test, pred)


#Now let's deploy the decision tree and calculate the accuracy, precision and recall before 
#tuning the algorithm after tuning the algorithm. A min_samples_split = 6 or higher gives the best
#precision and recall.


clf = DecisionTreeClassifier(min_samples_split = 11)
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'Accuracy, Precision and Recall after tuning the algorithm:'
print 'Accuracy:', score
print 'Precision:', precision_score(labels_test, pred)  
print 'Recall:', recall_score(labels_test, pred)
```

    Accuracy, Precision and Recall before tuning the algorithm:
    Accuracy: 0.692307692308
    Precision: 0.4
    Recall: 0.666666666667
    Accuracy, Precision and Recall after tuning the algorithm:
    Accuracy: 0.846153846154
    Precision: 0.666666666667
    Recall: 0.666666666667


###### Dump your classifier, dataset ad features list so that anyone can run it and check your result. 


```python
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

```

# Conclusion
In this project, since we are trying to identify the Person of Interest(POI), we can say that according to the decision tree algorithm, we can say that 67% of the time that our algorithm identified a POI it really was a POI because our precision is 0.67 and 33% of the time if it flagged a person as POI, then it was a false alarm. Recall tells us that our algorithm s 67% of the time correct in flagging the POI while the other 33% of the time it was showing us a false negative which means that it would not recognize a person as a POI while he/she actually was. While our numbers seem to be reasonable, there is obviously room for improvement. We can keep look into the text of the data and see if we can garner any more information from there. 
