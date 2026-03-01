import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

'''
1. Define the problem: Who is most likely to survive?
    - predict who is most likely to survive based on person's information
    - which features are important, extact only relevant features
    - based on the features, create a machine learning model where we provide 
      information and predicts if this person is likey to survive or not
'''

'''
2. Data Loading
    - Convert the csv file data given by Kaggle into pandas dataframe.
        1. Import necessary python libraries (pandas, numpy, tabulate - for the table)
        2. Download dataset from Kaggle and save in same folder directory
        3. use pd.read_csv for csv -> panda dataframe cpmversopm
'''

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
gender_data = pd.read_csv("gender_submission.csv")

# print(tabulate(train_data.head(5), headers = "keys", tablefmt = "psql"))

# print(gender_data.head())

'''
3. Understanding the data
    - Understand what the data means from kaggle. Their definition and meaning
'''

''' understand how data is formatted using pandas'''
# print(train_data.info())
'''info? null values? value counts? etc.'''

'''
There are 891 entries / people in total.
Age, Cabin, and Embarked have null values
'''
# print(train_data.isnull().sum())
'''
Age null values: 177, Cabin null values: 687, Embarked null value: 2
This means:
- we dont know the age of 177 people
- we dont know the cabin of 687 people, probably because they are families and are sharing cabins
- 2 people did not embark so that means that 2 people did not get on at all.
'''
# print(train_data.value_counts("Sex"))
# print(train_data.value_counts("Age"))
# print(train_data.value_counts("SibSp"))
# print(train_data.value_counts("Parch"))
# print(train_data.value_counts("Fare"))
# print(train_data.value_counts("Cabin"))
# print(train_data.value_counts("Embarked"))

'''
Summary of findings:

    Null values
    - Dont know age for 177 people
    - Dont lnow cabin of 687
    - Dont know embarkef for 2 people

    Sex
    - 577 male
    - 314 female

    Age
    - the mode is 24 years old
    - Most people are around 20 -30 years old

    Sibsp
    - most people have no siblings / spouse with them
    - there are less and less people with more sibsp
    - total does add up to 891

    Parch
    - most people dont have a parch prelationship
    - there are less and less people with parch
    - total does add up to 891

    Fare
    - mode was $8.05
    - while the mode was $8.05, there was alot of very specific prices like 8.112 and 8.137
    
    Cabin
    - maximum was 4 people per cabin

    Embarked
    - most embarked from s
    - few from c
    -very little from q

What data matters:
1. sex
2. pclass
3. age
4. fare
5. sibsp
6. parch
7. embarked
'''

'''
4. EDA (exploratory data analysis) and understanding of data via data visulaization 
    - libraries often used for data visualization:
        * pandas.dataframe.plot
        * matplotlib
        * seaborn
         -> statisitc plot provided exlcusively by seaborn
         -> clean graphs
         -> "palette" functionality for grpahs to be prettier
         -> highly compatable with pandas dataframe
'''

'''
Categorical Data
    what is categorical data?
        * categorical data is qualitative characterisitcs
        * sex, cabin, embarked, survived
'''

'''
4.1 Sex
'''

''' % of men who survied'''
total_male = train_data[train_data["Sex"] == "male"]
male_which_survived = total_male[total_male["Survived"] == 1]
male_survival_percentage = len(male_which_survived) / len(total_male)
#print(male_survival_percentage)

''' % of women who survived'''
total_female = train_data[train_data["Sex"] == "female"]
female_which_survived = total_female[total_female["Survived"] == 1]
female_survival_percentage = len(female_which_survived) / len(total_female)
#print(female_survival_percentage)

'''Function to make the pie chart'''
def pie_chart(feature):
    feature_ratio = train_data[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train_data[train_data['Survived'] == 1][feature].value_counts()
    dead = train_data[train_data['Survived'] == 0][feature].value_counts()
    
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()
    
    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')
    
    plt.show()

# pie_chart("Sex")
# sns.countplot(data = train_data, x = "Survived", hue = "Sex")
# plt.show()

'''
Findings for Sex
    - Male to Female retio: 64.8% male, 35.2% female
    - Male survival rate: 18.9% // Male death rate: 81.1%
    - Female survival rate: 74.2% // Female death rate: 74.2
'''

'''
4.2 Pclass
'''

# pie_chart("Pclass")
# sns.countplot(train_data, x = "Survived", hue = "Pclass")
# plt.show()

'''
Findings for Pclass
    - 24.2% in Pclass 1
    - 20.7% in Pclass 2
    - 55.1% in Pclass 3
    - Pclass 1 survival rate: 63.0% // death rate: 37%
    - Pclass 2 survival rate: 47.3% // death rate: 52.7%
    - Pclass 3 survival rate: 24.2% // death rate: 75.8%
    
    - Most people were in class 3, and most of them died
    - Class 2 was about 50 / 50
    - More people survived in class one than died. Only one that has higher than 50%
'''

'''4.3 Embarked'''
# pie_chart("Embarked")
# sns.countplot(train_data, x = "Survived", hue = "Embarked")
# plt.show()

'''
Findings on embarked
    - 75% embarked from s
    - 18.9% embarked from c
    - 8.7 embarked from q
    - S survival rate: 33.7% // death rate: 66.3%
    - C survival rate: 55.4% // death rate: 44.6%
    - Q survival rate: 39.0% // death rate: 61.0%

    - Even though most people died in S, most people also survived from s
    - Only people in C survived more than died
'''

'''4.4 Sibsp & Parch'''
'''SibSp'''

# survived = train_data[train_data['Survived'] == 1]["SibSp"].value_counts()
# dead = train_data[train_data['Survived'] == 0]["SibSp"].value_counts()
# df = pd.DataFrame([survived, dead])
# print(df)

# df.plot(kind = "bar", stacked = True, figsize = (10,5))
# plt.show()


def bar_chart(feature):
    survived = train_data[train_data['Survived']==1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.show()

# bar_chart("SibSp")

# SibSp_survive_rate = train_data[["SibSp", "Survived"]].groupby(["SibSp"]).mean().sort_values(by = "Survived", ascending = False)
'''
1. Data selection:
    train_data[["SibSp", "Survived"]] -> selects two columns of SibSp and Survived from the datatframe

2. Grouping: 
    .groupby(["Sibsp]) -> groups the data by SibSp column. All the rows with the same value in SibSp are collected together.

3. Calculating mean:
    .mean() -> for each group (each value of sibsp, so 0 or 1), calculates the mean of survived column.

4. Sorting:
    .sort_values(by = "Survived, ascending = False) -> sorts the resulting means in decending order

'''
# print(SibSp_survive_rate)

''' groupby sibsp and survived to see how many people survived and did not for each feature'''
# SibSp_survive_rate_with_features = train_data.groupby(["SibSp", "Survived"]).count()
# print(SibSp_survive_rate_with_features)

'''draw a seaborne factor plot'''
# sns.catplot(x = "SibSp", y = "Survived", data = train_data, kind = "point", aspect = 2.5)
# plt.show()

'''
Findings on Sibsp
    - 0 SibSp survival rate: 34.54%
    - 1 SibSp survival rate: 53.59%
    - 2 SibSp survival rate: 46.43%
    - 3 SibSp survival rate: 25.00%
    - 4 SibSp survival rate: 16.67%
    - 5 SibSp survival rate: 0%
    - 6 SibSp survival rate: n/a
    - 7 SibSp survival rate: n/a
    - 8 SibSp survival rate: 0%
    
    - The most people that survived did not have SibSp, but the opposite is true, The most people that dided did not have a SibSp also.
    - More people died with zero SibSp than survived. Ratio is about 1/3 survived.
    - 0 SibSp had the most amount of people
    
    - Best survival rate order: 1,2,0,3,4,5,8
'''

'''Parch'''
# bar_chart("Parch")

'''group by parch and survived'''
# print(train_data[["Parch", "Survived"]].groupby(["Parch"]).mean().sort_values(by = "Survived", ascending = False))

'''factorplot'''
# sns.catplot(x = "Parch", y = "Survived", data = train_data, kind = "point", aspect = 2.5)
# plt.show()

'''total number of people in each parch '''
# print(train_data["Parch"].value_counts())
'''
tells how much of each value there are.
'''

'''
Findings on Parch
    - Most (678) people had 0 parch
    - Second (118) most was 1 parch
    - Third (80) most was 2 parch
    - Fourth tied (5) most was 5 parch
    - Fourth tied (5) was 3 parch
    - Sixth (4) most was 4 parch 
    - Seveth (1) was 6 parch
    - Most people in order decending: 0,1,2,5,3,4,6


    - People with 3 parch had 60% survival rate
    - People with 1 parch had 55% survival rate
    - People with 2 parch had 50% survival rate
    - People with 0 parch had 34% survival rate
    - People with 5 parch had 20% survival rate
    - People with 4 parch had 0% survival rate
    - People with 6 parch had 0% survival rate
    - no one had 7
    -Survival rate in order decending: 3,1,2,0,5,4,6,

    - data does not correlate with sibsp
    - 0, 4, 5, 6 have under 50% chance survival
    - 1, 2, 3 have over 50% chance survival
'''

'''Numerical Data'''

'''Age'''

data = [train_data, test_data]
for dataset in data:
    mean = train_data["Age"].mean()

    '''Calculating the standard deviation in test_data. How speadout the data is'''
    # std = test_data["Age"].std()

    '''stores the number of null values to the dataset.
    Either train or test, depending on which dataset we are using.
    Used to determine how much numbers we have to generate ourselves'''
    # is_null = dataset["Age"].isnull().sum()

    '''Generating random ages with integers between mean - std and mean + std.
    Creates random values ne standard deviation above and below the mean'''
    # rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    '''size = is_null is telling it how much to generate. It will generate as much numbers as null values'''

    '''fill NaN values in Age column with random generated values'''
#     age_slice = dataset["Age"].copy()
#     age_slice[np.isnan(age_slice)] = rand_age
#     dataset["Age"] = age_slice
#     dataset["Age"] = train_data["Age"].astype(int)

# survived = 'survived'
# not_survived = 'not survived'
# fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))
# women = train_data[train_data['Sex']=='female']
# men = train_data[train_data['Sex']=='male']
# ax = sns.histplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False, color="green")
# ax = sns.histplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False, color="red")
# ax.legend()
# ax.set_title('Female')
# ax = sns.histplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False, color="green")
# ax = sns.histplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False, color="red")
# ax.legend()
# _ = ax.set_title('Male');
# plt.show()

# sns.histplot(train_data.Age.dropna())
# plt.show()

# g = sns.FacetGrid(train_data, col = "Survived")
# g.map_dataframe(sns.histplot, x = "Age")
# plt.show()

'''Fare'''

# sns.histplot(train_data.Fare.dropna())
# plt.show()

# num = sns.FacetGrid(train_data, col="Survived")
# print(num.map(plt.hist, "Fare", bins = 20))
# plt.show()

# num = sns.FacetGrid(train_data, col="Survived")
# print(num.map(plt.hist, "Fare", bins = 10))
# plt.show()

'''
Conclusion

    What does age and fare mean?

    age
    - most people were between the ages of 20-40
    - most of the younger people survived

    fare
    - most people paid about $50
    - most people that died paid the cheapest fare
    - porportionate to survalal chance and price paid, meaning the more people in a group, the more people survived.
'''

'''5. Feature Engineering'''

'''
    - pick the features that are used to train the machine learning model
    - the fearures that will be considered are Name, Sex, Embarked, Age, SibSp, Parch, Fare, and Pclass
    - Ticket and Cabin are not included because we could not find a value for them
'''

'''Since we have to process the train and test data the same, we are going to combine them'''
train_and_test = pd.concat([train_data, test_data], ignore_index=True)
# print(train_and_test)

'''
5.1 Name  Feature
We are going to make use of each persons honorifics or title. With it, we can identify their gender, maritial status, and age group.
We already have gender and age, so it might not be that useful, but we are going to add it anyways. We will make a new column called title 
'''

# print(train_and_test.head())

df1 = train_and_test.Name.str.split(",", expand=True)
df2 = df1[1].str.split(".", expand=True)
title = df2[0]
train_and_test.insert(3, "Title", title)
# print(train_and_test)

title_count = pd.crosstab(train_and_test["Title"], train_and_test['Sex'])
# print(title_count)

title_survival = train_and_test[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()
# print(title_survival)

'''going to replace titles with little ammount to others'''
train_and_test["Title"] = train_and_test["Title"].replace(["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess"], "Others")
'''going to replace the girl titles to Miss'''
train_and_test["Title"] = train_and_test["Title"].replace(["Mle", "Mme", "Ms"], "Miss")
# print(train_and_test)

train_and_test = [train_data, test_data]
# print(train_and_test)

traincopy = train_data

traincopy["Title"] = traincopy.Name.str.extract('^([^,]*)')
# print(traincopy)

for dataset in train_and_test:
    dataset["Title"] = dataset.Name.str.extract('([A-Za-z]+)\.')
# print(train_data.head(5))

new = dataset["Name"].str.split(",", n=-1, expand=True)
new2 = new[1].str.split(".", n=-1, expand=True)
# print(new2)

# print(pd.crosstab(train_data["Title"], train_data["Sex"]))

'''Replace the uncommon titles with others. Combine the same titles together, ex: miss and ms'''
for dataset in train_and_test:
    dataset["Title"] = dataset["Title"].replace(["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "Countess"], "Others")
    dataset["Title"] = dataset["Title"].replace(["Ms", "Mlle"], "Miss")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")
#print(train_data[["Title", "Survived"]].groupby(["Title"], as_index=False).mean())

for dataset in train_and_test: 
    dataset["Title"] = dataset["Title"].astype(str)

'''
5.2 Sex Feature
    - handling the sex feature -> gender
    - since it is already just male and female, we just have to convert it into a string type
'''

for dataset in train_and_test:
    dataset["Sex"] = dataset["Sex"].astype(str)

train_data["Sex"] = train_data["Sex"].astype(str)
train_data["Sex"].dtype # returns object which is a string

'''
5.3 Embarked Feature
    - embarked which is the port that a person left from
    - from the data overview from before, we found that there are some null values
'''

embarked_null = pd.isnull(train_data)["Embarked"].sum()
# print(embarked_null)

# print(train_data.Embarked.value_counts(dropna=False))

'''
The most common group is S, so it is highly likey that the 2 missing values are also S.
We will replace the missing value to S.
'''

for dataset in train_and_test:
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    dataset["Embarked"] = dataset["Embarked"].astype(str)
# print(train_data.Embarked.value_counts())

'''
5.4 Age
    - About 20% of the age data is NaN, we need a way to fill in the data ourselves.
    - There are 4 main ways to do this:
        1. Fill with the overall mean value
        2. Fill with random values drawn from normal distribution N
        3. Group the data by using subgroups (other features) adn calculate the mean
        4. Group data by using subgroups and fill with random value for each group.

'''
#print(train_data["Age"].isnull().sum())

# train = ["Gender", "Pclass", "Title"]
# print(train)

'''this dataframe contains the age and the categorical values'''
categorical = ["Sex", "Pclass","Title"]
df_age = train_data[categorical + ["Age"]]
# print(df_age)

agedata = train_data.groupby(by = ["Sex", "Pclass", "Title"], as_index = True)["Age"].mean()
# print(agedata)

df_age_mean = round(df_age.dropna().groupby(categorical, as_index=True).median(), 1)
# print(df_age_mean)

'''
The function takes an input the categorical variables and it returns the average age given the categorical varaibes
'''
def get_age(var, sex, pclass, title):
    if np.isnan(var):
        mean = df_age_mean["Age"][sex][pclass][title]
    else:
        mean = var
    return mean

df_age = df_age.copy()
df_age['Age'] = df_age.apply(lambda x: get_age(x['Age'], x['Sex'], x['Pclass'], x['Title']), axis=1)
df_age[df_age['Age'].isna()]
# print(df_age.isnull().sum())

'''Since all NaN values have been filled using method (3), let's update the train_data and test_data datasets.'''
for dataset in train_and_test:
    dataset['Age'] = train_data.apply(lambda x: get_age(x.Age, x.Sex, x.Pclass, x.Title), axis=1)

'''
There are several ways to convert numeric data into features, and in this project, we used a method called Binning.
(Binning is a technique where you define ranges or categories for various types of data, creating a smaller number of groups compared to the original data.)
In this case, we used pd.cut() to create five groups with equal-length intervals.
'''

'''bining'''
train_and_test = [train_data, test_data]
for dataset in train_and_test:
    dataset["Age"] = dataset["Age"].astype(int)
    train_data["AgeBand"] = pd.cut(train_data["Age"], 5)
    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
    dataset.loc[dataset["Age"] > 64, "Age"] = 4
    dataset["Age_str"] = dataset["Age"].map({0: "child", 1: "young", 2: "middle", 3: "prime", 4: "old"}).astype(str)

# print(train_data)

'''5.5 Fare Feature'''

'''
There was only one NaN value for fare feature in the test dataset. We are going to assume that fare and pclass
have a correlation. Because of that, we are going to replace the fare with the average price corresponding to pclass
'''

'''finding the average fare for pclass'''
average_Pclass_Fare = train_data[["Pclass", "Fare"]].groupby(["Pclass"], as_index=False).mean()
# print(average_Pclass_Fare)

'''Find where the NaN value is'''
# print(train_data["Fare"].isna())

'''Fill in the NaN value with the average price'''
for dataset in train_and_test:
    dataset["Fare"] = dataset["Fare"].fillna(13.675)

'''
We will be binning fare similarly as to how we did with age
    - Fare <= 7.854: Group 0
    - 7.854 < Fare <= 10.5: Group 1
    - 10.5 < Fare <= 21.679: Group 2
    - 21.679 < Fare <= 39.688: Group 3
    - 39.688 < Fare: Group 4
'''

for dataset in train_and_test:
    dataset.loc[dataset["Fare"] <= 7.854, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.854) & (dataset["Fare"] <= 10.5), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 10.5) & (dataset["Fare"] <= 21.679), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 21.679) & (dataset["Fare"] <= 39.688), "Fare"] = 3
    dataset.loc[dataset["Fare"] > 39.688, "Fare"] = 4
    dataset["Fare"] = dataset["Fare"].astype(int)

'''5.6 SibSp & Parch features (relatives)'''
'''
From what we have observed about the data earlier, the more relatives, the higher the chance of survival.
We will make a relative feature by adding SibSp and Parch
'''

for dataset in train_and_test:
    dataset["Relatives"] = dataset["Parch"] + dataset["SibSp"]
    dataset["Relatives"] = dataset["Relatives"].astype(int)
# print(train_data.head())

'''5.7 Feature Extraction and Remaining Preprocessing'''
'''
keep only the features that will be used in the model
'''
train_wanted_features = ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "Relatives"]
test_wanted_features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "Relatives"]

train_data = train_data[train_wanted_features]
test_data = test_data[test_wanted_features]
# print(test_data)

'''One Hot Encoding'''

'''
In machine learning models, computers cannot cannot train dirrectly on text(str) values.
We have to convert each data into an integer value. One Hot Encoding is where is category is
represented as binary (0 or 1)
'''

'''one hot encoding for categorical variables'''
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

train_label = train_data["Survived"]
train_dataset = train_data.drop("Survived", axis=1)
test_dataset = test_data.copy()
# print(train_dataset)

'''6. Model Design and Training'''
'''
We will train the model using representative algorithms used for prediction models. <br><br>

    - Logistic Regression
    - Support Vector Machine (SVM)
    - k-Nearest Neighbor (kNN)
    - Random Forest
    - Naive Bayes
'''

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle

def train_and_test(model):
    model.fit(train_dataset, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_dataset, train_label) * 100, 2)
    print("Accuracy for " + str(model) + ": ", accuracy, "%")
    return prediction

# model.fit(X,y)
# model.predict(test_data)

# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
# kNN
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))
# Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
# Navie Bayes
nb_pred = train_and_test(GaussianNB())

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": rf_pred
})
submission.to_csv('submission_rf.csv', index=False)



'''Implemeting my own - need to work on'''

'''

user_data = {}

needed_features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "Relatives"]
for features in needed_features:
    user_input = input(f"{features}: ")
    user_data[features] = user_input 

user_values = list(user_data.values())

user_df = pd.DataFrame([user_values])
prediction = user_df(RandomForestClassifier(n_estimators=100))
print(prediction)
'''
