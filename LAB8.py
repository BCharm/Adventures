
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


df = pd.read_csv("https://raw.githubusercontent.com/BCharm/Adventures/master/employee.csv")

df.head()

df.dtypes

#Data Preprocessing

df.isnull().sum()

#"""**Corelation** **among** **variables**"""

df.describe()

df = df.drop(["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"], axis=1)


def corrcof(a, b):
    return np.array(np.cov(a, b) / (np.std(a) * np.std(b)))


def convert(d):
    switcher = {
        'Yes': 1,
        'No': 0,
        'Married': 2,
        'Single': 1,
        'Divorced': 0}
    return switcher.get(d)


df["Attrition"] = df["Attrition"].apply(convert)
df["OverTime"] = df["OverTime"].apply(convert)
df["MaritalStatus"] = df["MaritalStatus"].apply(convert)

df.head()

x_attributes = [df.Age, df.WorkLifeBalance, df.DailyRate, df.DistanceFromHome, df.Education, df.EnvironmentSatisfaction,
                df.PerformanceRating, df.RelationshipSatisfaction, df.StockOptionLevel, df.TotalWorkingYears,
                df.TrainingTimesLastYear, df.WorkLifeBalance, df.YearsAtCompany, df.YearsInCurrentRole,
                df.YearsSinceLastPromotion, df.YearsWithCurrManager, df.OverTime, df.NumCompaniesWorked,
                df.PercentSalaryHike, df.MonthlyRate, df.MonthlyIncome, df.MaritalStatus, df.HourlyRate,
                df.JobInvolvement, df.JobLevel, df.JobSatisfaction]


def corr_each(a, b):
    j = 0
    for i in b:
        print()
        print(f"{a.name} and {b[j].name} " + f"({corrcof(a, i)[0, 1]})")
        j += 1


corr_each(df.Attrition, x_attributes)

#"""**Logistic** **Regression**
#From the above corelation coefficient values we can say that Employee Attrition is likely to be related on [JobSatisfaction, JobLevel, JobInvolvement, MonthlyIncome, Overtime, YearswithCurrManager, YearsInCurrentRole, YearsAtCompany, TotalWorkingYears, StockOptionLevel, EnvironmentSatisfaction, Age]


df.Attrition.value_counts() / 1470 * 100

#From the above visualization we can say that our dataset is imbalanced

df.groupby("Attrition").mean()

pd.crosstab(df.Attrition, df.OverTime).plot.bar()
pd.crosstab(df.Attrition, df.EnvironmentSatisfaction).plot.bar()
pd.crosstab(df.Attrition, df.JobSatisfaction).plot.bar()
pd.crosstab(df.Attrition, df.JobInvolvement).plot.bar()

pd.crosstab(df.Attrition, df.StockOptionLevel).plot.bar()
pd.crosstab(df.Attrition, df.JobLevel).plot.bar()

x_attr = ['JobLevel', 'StockOptionLevel', 'JobSatisfaction', 'EnvironmentSatisfaction', 'OverTime', 'Age',
          'TotalWorkingYears', 'YearsAtCompany', 'MonthlyIncome']
df_y = df['Attrition']
df_x = df[x_attr]

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=0)
# X_train = scale(X_train[['MonthlyIncome','Age','TotalWorkingYears','YearsAtCompany']])
# X_test = scale(X_test[['MonthlyIncome','Age','TotalWorkingYears','YearsAtCompany']])
logr = LogisticRegression(max_iter=1000)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)
accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)
p = pd.DataFrame(y_pred)
p[0].value_counts()

logr.score(X_test, y_test) * 100

print(classification_report(y_test, y_pred))

acc1 = cross_val_score(logr, df_x, df_y, cv=10,
                       scoring='accuracy')  # Cross Validation to check if our model is prone to Overfitting.
acc1.mean() * 100

#"""After implementing the Logistic Regression, the score obtained is 86.25% for this model and this dataset. From the confusion matrix we can say that there are 634 correct predictions and 101 wrong predictions. Will implement the k-NN Algorithm on the same dataset and check with the results.
#**k**-**NN** **Model**
knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=0)
knn.fit(X_train, y_train)
y_predi = knn.predict(X_test)
confusion_matrix(y_test, y_predi)
knn.score(X_test, y_test)

plt.scatter(X_test['OverTime'], X_test['MonthlyIncome'], c=y_predi, cmap='coolwarm')

df_y.value_counts()

print(classification_report(y_test, y_predi))

acc2 = cross_val_score(knn, df_x, df_y, scoring='accuracy',
                       cv=10)  # Cross Validation to check if our model is prone to Overfitting.
acc2.mean() * 100

#"""We can see that the accuracy level of logistic regressor is high than a kNN classifier. This lazy learning algorithm is given a k-value of 10 which takes 10 nearest neighbors in each fold. The claasification report which give us the details such as F-Measure, Precision, Recall and both the models are provided with such report.
#**Import** **Models**



def logi_reg(x_var, y_var, t_size):
    X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=t_size, random_state=2)
    lr = LinearRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score = accuracy_score(y_pred, y_test)  # This gives us the accuracy score of these predictions
    c_report = classification_report(y_test, y_pred)  # This report displays the F1-Measure, Precision, Recall

    return lr


def knn_class(x_var, y_var, t_size, k):
    X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=t_size, random_state=2)
    knn = KNeighborsClassifier(n_neighbours=k, metric='euclidean').fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_pred, y_test)  # This gives us the accuracy score of these predictions
    c_report = classification_report(y_test, y_pred)  # This report displays the F1-Measure, Precision, Recall

    return knn


df["Attrition"].value_counts() / 1470 * 100

print(classification_report(y_test, y_predi))  # kNN Classifier

print(classification_report(y_test, y_pred))  # LogisticRegression