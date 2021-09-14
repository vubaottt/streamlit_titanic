from pandas.core.algorithms import mode
import streamlit as sl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics

# Part 1: Build project
data = pd.read_csv("train.csv")

# Data preprocessing
data['Sex'] = data['Sex'].map(lambda x: 0 if x == 'male' else 1)
data = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Survived']]
data = data.dropna()

X = data.drop(['Survived'], axis=1)
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=41)

# Scale data
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.fit_transform(X_test)

# Build model
model = LogisticRegression()
model.fit(train_features, y_train)

# Evaluation
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)
y_predict = model.predict(test_features)
confusion_matrix  = metrics.confusion_matrix(y_test, y_predict)
FN  = confusion_matrix[1][0]
TN  = confusion_matrix[0][0]
FP  = confusion_matrix[0][1]
TP  = confusion_matrix[1][1]

# Calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)

# Calculate AUC
auc = metrics.roc_auc_score(y_test, y_predict)

# Part 2: Show project's result with Streamlit
sl.title("Data science")
sl.write("## Titanic Survival Prediction Project")

menu = ["Overview", "Build Project", "New Prediction"]
choice = sl.sidebar.selectbox('Menu', menu)
if choice == 'Overview':
    sl.subheader("Overview")
    sl.write("""
    #### The data has been split into two groups:
    - Training set(train.csv): The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the "ground truth") for each passenger. Your model will be based on "features" like passengers' gender and class. You can also use feature engineering to create new features.
    - Test set(test.csv): The test set should be used to see how well your model performs on unseen data, For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcome. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
    - Gender submission(gender_submission.csv): A set of predictions that assume all and only female passengers survived, as an example of what a submission file should look like.
    """)
elif choice == "Build Project":
    sl.subheader("Build Project")
    sl.write("### Data Preprocessing")
    sl.write("#### Show data:")
    sl.table(data.head(5))
    sl.write("### Build model and evaluation:")
    sl.write("Train set score: {}".format(round(train_score, 2)))
    sl.write("Test set score: {}".format(round(test_score, 2)))
    sl.write("Confusion matrix:")
    sl.table(confusion_matrix)
    sl.write(metrics.classification_report(y_test, y_predict))
    sl.write("#### AUC: %.3f" % auc)

    sl.write("#### Visulization")
    fig, ax = plt.subplots()
    ax.bar(["False Negative", "True Negative", "False Positive", "True Positive"], [FN, TN, FP, TP])
    sl.pyplot(fig)

    # Roc curve
    sl.write("Roc curve")
    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.plot(fpr, tpr, marker='.')
    ax1.set_title("Roc Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    sl.pyplot(fig1)
elif choice == 'New Prediction':
    sl.subheader("New Prediction")
    sl.write("##### Input/Select data")
    name = sl.text_input("Name of passenger")
    sex = sl.selectbox("Sex", options=['Male', 'Female'])
    age = sl.slider("Age", 1, 100, 1)
    Pclass = np.sort(data['Pclass'].unique())
    pclass = sl.selectbox("Pclass", options=Pclass)
    max_sibsp = max(data['SibSp'])
    sibsp = sl.slider("Siblings", 0, max_sibsp, 1)
    max_parch = max(data['Parch'])
    parch = sl.slider("Parch", 0, max_parch, 1)
    max_fare = round(max(data['Fare']) + 10, 2)
    fare = sl.slider("Fare", 0.0, max_fare, 0.1)

    # make new prediction
    sex = 0 if sex == 'Male' else 1
    new_data = scaler.transform([[sex, age, pclass, sibsp, parch, fare]])
    prediction = model.predict(new_data)
    predict_prob = model.predict_proba(new_data)

    if(prediction[0] == 1):
        sl.subheader("Passenger {} would have survived with a probability of {}%".format(name, round(predict_prob[0][1]*100, 2)))
    else:
        sl.subheader("Passenger {} would not have survived with a probability of {}%".format(name, round(predict_prob[0][1]*100, 2)))