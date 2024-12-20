import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
from scipy.stats import zscore
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import Image

# KNN function 
def KNN(dataframe , features,target,k , metric, predict_val = False , features_vals = [0]):
    X = dataframe[features]
    Y = dataframe[target]

    # 1.2 Splitting the data into train (80%) & test (20%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

    # 1.3 Standrization the dataset for accurate process
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 1.4 KNN - Traing the model with the data
    # when p = 1 -> Manhatten , p=2 -> eculdian 
    knn = KNeighborsClassifier(n_neighbors=k,metric=metric)  
    knn.fit(X_train, Y_train) 

  
    # 1.6 Does the user want to predict a new value ?
    if predict_val:
        print("-"*50)
        try:
            # Predict new value 
            feat_measures = features_vals
            predicted_value = knn.predict(scaler.transform([feat_measures]))
            # Simple message with the prediction
            for feat in feat_measures:
                print(f"- If {dataframe.columns[feat_measures.index(feat)]} = {feat}")

            print(f"Then the category of {target} is probably : {predicted_value}")
        except Exception as e:
            print(f"Error details : {e}")

        print("-"*50)
    # 1.7 Evalutate the model

    y_train_pred = knn.predict(X_train)
    train_accuracy = accuracy_score(Y_train, y_train_pred)

    y_test_pred = knn.predict(X_test)
    test_accuracy = accuracy_score(Y_test, y_test_pred)

    # ---------------- Training info --------------
    print(f"Model data : K = {k} , distance metric = {knn.get_params()['metric']}")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"\nClassification Report:\n {classification_report(Y_train, y_train_pred)}")
    print(f"\nConfusion Matrix:\n")
    plt.figure(figsize=(15,5))
    sns.heatmap(confusion_matrix(Y_train, y_train_pred) ,cmap="Blues" ,annot=True , fmt='d')
    plt.title("Training Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("-"*60)
    
    # ---------------- Testing info --------------
    print(f"Model data : K = {k} , distance metric = {knn.get_params()['metric']}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print(f"\nClassification Report:\n {classification_report(Y_test, y_test_pred)}")
    print(f"\nConfusion Matrix:\n")
    plt.figure(figsize=(15,5))
    sns.heatmap(confusion_matrix(Y_test, y_test_pred) , annot=True , fmt='d')
    plt.title("Testing Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("-"*60)

def Decision_Tree(dataframe,features,target,max_depth,criterion,min_samp_slt,Enable_Search_Mode=False):
    
    # 2.1 Define the used features and the target features
    X = dataframe[features]
    y = dataframe[target]

    # 2.2 Encode target variable (if categorical)
    # 0 -> Good , 1 -> Hazardous , 2 -> Moderate , 3 -> Poor
    encoder = LabelEncoder()
    y = encoder.fit_transform(y) 

    # 2.3 Split the dataset to train(80%) , test(20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20) 
    # 2.4 Train The Decision Tree 
    clf = tree.DecisionTreeClassifier(
        criterion=criterion, 
        max_depth=max_depth,
        min_samples_split= min_samp_slt       
              )
    clf.fit(X_train, y_train)
    print("\nDecision Tree Model Trained Successfully!") 

    # Predict on train set
    y_train_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_train_pred)


    # Predict on test set
    y_test_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)


    # If Enable_Search_Mode is enabled , Use Grid Search To find the best params
    if Enable_Search_Mode:

        parms = {
            "max_depth" : [3,5,10,13] ,
            "min_samples_split" : [3,5,10,13] ,
            "criterion" : ["gini","entropy"] 
        }
        clf = tree.DecisionTreeClassifier(random_state=20)
        grid_search = GridSearchCV(
            estimator=clf ,
            param_grid=parms,
            cv = 5,
            scoring="accuracy",
            verbose=1
            )
        grid_search.fit(X_train,y_train)
        return f"Best Parmaters : {grid_search.best_params_ }\n Best Accuracy :{grid_search.best_score_}"
    print(f"\nAccuracy: {accuracy:.2f}")

    # Training Confusion Matrix
    print("\Training Confusion Matrix:")
    cm = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Training Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("\nTesting Classification Report:")
    print(classification_report(y_train, y_train_pred, target_names=encoder.classes_))

    print("-"*50)
    # Testing Confusion Matrix
    print("\nTesting Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Testing Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("\nTesting Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=encoder.classes_))
    

    # The Tree 
    #    ------ Wrok on juypter but not visual 
    # data = tree.export_graphviz(clf, out_file=None,feature_names= X_train.columns, filled = True)
    # graph = pydotplus.graph_from_dot_data(data)
    # Image(graph.create_png())

def Naive_Bayes(dataframe):
    
    # 3.1 Prepare the data
    # Splitting the dataset 
    X =  dataframe.iloc[:,:-1]  
    y = dataframe.iloc[:,-1]    

    # Encode categorical target 
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # 3.2 Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_test_pred = model.predict(X_test)

    # 3.4 Evaluate the model
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Testing accuracy of the Naive Bayes model: {accuracy}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=encoder.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nTesting Confusion Matrix: {cm}")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=encoder.classes_,yticklabels=encoder.classes_)
    plt.title("Testing Confusion Matrix ")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


    # Predicting new value
    new_data = [[25, 75, 2, 12, 29, 9, 1, 6, 600]] 
    predicted_quality_class = model.predict(new_data)

    # Decode the predicted label
    decoded_class = encoder.inverse_transform(predicted_quality_class)
    print(f"Predicted Air Quality Category: {decoded_class[0]}")

def SVM(dataframe , kernels):
    # 4.1 Prepare the dataset
    X = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    
    # Scalling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4.2 SVM with various kernels
    kernels = kernels # kernel functions
    for kernel in kernels:
        model = SVC(kernel=kernel, random_state=20)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Kernel: {kernel}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=encoder.classes_))
        print("-" * 50)

    # 4.3 Hyperparameter tuning using grid s earch

    # parameters
    params= {
        'C'     : [0.1, 1, 10],           # Regularization params
        'gamma' : [1, 0.1, 0.01],         # Kernel coefficient
        'kernel': ['linear', 'poly']      # Kernel function
    }


    # Splitting the train dataset into smaller samples to speed up the process
    x_train_sample, _ ,y_train_sample,_ =  train_test_split(X_train, y_train , test_size=0.7, random_state=20)
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        SVC(random_state=20), 
        params,
        cv=3, 
        scoring='accuracy', 
        verbose=3
        )
    grid_search.fit(x_train_sample, y_train_sample)

    # Best params
    print("Best Hyperparameters:", grid_search.best_params_)
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

    # 4.4 Evaluate the best model on the testing set
    # Train the best model for the whole set
    best_model = SVC(**grid_search.best_params_,random_state=20)
    best_model.fit(X_test , y_test)    
    y_pred_test = best_model.predict(X_test)

    print(f"Testing Accuracy:{accuracy_score(y_test, y_pred_test)}")
    print(f"Testing Confusion Matrix: {confusion_matrix(y_test, y_pred_test)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=encoder.classes_))

    #Visualization of Results
    disp = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=encoder.classes_, cmap='Blues')
    plt.title("Testing Confusion Matrix")
    plt.show()


# ---------- Start Here ---------------
# Reading the dataset
df = pd.read_csv("pollution_dataset.csv")

# Getting some info before preprocessing 
"""print(df.shape)
print(df.info())  # Non-Null for each feature(col)
print(df.describe()) # data decribtion
print(df.iloc[:,:-1].min())""" # minimums of each numerical feature




# 1.0 Preprocessing

# 1.1 Cleaning missing values by drpoing the whole row
df = df.dropna()

# 1.2 Replacing negative values in SO2,PM10 with the mean of each one of them
df[df[["SO2" , "PM10"]] < 0 ] = None
df.SO2 = df.SO2.fillna(df.SO2.mean())
df.PM10 = df.PM10.fillna(df.PM10.mean())


# 1.3 Remove outliers
df_without_outliers = df[
    (np.abs(zscore(df.iloc[:,:-4]))<3 ).all(axis=1)
    ]

# Dataset info after preprocessing

print(df.shape)
print(df.info()) # Non-Null for each feature(col)
print(df.describe()) # data decribtion
print(df.iloc[:,:-1].min())



# 2.0 Exploratory Data Analysis (EDA)

# 2.1 Distribution of the dataset (Ranges) = Histogram
fig, ax = plt.subplots(3, 3, figsize=(10, 10)) 
ax = ax.flatten()
i = 0
for feature in df_without_outliers.iloc[:,:-1].columns:
    sns.histplot(data = df_without_outliers, x=feature,kde = True, color='red' , ax=ax[i])
    i+=1
plt.tight_layout()
plt.show()



# 2.2 Boxplot
fig, ax = plt.subplots(3, 3, figsize=(10, 10)) 
ax = ax.flatten() 
i = 0 
for feature in df_without_outliers.iloc[:,:-1].columns: 
    sns.boxplot(data=df_without_outliers, x='Air Quality', y=feature, ax=ax[i]) 
    i += 1  
plt.tight_layout() 
plt.show()

# 2.3 Scatterplot between for each featrue and Air Quality
fig, ax = plt.subplots(3, 3, figsize=(20, 20)) 
ax = ax.flatten() 
i = 0 
for feature in df_without_outliers.iloc[:,:-1].columns: 
    scatter = sns.scatterplot(data=df, x='Air Quality', y=feature, ax=ax[i],hue='Air Quality')
    scatter.legend(loc='upper right' , fontsize='small',title='Air Quality' , title_fontsize='small') 
    i += 1 
plt.tight_layout() 
plt.show()

# 2.4 Correlation matrix for numerical columns
corr_mat = df_without_outliers.iloc[:,:-4].corr()
plt.figure(figsize=(8,4))
sns.heatmap(corr_mat , annot=True , cmap="coolwarm",fmt='.2f',linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# 2.5 Target variable statistics
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
trgt = df_without_outliers.iloc[:,-1].value_counts().reset_index()
trgt.columns = ['Air Quality' , 'count']
ax[0].pie(trgt['count'] , labels =trgt['Air Quality'] , autopct ='%.2f%%')
ax[0].set_aspect('equal')

sns.countplot(data = df_without_outliers , x = df_without_outliers['Air Quality'] , ax=ax[1])
plt.tight_layout() 
plt.show()
# --------------------- KNN ----------------------------------------

features = df_without_outliers.columns[:-1]
target = ["Air Quality"]


vals =[30,40,2.2,16.9,17.5,9,1,10,300]
KNN(df_without_outliers,features,target,15,"euclidean",True,vals)
# Best value for the k was 15

# --------------------- Decision Tree --------------------------------

# Trying the model for one value 
#print(Decision_Tree(df_without_outliers,features,target,3,"entropy",3))

# Testing many for hyperparmameters selection(Using Grid Search)
'''
Best Parmaters : {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 13}
Best Accuracy :0.9332801457270884
'''
print(Decision_Tree(df_without_outliers,features,target,13,"entropy",12,True))


# --------------------- Naive Bayes ----------------------------------
Naive_Bayes(df_without_outliers)

# --------------------- SVM ------------------------------------------
SVM(df_without_outliers,['linear', 'poly', 'rbf'])
