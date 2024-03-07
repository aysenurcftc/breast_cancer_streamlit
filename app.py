import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import os
os.environ['OMP_NUM_THREADS']='1'


class App:
    
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.data = None
        self.Init_Streamlit_Page()
        self.params = dict()
        self.clf = None
        self.X, self.y = None, None
        
        
    def run(self):
        self.get_dataset()
        self.generate()
        self.plot_mb()
        self.plot_corr()
        
   
   
   
    def Init_Streamlit_Page(self):
        st.title('Breast Cancer Wisconsin ')

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer',)
        )
        st.write(f"## {self.dataset_name} Dataset")

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Random Forest', 'Gaussian Naive Bayes')
        )
      
        
        
    def get_dataset(self):
        
        data = pd.read_csv("data/data.csv")
    
        data = data.drop('Unnamed: 32',axis=1)
        data = data.drop('id',axis=1)
        
        self.data = data
        self.X = data.drop(columns=['diagnosis'], axis=1)
        self.y = data['diagnosis']
    
        
        st.write(data.head(10))
        st.write('Shape of dataset:', self.X.shape)
        st.write('number of classes:', len(np.unique(self.y)))
        
        

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            model = SVC() 
            svm_params = {'C': [0.1, 1, 10, 50, 100, 1000],  
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                        'kernel': ['rbf']} 
            self.clf = GridSearchCV(model,
                           svm_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1)
            
            
        elif self.classifier_name == 'KNN':
            model  = KNeighborsClassifier()
            knn_params = {"n_neighbors": range(2, 50)}
            self.clf = GridSearchCV(model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1)
            
            
        elif self.classifier_name == "Gaussian Naive Bayes":
            model = GaussianNB()
            naive_bayes_params = {
                'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
            }
            self.clf = GridSearchCV(model,
                           naive_bayes_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1)
            
            
        else:
            model  = RandomForestClassifier()
            rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [20, 300]}
            self.clf = GridSearchCV(model,
                           rf_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1)
            
        return self.clf
            
         
            
    def plot_confusion_matrix(self,con_mat):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(con_mat, annot=True, cmap='PiYG', fmt='d', cbar=False, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        
        
    def plot_corr(self):
        df = pd.DataFrame(self.X)
        fig, ax = plt.subplots(figsize=(20, 18))
        heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='PiYG')
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
        ax.set_xlabel('Features')
        ax.set_ylabel('Features')
        st.pyplot(fig)
    
        
        
    def plot_mb(self):
        data = self.data
        malignant_data = data[data['diagnosis'] == 'M']
        benign_data = data[data['diagnosis'] == 'B']
        fig, ax = plt.subplots(figsize=(10, 6))
       
        ax.scatter(malignant_data['radius_mean'], malignant_data['texture_mean'], color='pink', label='Malignant')
        ax.scatter(benign_data['radius_mean'], benign_data['texture_mean'], color='green', label='Benign')
        
        ax.set_xlabel('Radius Mean')
        ax.set_ylabel('Texture Mean')
        ax.set_title('Scatter Plot of Radius Mean vs Texture Mean')
        fig.legend()
        st.pyplot(fig)
        
   
       
        

    def generate(self):
        
        self.get_classifier()
        
          
        scaler = MinMaxScaler() # max value = 1 , min value = 0 
        self.X = scaler.fit_transform(self.X)
        
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.y)
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        
        self.clf.fit(X_train, y_train)
        
        #best_params = self.clf.best_params_
        #st.write("En iyi parametreler:", best_params)
    
        y_pred = self.clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

         
        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', acc)
        st.write(f'f1 score = ', f1)
        st.write(f"precision = ", precision)
        st.write(f'recall = ', recall)
        
        
        ### plot confussion matrix
        con_mat = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(con_mat)
        
        
        
        
        
        
        
        
        
            
        
        
        

       