import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import time


class App:
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.Init_Streamlit_Page()
        self.params = dict()
        self.clf = None
        self.X, self.y = None, None
        
    def run(self):
        self.get_dataset()
        self.add_parameter_ui()
        self.generate()
   
    def Init_Streamlit_Page(self):
        st.title('Breast Cancer Wisconsin ')

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer',)
        )
        st.write(f"## {self.dataset_name} Dataset")

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Random Forest')
        )
        
    def get_dataset(self):
        data = None
        if self.dataset_name == 'Breast Cancer':
            data = pd.read_csv("data/data.csv", )
            diagnosis = {
                'M' : 0,
                'B' : 1
            }
            data['diagnosis'] = data['diagnosis'].map(diagnosis)
            data = data.drop('Unnamed: 32',axis=1)
            
            self.X = data.drop(columns=['diagnosis'])
            self.y = data['diagnosis']
            
            scaler = MinMaxScaler() # max value = 1 , min value = 0 
            self.X = scaler.fit_transform(self.X)
            
            encoder = LabelEncoder()
            self.y = encoder.fit_transform(self.y)
            
            st.write(data.head(10))
            st.write('Shape of dataset:', self.X.shape)
            st.write('number of classes:', len(np.unique(self.y)))
        
            
           
    

    def add_parameter_ui(self):
        if self.classifier_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 15.0)
            self.params['C'] = C
        elif self.classifier_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            self.params['K'] = K
        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            self.params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            self.params['n_estimators'] = n_estimators



    def get_classifier(self):
        if self.classifier_name == 'SVM':
            self.clf  = SVC(C=self.params['C'])
        elif self.classifier_name == 'KNN':
            self.clf  = KNeighborsClassifier(n_neighbors=self.params['K'])
        else:
            self.clf  = RandomForestClassifier(n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'], random_state=42)
            
         
            
    def plot_confusion_matrix(self,con_mat):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(con_mat, annot=True, cmap='viridis', fmt='d', cbar=False, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)



    def generate(self):
        
        self.get_classifier()
        
        #### CLASSIFICATION ####
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', acc)
        
        #### PLOT DATASET ####
        # Project the data onto the 2 primary principal components
        
     
        pca = PCA(2)
        X_projected = pca.fit_transform(self.X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2,
                c=self.y, alpha=0.8,
                cmap='viridis')
        st.pyplot(fig)
     
        
        
        ### plot confussion matrix
        con_mat = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(con_mat)
        
        
        
        

       