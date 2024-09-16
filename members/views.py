from pyexpat.errors import messages
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from requests import request
from django.contrib.auth import authenticate
from django.contrib.auth import authenticate,login,logout,update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm,SetPasswordForm
from django.contrib.auth import login
from django.views.decorators.csrf import csrf_protect
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import random

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#import the sklearn libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


account={}
otp_number = str(random.randint(100000, 999999))
detection ={}



from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages

def main(request):
    return render(request,"main.html")


def index(request):
    # If the login was unsuccessful or it's not a POST request, render the login page
    return render(request, 'login.html')


@csrf_protect   
def welcome(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
            
        user=authenticate(username=username,password=password)
        print(username,password)
        if user is not None:
           login(request,user)
           messages.success(request,"Welcome,You are Successfully Logged in!!!")
           return render(request,"dashboard.html")
        else:
            messages.error(request,"Username or Password is incorrect.Please try again..")
            return render(request,"error.html")
    
    return render(request,"index.html")

# Creating a Account
def register(request):
            
 return render(request,"signup.html")
        


def detect(request):
    #read and load the data
    data=pd.read_csv("dataset/indian_liver_patient.csv")
    print (f"Total number of samples: {data.shape[0]}. Total number of features in each sample: {data.shape[1]} .")

    #get the first five data 
    data.head()

    #to get the tail info of the dataset
    data.tail()
    # help to know if there are missing data or not
    data.info()
    data.describe()
    data.shape
    #commence data preprocessing

    #check to see if there are duplicates
    ''' Duplicates are removed as it's most likely these entries has been inputed twice.'''
    data_duplicate = data[data.duplicated(keep = False)] 
    # keep = False gives you all rows with duplicate entries
    data_duplicate
    data = data[~data.duplicated(subset = None, keep = 'first')]
    # Here, keep = 'first' ensures that only the first row is taken into the final dataset.
    # The '~' sign tells pandas to keep all values except the 13 duplicate values
    data.shape
    #checking if there are any NULL values in our Dataset
    data.isnull().values.any()
    #removing null values
    # display number of null values by column# display number of null values by column
    print(data.isnull().sum()) 
    # We can see that the column 'Albumin_and_Globulin_Ratio' has 4 missing values
    # One way to deal with them can be to just directly remove these 4 values
    print ("length before removing NaN values:%d"%len(data))
    data_2 = data[pd.notnull(data['Albumin_and_Globulin_Ratio'])]
    print ("length after removing NaN values:%d"%len(data_2))

    new_data=data.dropna(axis = 0, how ='any')

    new_data.isnull().values.any()
    data.info()
    '''
    The Albumin-Globulin Ratio feature has four missing values, as seen above. Here, we are dropping those particular rows which have missing data. We could, in fact, fill those place with values of our own, using options like:

    A constant value that has meaning within the domain, such as 0, distinct from all other values.
    A value from another randomly selected record, or the immediately next or previous record.
    A mean, median or mode value for the column.
    A value estimated by another predictive model.
    But here, since a very small fraction of values are missing, we choose to drop those rows.
    '''
    #Transform our data
    le = preprocessing.LabelEncoder()
    le.fit(['Male','Female'])
    data.loc[:,'Gender'] = le.transform(data['Gender'])

    #Remove rows with missing values
    data = data.dropna(how = 'any', axis = 0)

    #Also transform Selector variable into usual conventions followed
    data['Dataset'] = data['Dataset'].map({2:0, 1:1})
    #Overview of data
    data.head()

    #features characteristics to determine if feature scaling is necessary
    data.describe()
    #split the data into test and train samples
    X_train, X_test, y_train, y_test = train_test_split(data, data['Dataset'], random_state = 0)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Exploratory Data Analysis

    #Determining the healthy-affected split
    print("Positive records:", data['Dataset'].value_counts().iloc[0])
    print("Negative records:", data['Dataset'].value_counts().iloc[1])

    #The above output confirms that we have 414 positive and 165 negative records. This indicates that this is a highly unbalanced dataset.

    #Determine statistics based on age
    plt.figure(figsize=(12, 10))
    plt.hist(data[data['Dataset'] == 1]['Age'], bins = 16, align = 'mid', rwidth = 0.5, color = 'black', alpha = 0.8)
    plt.xlabel('Age')
    plt.ylabel('Number of Patients')
    plt.title('Frequency-Age Distribution')
    plt.grid(True)
    plt.savefig('fig1')
    plt.show()
    #Looking at the age vs. frequency graph, we can observe that middle-aged people are the worst affected. Even elderly people are also suffering from liver ailments,
    #as seen by the bar sizes at ages 60-80.

    #correlation-matrix
    plt.subplots(figsize=(12, 10))
    plt.title('Pearson Correlation of Features')
    # Draw the heatmap using seaborn
    sns.heatmap(data.corr(),linewidths=0.25, vmax=1.0, square=True,annot=True)
    plt.savefig('fig2')
    plt.show()
    '''
    The correlation matrix gives us the relationship between two features. As seen above, the following pairs of features seem to be very closely related as indicated by their high correlation coefficients:

    1.Total Bilirubin and Direct Bilirubin(0.87)
    2.Sgpt Alamine Aminotransferase and Sgot Aspartate Aminotransferase(0.79)
    3.Albumin and Total Proteins(0.78)
    4.Albumin and Albumin-Globulin Ratio(0.69)
    '''

    '''
    #Using Classification Algorithms
    Let us now evaluate the performance of various classifiers on this dataset. 
    For the sake of understanding as to how feature scaling affects classifier performance, 
    we will train models using both scaled and unscaled data. 
    Since we are interested in capturing records of people who have been tested positive,
    we will base our classifier evaluation metric on precision and recall instead of accuracy.
    We could also use F1 score, since it takes into account both precision and recall.
    '''

    #Logistic Regression: Using normal data
    logreg = LogisticRegression(C = 0.1).fit(X_train, y_train)
    print("Logistic Regression Classifier on unscaled test data:")
    print("Accuracy:", logreg.score(X_test, y_test))
    print("Precision:", precision_score(y_test, logreg.predict(X_test)))
    print("Recall:", recall_score(y_test, logreg.predict(X_test)))
    print("F-1 score:", f1_score(y_test, logreg.predict(X_test)))


    #Using feature-scaled data
    logreg_scaled = LogisticRegression(C = 0.1).fit(X_train_scaled, y_train)
    print("Logistic Regression Classifier on scaled test data:")
    print("Accuracy:", logreg_scaled.score(X_test_scaled, y_test))
    print("Precision:", precision_score(y_test, logreg_scaled.predict(X_test_scaled)))
    print("Recall:", recall_score(y_test, logreg_scaled.predict(X_test_scaled)))
    print("F-1 score:", f1_score(y_test, logreg_scaled.predict(X_test_scaled)))

    '''
    Well! The performance has definitely improved by feature scaling, though not drastically, 
    as there was already very little scope of improvement. 
    Let us look at other classifiers and analyse how they react to scaling.
    '''

    #SVM Classifier with RBF kernel: Using normal data
    svc_clf = SVC(C = 0.1, kernel = 'rbf').fit(X_train, y_train)
    print("SVM Classifier on unscaled test data:")
    print("Accuracy:", svc_clf.score(X_test, y_test))
    print("Precision:", precision_score(y_test, svc_clf.predict(X_test)))
    print("Recall:", recall_score(y_test, svc_clf.predict(X_test)))
    print("F-1 score:", f1_score(y_test, svc_clf.predict(X_test)))

    #Using scaled data
    svc_clf_scaled = SVC(C = 0.1, kernel = 'rbf').fit(X_train_scaled, y_train)
    print("SVM Classifier on scaled test data:")
    print("Accuracy:", svc_clf_scaled.score(X_test_scaled, y_test))
    print("Precision:", precision_score(y_test, svc_clf_scaled.predict(X_test_scaled)))
    print("Recall:", recall_score(y_test, svc_clf_scaled.predict(X_test_scaled)))
    print("F-1 score:", f1_score(y_test, svc_clf_scaled.predict(X_test_scaled)))

    #Random Forest Classifier: using normal data
    rfc = RandomForestClassifier(n_estimators = 20)
    rfc.fit(X_train, y_train)
    print("SVM Classifier on unscaled test data:")
    print("Accuracy:", rfc.score(X_test, y_test))
    print("Precision:", precision_score(y_test, rfc.predict(X_test)))
    print("Recall:", recall_score(y_test, rfc.predict(X_test)))
    print("F-1 score:", f1_score(y_test, rfc.predict(X_test)))


    #using scaled data
    rfc_scaled = RandomForestClassifier(n_estimators = 20)
    rfc_scaled.fit(X_train_scaled, y_train)
    print("Random Forest Classifier on scaled test data:")
    print("Accuracy:", rfc_scaled.score(X_test_scaled, y_test))
    print("Precision:", precision_score(y_test, rfc_scaled.predict(X_test_scaled)))
    print("Recall:", recall_score(y_test, rfc_scaled.predict(X_test_scaled)))
    print("F-1 score:", f1_score(y_test, rfc_scaled.predict(X_test_scaled)))


    print('################### --------------- Code Was Succeffully Executed!')

    return render(request,"dashboard.html")

def send_otp(request):
    if request.method == 'POST':

        account['user'] = request.POST.get("username")
        account['email']  = request.POST.get("email")
        account['mobile'] = request.POST.get("mobile")
        account['password'] = request.POST.get("password")
        account['repassword'] = request.POST.get("confirmPassword")
        account['method'] = request.POST.get('Verification')

        credential = {'name':account['user'],'email':account['email'],'mobile':account['mobile'],'password':account['password'],'repassword':account['repassword'],'method':account['method']}
        # Open the file in write mode
        with open('credential.txt', 'w') as file:
        # Write the content to the file
            file.write(str(credential))
        
        if account['method'] == 'email':
            # Your email credentials
            fromaddr = "ramdevops2005@gmail.com"
            toaddr = request.POST.get("email")
            smtp_password = "rcau rkir ffiw megr"

            # Create a MIMEMultipart object
            msg = MIMEMultipart()

            # Set the sender and recipient email addresses
            msg['From'] = fromaddr
            msg['To'] = toaddr
            
            # Set the subject
            msg['Subject'] = "Liver Detection Otp Verification"

            # Set the email body
            body = f"Your OTP is: {otp_number}"
            msg.attach(MIMEText(body, 'plain'))

            try:
                # Connect to the SMTP server
                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    # Start TLS for security
                    server.starttls()

                    # Log in to the email account
                    server.login(fromaddr, smtp_password)

                    # Send the email
                    server.sendmail(fromaddr, toaddr, msg.as_string())

                # Email sent successfully, render a template
                return render(request, 'verification_otp.html')

            except Exception as e:
                # An error occurred while sending email, redirect with an error message
                messages.error(request, f"Error sending OTP email: {e}")
                return render(request,'signup.html')  # You need to replace 'verify_it' with the appropriate URL name
        else:
            # Invalid method, redirect with an error message
            messages.error(request, "Invalid verification method")
            return render(request,'signup.html')  # You need to replace 'verify_it' with the appropriate URL name

    # If the request method is not POST, redirect with an error message
    messages.error(request, "Invalid request method")
    return render(request,'signup.html') # You need to replace 'verify_it' with the appropriate URL name


def verify_it(request):
    
    if request.method=="POST":


       

        verifi_otp1 = request.POST.get("otp1")
        verifi_otp2 = request.POST.get("otp2")
        verifi_otp3 = request.POST.get("otp3")
        verifi_otp4 = request.POST.get("otp4")
        verifi_otp5 = request.POST.get("otp5")
        verifi_otp6 = request.POST.get("otp6")

        six_digits=f"{verifi_otp1}{verifi_otp2}{verifi_otp3}{verifi_otp4}{verifi_otp5}{verifi_otp6}"
        if six_digits==otp_number:

         my_user=User.objects.create_user(account['user'],account['email'],account['password'])
         my_user.save() 
         messages.success(request,"Your account has been Created Successfully!!!")
         redirect(index)


        # else:
        #     messages.success(request,"Registration Failed!!")
        #     return render(request, 'success.html',six_digits)
        
    return render(request,"login.html")  

