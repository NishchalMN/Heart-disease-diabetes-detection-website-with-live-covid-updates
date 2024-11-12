# Heart Disease and Diabetes Detection with COVID-19 Updates

## Overview
A web application that assesses the risk of heart disease and diabetes based on user inputs and provides real-time COVID-19 updates via email. The application is built using Flask in Python.
Here is the link to the project demo -> https://drive.google.com/file/d/19Kl-LBSLNTBkrmfFlhxlH-t3TzZvkwZb/view?usp=drive_link

## Features
- **Health Assessment**: Detects heart disease and diabetes risk based on input data.
- **COVID-19 Subscription**: Allows users to subscribe or unsubscribe for COVID-19 updates delivered by email.
- **Multi-Page Navigation**: Includes dedicated pages with information on heart disease, diabetes, and doctor contacts.
- **Predictive Fetching**: Dynamically loads content related to health topics as users scroll.

## Testing Summary
- **Unit Testing**: Validates core functionalities and handles various user scenarios.
- **Load Testing**: Assesses performance under different loads, from single-user access to multiple concurrent users.
- **System Testing**: Simulates end-to-end interactions to ensure complete functionality.

## Technologies Used
- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, JavaScript
- **Testing Tools**: unittest, K6, Selenium

## General Information

The website has been built using Angularjs as the front-end framework and flask as the back-end framework along with HTML, CSS, JS.

For the detection of diseases, existing datasets have been collected and seven classifiers have been trained and the confidence is weighted. 
The classifiers used are ANN, Random-Forest, Decision Trees, KNN, Logistic Regression, SVM and Naive Bayes Classifier.

The user has to enter the required data and the result is returned. This website has a special feature which provides the real time COVID-19 case updates in India statewise to the provided email.
