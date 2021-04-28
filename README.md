# Disaster Relief ML Pipeline

NLP classification pipeline for emergency response messages. 

This repository contains a project to build an ETL pipeline and machine learning pipeline to classify messages during the time following a disaster. The project takes in pre-labeled data, fits a model on that data and contains afront-end web app to display and test the results 

---

## Goal 

Present a disaster relief organization with filtered and important messages to in order to direct response efforts of individual teams to specific areas. Messages containing requests for water, blocked, roads or medical supplies can be directed to the appropriate teams in the organization for a more efficient response. 

--- 

## Project Components and Installation

**Installation**
Create a virtual environment and install the necessary library packages needed to run project.

On the command line inside the directory:

```
python3 -m venv <yourEnvName>
source <yourEnvName>/bin/activate
python3 -m pip install -r requirements.txt
```


There are three components as a part of this project.

1. ETL Pipeline: `process_data.py`

The script takes the file paths of the two `.csv` datasets (messages and categories) and the name of a database to create. The script then cleans the datasets, joins them together, and stores the clean data into a SQLite database in the database file path specified.

Example - Run in command line 

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

2. ML Pipeline: `train_classifier.py`

This script uses loads data from the SQLite database, splits the dataset into training and test sets and builds a text processing and machine learning pipeline. It then trains and tunes a model on the new data using GridSearchCV to find the best paramaters for that data. The trained mode is then exported as a pickle file to be used in the next step of classifing new messages in the front-end web app. 

Example - Run in command line from top level directory

```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```


3. Flask Web App

The final piece of this project includes a basic front-end web app in Flask that will highlight some visualizations about the data and provide a text field to enter new messages to be classifed with the model trained in the previous step.

Example - Run in command line 

```
python app/run.py
```

This will start a development process serving the web app at local address `0.0.0.0` on port `3001`.  Once confirmed that the web app is running, go to `0.0.0.0:30001` in a browser on your local machine to view the front-end html. 

Sample view of front-end

![Homepage of site](/viz/home.png)


![classification portion of site](/viz/classify.png)



## File Structure


```
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to
|- keywords.csv  # csv file to build Word Frequency chart

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```



## A note on Class Imbalance 

There are thousands of messages are not relavent to the target. Becuase of this we may have difficulties predicting them with standard algorithms. Conventional algorithms are often biased towards the majority class and training rewards simply choosing the dominant class without taking the data distribution into consideration. In the worst case, minority classes are treated as outliers and ignored. For these cases we need to artificially balance the dataset, for example by undersampling or oversampling each class.

In this case of imbalanced data, each class is trained individually and the majority class is needed to predict true neagtives. It is desirable to have a classifier that gives high prediction accuracy over the majority class, while maintaining reasonable accuracy for the minority classes. 

In training, each class was given a *stratified* K-fold in the cross validation step. This means that each fold of our cross validation is forced to match the original distrobution of the total training set of data. If need for water was 10% of the samples, each fold in our cross validation would also include 10% of the data labeled as such. 



## Licensing, Authors, Acknowledgements<a name="licensing_authors_acknowledgements"></a>

Data for this project was provided by [Figure Eight](https://www.figure-eight.com/) and support/framework of the project was provided as a part of the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) program. 

License MIT.