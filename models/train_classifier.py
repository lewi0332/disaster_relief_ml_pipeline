import sys
import pickle as pkl

# import libraries
import nltk
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet'])
st = set(stopwords.words('english'))

import re
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC


def load_data(database_filepath):
    """
    Loads data from database

    Input - database_filepath: filepath to sqlite database
    Output - X, y: Pandas DataFrames with Data and labels for training.
    """
    # get table name from filepath
    table = database_filepath.split('/')[-1].split('.')[0]

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table, engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, y


def tokenize(text):
    """
    Tokenize with NLTK and removes URLs

    Input - text - Single string object with english message
    Output - list of lowercase, lemmatized word tokens
    """
    # Regex string to match URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Remove Punctiation and other characters
    text = re.sub('[^a-zA-Z0-9]', ' ', text)

    tokens = word_tokenize(text)  # Tokenize block of text

    lemmatizer = WordNetLemmatizer()  # Initialize Lemmatizer

    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    #remove stopwords
    clean_tokens = [x for x in clean_tokens if x not in list(st)]
    clean_tokens = [x for x in clean_tokens if x not in ['said', 'wa', 'ha', 'u', '000']]
    return clean_tokens


def build_model():
    """
    Builds an Sklearn Pipeline with a countVectorizer, TF-IDF Transformer and Linear Support Vector Classifier object. 

    Input - None

    Output - Grid Search Cross Validation object with 3 stratified folds to balance target class to distrobution. 
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer(use_idf=False)),
                         ('clf', MultiOutputClassifier(LinearSVC()))])
    parameters = {
        'vect__max_df': (.45, .5, .65),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__C': [.45, .5, .65]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=4)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Function to gather basic results for printing to standard out. 

    Input - Model : trained model object
            X_test : Unseen Input features to evaluate model
            y_test : Unseen labels to evaluate model
    
    Output - Pandas dataframe with 'precision', 'recall', 
             'f1-score', 'support', and 'accuracy' for each class
    """
    y_pred = model.predict(X_test)
    results_df = pd.DataFrame(columns=['precision', 'recall', 'f1-score', 'support', 'accuracy'])
    for index, column in enumerate(y_test.columns):
        cr_dict = classification_report(y_test[column],
                                        y_pred[:, index],
                                        output_dict=True,
                                        labels=np.unique(y_pred[:, index]))
        cr_dict['weighted avg']['accuracy'] = accuracy_score(y_test[column], y_pred[:, index])
        results_df = results_df.append(pd.DataFrame(index=[column], data=cr_dict['weighted avg']))
    return results_df


def save_model(model, model_filepath):
    """
    Saves model as pickle object.

    Input - model : model object
            model_filepath : filepath destination for output
    Output - None, file stored 
    """
    pkl.dump(model, open(model_filepath, 'wb'))
    pass


def build_word_freq(X, y):
    """
    Builds a csv table with top 20 most frequent words for each target. 
    To be used in visualization to demonstrate NLP functionality

    Input - X message feature associated with the model 
            y label features associated with the model
    
    Output - keywords.csv stored in 'data' directory 
    """
    dff = pd.concat([X, y], axis=1)
    corpus = []
    corpus_names = []
    for _ in dff.columns[4:]:
        corpus.append(dff.loc[dff[_] == 1]['message'].str.cat(sep=' '))
        corpus_names.append(_)

    vectorizer = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer())])
    vectors = vectorizer.fit_transform(corpus)

    names = vectorizer.named_steps['vect'].get_feature_names()
    data = vectors.todense().tolist()
    # Create a dataframe with the results
    keywords_ = pd.DataFrame(data, columns=names)

    key_dict = {}
    N = 20
    for i, v in enumerate(keywords_.iterrows()):
        key_dict[corpus_names[i]] = v[1].sort_values(ascending=False)[:N].to_dict()

    pd.DataFrame(key_dict).to_csv('./data/keywords.csv')
    return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        results_df = evaluate_model(model, X_test, y_test)
        print("Model results: \n", results_df.mean())

        print("Model Best Parameters: \n", model.best_params_)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        print('Building a Word Frequency table for landing page.')
        build_word_freq(X, y)

        print('Saved word frequency table.')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()