# Sentiment-Analysis---Democrats-vs-Republican-Tweets

Sentiment analysis is one of the most interesting NLP topics in which I always wanted to do a project.

As the US election was the main discussion point in the last few months, people are very expressive about their support to Democratic or Republican parties on twitter platform. Just based on some phrases, people say “You sound just like a Democrat!” or “Are you an anti-Republican?” These posts inspired me to search for the extracted tweets dataset to conduct sentiment analysis and use ML algorithms to train a model and determine if the tweet was written by Democrat or a Republican handle/supporter.

This dataset is added to kaggle in 2018 by Kyle Pastor. This data set is about political data, which contains extracted tweets handled by representative parties and user information.

The data set consists of 3 columns –

**Party** : Name of the Party – Democrats or Republicans

**Handle**: Representative from a particular party who post the Tweet or Retweet

**Tweet**: Complete Tweet posted by the party handlers

It has 86460 tweets by 433 unique handle.
Party is the label which contains two classes- Democratic and Republican.
Among the extracted tweets, 51% belongs to Republican Party and 49% belongs to Democrat Party.
 
**Following is the kaggle link to access the dataset:**
https://www.kaggle.com/kapastor/democratvsrepublicantweets?select=ExtractedTweets.csv

·     Conducted Exploratory Data Analysis for 85000 tweets collected from Democrats and Republican twitter accounts

·     Preprocessed the data to eliminate Non-Ascii characters, STOPWORDS and performed Lemmatization and stemming

·     Built classification model by training the data over different classifiers such as Naïve Bayes model, Logistic Regression, Random forest and ADA Boost using TF-IDF Vectorizer to classify the Democrats and Republican Tweets

·     Developed a web application using Streamlit app to exhibit the predictive results. Evaluated the application with real time tweets.
