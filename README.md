# project-group-11
# Working with Large Data Set --- DNSC 6290

## 1. Executive Summary
> Google Books Ngrams is a series of data sets unveiled by Google. It contains word occurrence from a large corpus of books, which is a good data set for the examination of cultural change as it is reflected in books. This project is going to use the British English section in Google Ngram (1-gram is 2GB and 3-gram is 46.8GB) to study the popularity and occurrence fluctuation of specific words and build a Language Generation Model in NLP.  
Data Introduction Link:  
> https://aws.amazon.com/datasets/google-books-ngrams/  

## 2. Project Introduction
N-grams are fixed size tuples of items, which are words extracted from the Google Books corpus. The n-grams in this dataset were produced by passing a sliding window of the text of books and outputting a record for each new token. The data set we used is British English 1-gram(2GB) and British English 3-gram(46.8GB), they are essentially the same data set with a different structure. As 1-gram is just for specific words, which is good for us to know more about single words, including their frequency and the changes for use, we will primarily use the 1-gram data set for data exploration and 3-gram for NLP modeling.  
To better illustrate n-gram structure: the sentence “`Working with Large Datasets`” would produce the following 2-grams: 

["Working with"] 
["with Large"] 
["Large Datasets"] 

Or the following 3-grams: 

["Working with Large"] 
["with Large Datasets"] 

The goal for this project is to utilize this unique data structure (specifically 3-gram) to predict the following words given two preceding words. 


## 3. Exploratory Analysis
In order to speed up, we use Spark to process our large data set. We use `spark.sparkContext.sequenceFile` reading 1-gram data into RDDs (`s3://datasets.elasticmapreduce/ngrams/books/20090715/eng-gb-all/1-gram`). The structure of the sequence file look like this: 

`(1, '#\t1547\t1\t1\t1')`

The key is the row number of the dataset and the value is the raw data containing the following fields:

* n-gram - The actual n-gram
* year - The year for this aggregation
* occurrences - The number of times this n-gram appeared in this year
* pages - The number of pages this n-gram appeared on in this year
* books - The number of books this n-gram appeared in during this year

1-gram data set has 188,660,459 rows, while the 3-gram data set has 5,186,054,851 rows. Those data sets contain lists of words or word combinations from British English books before 2009. Since it is inconvenient to calculate means, median or other statistics for words, we produce word cloud and trend viewer later to explore our data sets further.

## 4. Analysis Methods 
### 4.1 Data Cleaning and Processing
Generally speaking, our data set is clean and tidy. There’re only three things that need to be cleaned. 
* As the previous exploration shows, the structure of each row is a tab-separated raw text;
* For 1-gram, it treats punctuations as different words. The huge volume of different punctuations will occupy our word cloud if we don’t remove them from DataFrame;
* 3-gram data set contains some irregular word combinations. In other words, some word combinations in 3-gram only have 2-gram rather than 3 words. We handle these problems by eliminating punctuations in 1-gram and NULLs in 3-gram. 

We also provide our solutions respectively for each problem.
* Using the regular expression, we define a function called `process_raw()` which can separate different elements in the raw text, and return specific parts under different scenarios;
* Punctuations will only affect plotting the word cloud but have no impact on NLP;
* the number of Nulls word combinations is less than 0.4% of the data. Since it is only a tiny portion of the data set, the integrity of the data is preserved. 

### 4.2 Data Visualization
We’ve come up with two visualizations to help us extract the most important information from the data set. Word cloud is used to show the hottest word or phrase from the data set. Trend viewer is the plot of occurrence frequency for some words in chronicle order. The sample output as followings:

Word cloud: Most frequently used one-gram word in the data set

![image](https://github.com/gwu-bigdata/project-group-11/blob/master/IMG/Screen%20Shot%202020-06-28%20at%206.03.14%20PM.png)
 
The word cloud result is in accordance with our hypothesis that the most frequent words are prepositions. Words `the`, `of` and `and` are the top 3 popular words across our 1-gram data set.


Trend viewer: eg. The popularity of the words “`nuclear`” and “`expressionism`”
![image](https://github.com/gwu-bigdata/project-group-11/blob/master/IMG/pic%202.png)
 
The trend viewer is designed to examine a word’s average popularity and its first occurrence. For the example above, the word “`nuclear`” was first introduced around 1700 and began widely discussed in the 19th and 20th centuries. Similarly, the word “`expressionism`” began popular in the 20th century. 

### 4.3 Trigrams Language Model

Language modeling is the task of predicting what word comes next or more generally, a system that assigns a probability to a piece of a text sequence. This model is based on the Markov assumption, which means we approximate the probability of a specific word by looking only at the last several words of the context. A 3-gram language model predicts the probability of a given 3-gram within any sequence of words in the language. If our model works well, we can predict `p(w | h1, h2)`, which is the probability of word `w` given the history of previous words `h1` and `h2`.

To build the language model, we use the dataset of British English 3-gram(46.8GB). In order to calculate the probabilities, we split the trigram tuples into the first two words column `word_12` and the third-word column `word_3`, and count the total frequency for every word combination across years. After transformation, the data frame looks like below:

![image](https://github.com/gwu-bigdata/project-group-11/blob/master/IMG/pic%203.png)

Given the first two words for predicting the next word, we calculate the probability (the frequency of a three-words combination divided by the frequency of the first two-words combination). Using two words for example “`I am`” to test the model, we got the following probability for the next word:

![image](https://github.com/gwu-bigdata/project-group-11/blob/master/IMG/pic%204.png)

We set a random threshold between 0.045 and 0.005. Our model will find all words based on the input two words and record them into a candidate list. If the probability of the found words less than the random threshold, they will be removed from the candidate list. Then our model will randomly pick one of the words from the candidate list. Our model will repeat this process until the probability is under the threshold.

## 5. Result and Conclusions

The following is a sample output from the language model. The first two words in 3-grams were given as “`I am`”, and the model is trying to interpret the following words. The way we examine our result is intuitive, if the model generated sentences make sense, then it would be successful. The following output resembles an actual sentence for the first half, but as it stretches out it will become irrelevant to the first half of the sentence. Because the model is based on 3-grams, it will have the highest predictability for the first 3 to 4 words.

![image](https://github.com/gwu-bigdata/project-group-11/blob/master/IMG/pic%205.png)
 
## 6. Challenges

### 6.1 Technical part:
1. How to effectively allocate computing resources for various tasks is something to think about. 
2. Debugging with a large data set is not intuitive. 	

### 6.2 Non-technical part:
1. The data was collected before July 2009 which might be outdated.
2. The data set is biased because it was solely based on British English literature.

## 7. Future work
There are many ways to elaborate this project into a mature search engine or a mature sentence-created model, some of the ideas including:
1. Using streaming data can produce more relevant results than static data.
2. Increase the data scope to include more English literature. 
3. Using 4 grams or even 5 grams rather than 3 grams to increase the rationality of sentences produced by our NPL model.


## Takeaways from the Course:

In our project, we mainly use things about Spark section learning from this course. Because we're processing 3 grams data set which is larger than 46GB, Spark allows effectively in-memory computations, while Hadoop MapReduce needs to read and write to the disk every time it runs. So running this project in Spark saves us a lot of time. And it is also impressive to see how a dataset with billions of rows can be processed without long wait times. In addition, we also convert RDDs to DataFrame and use SparkSQL to extract targeted information. 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
