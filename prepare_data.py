import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE


def make_tokens(corpus):
    """"
    make_tokens() utilizes nltk's word_tokenize function to return a list of
    tokenized documents from a corpus.
    PARAMETERS:
        corpus - list(str) - a list of documents to be tokenized
    RETURNS:
        list(list(str)) - a list of lists of tokens for each document in the 
                     corpus
    
    """
    return [word_tokenize(str(document)) for document in corpus]

def lemmatize(tokenized_corpus):
    """
    lemmatize() uses nltk's WordNetLemmatizer to lemmatize each tokenized
    document and returns a list of tokenized and lemmatized strings
    corresponding to the corpus.
    PARAMETERS:
        tokenized_corpus - list(list(str)) - a list of lists of tokens for
                                             each document in the corpus
    RETURNS:
        list(str) - a list of strings that have been tokenized and lemmatized
    """
    lemmatizer = WordNetLemmatizer()
    return [" ".join([lemmatizer.lemmatize(word) for word in doc]) for doc in tokenized_corpus]

def tfidf_matrix(corpus, stop_words=set(nltk.corpus.stopwords.words('english')), extra_stop_words=[]):
    """
    tfidf_matrix() takes a corpus that has been tokenized and stemmed/lemmatized
    and returns a Tfidf object as well as a matrix.
    PARAMETERS:
        corpus - list(str) - a list of documents that have been tokenized and  
                             stemmed/lemmatized
        stop_words - set(str) - set of stop words (defaults to nltk english stop words)
        extra_stop_words - list(str) - a list of added stop words specific to the corpus
    RETURNS:
        tv - TfidfVectorizer() - a trained tfidfVectorizer() object
        X - sparse matrix of type '<numpy.float64'> - a tfidf matrix representing 
                                                      each document and all the
                                                      words in the corpus
            
    """
    # add extra stop words
    if len(extra_stop_words) > 0:
        for word in extra_stop_words:
            stop_words.add(word)
    
    tv = TfidfVectorizer(stop_words=stop_words, max_features=10000)
    X = tv.fit_transform(corpus)
    return tv, X
def tsne_map(X, n_dim=2, metric='euclidean'):
    """
    tsne_map() takes a matrix and returns a matrix with reduced dimensions using tsne.
    PARAMETERS:
        X - numpy.ndarray - the matrix to be used in the tsne computation
        n_dim - int - the number of feature columns that tsne maps the original matrix to
        metric - str - the type of distance metric used in the tsne object
    RETURNS:
        reduced - np.ndarray - the reduced matrix
    """
    tsne = TSNE(n_components=n_dim, metric=metric)
    reduced = tsne.fit_transform(X)
    return reduced

if __name__ == '__main__':
    # prepare nltk packages
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # create a balanced sample from the dataset 
    df = pd.read_csv('data/complaints_processed.csv')
    df.drop('Unnamed: 0',axis=1, inplace=True)
    df.dropna(inplace=True)
    # separate into categories and pull random samples from each
    cards = df[df['product'] =='credit_card']
    banking = df[df['product'] =='retail_banking']
    credit = df[df['product'] =='credit_reporting']
    mortgage = df[df['product'] =='mortgages_and_loans']
    debt = df[df['product'] =='debt_collection']

    balanced = cards.sample(3000)
    balanced = balanced.append(banking.sample(3000), ignore_index=True)
    balanced = balanced.append(credit.sample(3000), ignore_index=True)
    balanced = balanced.append(mortgage.sample(3000), ignore_index=True)
    balanced = balanced.append(debt.sample(3000), ignore_index=True)

    # save as parquet file
    balanced.to_parquet('data/balanced_sample')


    # read file, assign feature and target objects
    complaints = pd.read_parquet('data/balanced_sample')
    X = complaints['narrative'].values
    y = complaints['product'].values

    # tokenize, lemmatize, and transform the corpus into a tfidf matrix
    stopWords = ['account','bank','report','reporting','company','would',\
        'payment','would','time','information','paid','pay','day','collection',\
        'late','received','debt','call','called','get','said','told']
    tokens = make_tokens(X)
    lemmatized = lemmatize(tokens)
    tv, matrix= tfidf_matrix(lemmatized, extra_stop_words=stopWords)

    #save tfidf vectorizer object
    pickle.dump(tv, open('tfidf_vectorizer.pkl','wb'))
    
    # add tsne features to original tfidf matrix
    reduced = tsne_map(matrix.todense())
    matrix = np.hstack((matrix.todense(),reduced))

    # split into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(matrix, y, stratify=y)

    #  save as parqet files
    cols = tv.get_feature_names()
    cols.append('tsne1')
    cols.append('tsne2')
    df_train = pd.DataFrame(X_train, columns=cols)
    df_test = pd.DataFrame(X_test, columns=cols)
    df_train['Target'] = y_train
    df_test['Target'] = y_test
    df_train.to_parquet('data/train')
    df_test.to_parquet('data/test')
