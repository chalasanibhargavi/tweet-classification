"""
                ##### TWEET CLASSIFICATION #####
    The following code takes a set of tweets as input and predicts the location of their origin

    The predictions are made based on a Multinomial Naive Bayes model where the probability of the label/location
    given the tweet is calculated as follows.

    P(label/ words_in_tweet) = P(words_in_tweet/label) * P(label) / P(words_in_tweet)

    A simple bag of words approach is used to obtain the words associated with a tweet. The occurence of a word is
    considered as binary: present or not present to ensure that probabilities are not affected by repeated terms which
    is expected in twitter data.

    Data is preprocessed to remove stop-words and duplicate terms.

    Design decision: If a term in the tweet is a substring of the location name, the term is given a higher weight

"""
# coding: utf-8

import pandas as pd
import re
import unicodedata
from datetime import datetime
import sys
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")


##Stop word list
stop_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

digit_list = [0,1,2,3,4,5,6,7,8,9]

##Function to read the file
def read_file(fname, file_type):
    city_label = []
    bow_list = []
    raw_list = []
    
    file_inp = open(fname, "r")   
    for line in file_inp.readlines():
	line1 = line.split(" ")[1:]
	line_str = ' '.join(line1)
        raw_list.append(line_str)

        l1 = line.replace(',_','_')
        l2 = re.sub(r'\W+', ' ', l1).split(' ')
        
        city_label.append(l2[0])
        bow_list.append(l2[1:])
     
    if file_type == 'test':
        return raw_list, city_label, bow_list
    else:
        return city_label, bow_list

##Function to clean the tweets - Remove duplicates, lower case, remove stop words
def clean_tweets(bow_list):
    bow1 = [j.lower() for i in set(bow_list) if i!='' and len(i)>1 for j in i.split("_") if j.isalnum() and len(j)>1 and i not in digit_list]
    #bow_stem = [stemmer.stem(i) if i not in city_terms else i for i in bow1 ]
    bow2 = [i for i in set(bow1) if i not in stop_list]
    
    return set(bow2)

##Function to calculate class probability
def class_probability(clabel):
    
    cval = class_count_dict[clabel]
    return float(cval)/ all_clabel_count


##Function to calculate conditional probability
def conditional_probability(term, clabel):
    
    vocab_count = vocab_label_dict[clabel]
    fin_key = term+'_'+clabel
    
    if fin_key in term_label_dict:
        term_clabel_count = term_label_dict[fin_key]
    else:
        
        term_clabel_count = 0.000000001
    
    return float(term_clabel_count)/vocab_count


##Function to get posterior probability
def posterior_probability(bow_list,clabel):
    post_prob = float(1)
    for word in bow_list:
        post_prob = post_prob * conditional_probability(word,clabel)
    
    post_prob2 = post_prob * class_probability(clabel)
        
    return post_prob2

##Function to predict the class label
def predict(bow_list):
    predict_probabilities = []
    for cname in city_vals:
        predict_probabilities.append(posterior_probability(bow_list, cname))
     
    max_index = predict_probabilities.index(max(predict_probabilities))
    
    return city_vals[max_index]

##Functions to manipulate class labels
def update_label(lval):
    return city_map_dict[lval]

def map_back_label(lval):
    return rev_map_dict[lval]


##Get file inputs
train_filename = sys.argv[1]
test_filename = sys.argv[2]
output_filename = sys.argv[3]

##Read train data
y_list, x_list = read_file(train_filename, "train")
df_tweets = pd.DataFrame(y_list, columns=['city_label'])
df_tweets['bow_list'] = x_list
print "Training data size: ",df_tweets.shape[0]

city_map_dict = {'Los_Angeles_CA':'LA', 'San_Francisco_CA':'SF', 'Manhattan_NY':'MHT',
       'San_Diego_CA':'SD', 'Houston_TX':'HT', 'Chicago_IL':'CHI', 'Philadelphia_PA':'PL',
       'Toronto_Ontario':'TR', 'Atlanta_GA':'ATL', 'Washington_DC':'DC', 'Boston_MA':'BST',
       'Orlando_FL':'ORL'}
city_vals = city_map_dict.values()

city_terms = [j for i in city_map_dict.keys() for j in i.lower().split("_")]

df_tweets['city_label_new'] = df_tweets['city_label'].apply(update_label)

df_tweets['bow_clean'] = df_tweets['bow_list'].apply(clean_tweets)

##Get frequency of classes
df_class_count = pd.DataFrame(df_tweets['city_label_new'].value_counts())
df_class_count.reset_index(inplace=True)
df_class_count.columns=['city','label_count']

class_count_dict = df_class_count.set_index('city')['label_count'].to_dict()

all_clabel_count = df_class_count['label_count'].sum()

print "Training the model.."

vocab_label_dict = {}
term_label_dict = {}
for corgname, cname in city_map_dict.iteritems():
    bow_all_list = df_tweets[df_tweets['city_label_new'] == cname]['bow_clean'].tolist()
    
    bow_all_list2 = set([j for i in bow_all_list for j in i])
    vocab_label_dict[cname] = len(bow_all_list2)
    
    cntr = 0
    for term in bow_all_list2: 
    
        tmp_list = [term for ilst in bow_all_list if term in ilst]

        key_str = term+'_'+cname
        
        if term in corgname.lower().split("_") or term!=term:
            
            val1 = len(tmp_list) * 1.75
        else:
            val1 = len(tmp_list)
            
        term_label_dict[key_str] = val1


##Get unique list of words
ubow_list = [j for i in df_tweets['bow_clean'].tolist() for j in i]
ubow_list = list(set(ubow_list))

##Get the top words associated with each label
label_prob_dict = {}
for cname in city_vals:
    
    prob_list = []
    for uword in ubow_list:
    
        ##Get posterior probability
        prob_list.append((uword, conditional_probability(uword,cname) * class_probability(cname)))
    
    label_prob_dict[cname] = sorted(prob_list, key=itemgetter(1), reverse=True)

rev_map_dict = {'LA':'Los_Angeles_CA', 'SF':'San_Francisco_CA', 'MHT':'Manhattan_NY',
       'SD':'San_Diego_CA', 'HT':'Houston_TX', 'CHI':'Chicago_IL', 'PL':'Philadelphia_PA',
       'TR':'Toronto_Ontario', 'ATL':'Atlanta_GA', 'DC':'Washington_DC', 'BST':'Boston_MA',
       'ORL':'Orlando_FL'}

print("\n\n****Top words associated with cities****")
for k,v in label_prob_dict.iteritems():
    word_list = [i[0] for i in v[:5]]
    word_str = ', '.join(word_list)
    print("\n"+rev_map_dict[k]+" --> Words: "+word_str)

##Make predictions
df_tweets['predicted_label'] = df_tweets['bow_clean'].apply(predict)

##Training accuracy
df_sub1 = df_tweets[df_tweets['city_label_new'] == df_tweets['predicted_label']]
train_acc = round(100*float(df_sub1.shape[0])/df_tweets.shape[0] , 3)
print "\n\n***Training accuracy: ", train_acc,"%"



##Get Test Data
raw_list, y_test, x_test = read_file(test_filename,"test")

df_test = pd.DataFrame(y_test, columns=['city_label'])
df_test['bow_list'] = x_test
df_test['raw_tweet'] = raw_list
print "\nTest data size: ",df_test.shape[0]


df_test['bow_clean'] = df_test['bow_list'].apply(clean_tweets)
df_test['city_label_new'] = df_test['city_label'].apply(update_label)

##Make predictions
df_test['predicted_label'] = df_test['bow_clean'].apply(predict)

##Testing accuracy
df_sub2 = df_test[df_test['city_label_new'] == df_test['predicted_label']]
test_acc = round(100*float(df_sub2.shape[0])/df_test.shape[0], 3)

print "\n***Testing accuracy: ", test_acc,"%"


##Writing test data predictions to file
df_outfile = df_test[['predicted_label','city_label_new','raw_tweet']]
df_outfile['predicted_label'] = df_outfile['predicted_label'].apply(map_back_label)
df_outfile['city_label_new'] = df_outfile['city_label_new'].apply(map_back_label)

print("\nTest data predictions written to: "+output_filename)
df_outfile.to_csv(output_filename, sep=' ', index=False, header=False)

