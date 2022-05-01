# twitter-emotion
code to predict twitter emotion
import pandas as pd
!git clone https://github.com/Sonali210/Twitter_Classification.git
ls
cd Twitter_Classification/
ls
cd Dataset/
ls
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
validation = pd.read_csv('validation.csv')
#Print our top 5 rows of train data
train.head()
#To read emotions we make a new column(description) defining the label emotion and print head data
labels_description = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
train['description'] = train['label'].map(labels_description )
train.head()
train.describe
#Count of emotions 
train.description.value_counts(normalize=True)
#Visualise count of emotions
import seaborn as sns
import matplotlib.pyplot as plt
emotion_val=train.groupby('description').count()
plt.bar(emotion_val.index.values, emotion_val['text'])
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.show()
#Create a column for length of tweets
train['text_length'] = train['text'].astype(str).apply(len) 
#Create a column for count of words in single tweet
train['text_word_count'] = train['text'].apply(lambda x: len(str(x).split()))
train.head()
#Visualise text length graph
sns.distplot(train['text_length'])
plt.xlim([0, 512]);
plt.xlabel('Text Length');#Maximum length of text
train.text_length.max()
import unicodedata
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

#Converting unicode to ascii 
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

#Removing stopwords and shortwords
def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

#Combined function that will be called to preprocess our text data
def preprocess_sentence(w):
    #lowercase all the text
    w = unicode_to_ascii(w.lower().strip())
    #Remove puntuations
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    #Calling stopword function
    w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w
    #While training our model, we only need tweet and label, therefore we drop all other columns
train=train.drop(['description','text_length','text_word_count'],axis=1)
train=train.reset_index(drop=True)
train.head()
#We define a function to call data from train,validation and testing data
def get_tweet(data):
  tweets = data['text']
  labels = data['label']
  return tweets, labels
  #Call train data
tweets, labels = get_tweet(train)
#Print first row of train data
tweets[0], labels[0]
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)
  #Print tokens for first tweet data
tokenizer.texts_to_sequences([tweets[0]])
  tweets[0]
  maxlen = 200

from tensorflow.keras.preprocessing.sequence import pad_sequences
#A function to return padded sentence
def get_sequences(tokenizer, tweets):
  sequences = tokenizer.texts_to_sequences(tweets)
  padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=maxlen)
  return padded
  #calling train data for padding
padded_train_seq = get_sequences(tokenizer, tweets)
#print first tweet after padding
padded_train_seq[0]
  import tensorflow as tf
model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 16, input_length=maxlen), #Turns positive integers (indexes) into dense vectors of fixed size
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(20, activation='tanh')),
        tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    loss = 'sparse_categorical_crossentropy', #for multiclassification 
    optimizer = 'adam',
    metrics = ['accuracy']
)
  model.summary()
  #Calling validation data
val_tweets, val_labels = get_tweet(validation)
#tokenizing tweet text
val_seq = get_sequences(tokenizer, val_tweets)
  #Print validation tweet and label for first row
val_tweets[0], val_labels[0]
  h = model.fit(
    padded_train_seq, labels,
    validation_data=(val_seq, val_labels),
    epochs=25,
    callbacks=[
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    ]
)
  #Plot graph for loss and accuracy

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

show_history(h)
  test_tweets, test_labels = get_tweet(test)
test_seq = get_sequences(tokenizer, test_tweets)
  score =model.evaluate(test_seq, test_labels)
  import numpy as np
import random
#calling a random test tweet and its label stored and label predicted
i = random.randint(0, len(test_labels) -1)
print("Tweet: ", test_tweets[i])
print("Emotion: ", test_labels[i])

p = model.predict(np.expand_dims(test_seq[i], axis=0))[0]
pred_class = np.argmax(p).astype('uint8')

print('Predicted Emotion: ', pred_class)
