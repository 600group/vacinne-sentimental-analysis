import pandas as pd
import numpy as np
import os
import re
import string
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model


def clean_text(text):
    text = text.lower()   
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                        u"\U0001F600-\U0001FFFF"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text

def CleanTokenize(df):
    nltk.download('punkt')
    nltk.download('stopwords')
    head_lines = list()
    lines = df["headline"].values.tolist()

    for line in lines:
        line = clean_text(line)
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        head_lines.append(words)
    return head_lines

# read the training (.json and .csv) files
data_1 = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
data_2 = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)
data_3 = pd.read_csv('data.csv', encoding='utf-8')

# data_3 preprossing:
# remove redundant columns
data_3 = data_3[['text', 'isSarcasm', 'Unnamed: 10']]

# rename columns
data_3 = data_3.rename(columns={'text': 'headline', 'isSarcasm': 'is_sarcastic', 'Unnamed: 10': 'article_link'})

# reorder columns
cols = data_3.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_3 = data_3[cols]

# remove rows that contain NaN values
data_3 = data_3[~data_3['headline'].isna() & ~data_3['is_sarcastic'].isna()]
data_3 = data_3.loc[data_3['is_sarcastic'].isin(['1','0'])]

# type conversion
data_3['is_sarcastic'] = data_3['is_sarcastic'].astype('int64')

# combining seperate dataset, data = data_1 + data_2 + data_3
data =  pd.concat([data_1, data_2, data_3])
print('dataset size:', data.shape)

# clean the raw text
head_lines = CleanTokenize(data)

# hyper parameters: length of each text
max_length = 25

tokenizer_obj = Tokenizer()

# fit the model, update internal vocabulary
tokenizer_obj.fit_on_texts(head_lines)

# if you choose our pre-trained model
if os.path.exists('./model/'):
    model = load_model('model')
# if you choose to train a new model
else:
    sequences = tokenizer_obj.texts_to_sequences(head_lines)

    word_index = tokenizer_obj.word_index
    print("unique tokens - ", len(word_index))
    vocab_size = len(tokenizer_obj.word_index) + 1
    print('vocab size -', vocab_size)

    # accuracy drops if use the longest length instead of the max_length
    # short text has its padding
    lines_pad = pad_sequences(sequences, maxlen=max_length, padding='post')

    # if text is sarcastic or not, ground truth
    sentiment = data['is_sarcastic'].values

    # shuffle dataset
    indices = np.arange(lines_pad.shape[0])
    np.random.shuffle(indices)
    lines_pad = lines_pad[indices]
    sentiment = sentiment[indices]

    # segment training and test data
    validation_split = 0.2
    num_validation_samples = int(validation_split * lines_pad.shape[0])
    x_train_pad = lines_pad[:-num_validation_samples]
    y_train = sentiment[:-num_validation_samples]
    x_test_pad = lines_pad[-num_validation_samples:]
    y_test = sentiment[-num_validation_samples:]

    # an embedding vocabulary used for word embedding
    file = open('glove.twitter.27B.100d.pkl', 'rb')
    embeddings_glove = pickle.load(file)

    # hyper parameters: dembedding dimension
    embedding_dim = 100
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    # embed each word in the text
    c = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_glove.get(word)
        if embedding_vector is not None:
            c+=1
            embedding_matrix[i] = embedding_vector
    print('The number of words founded in Glove word embeddings:', c)

    # define the embedding layer
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)

    # build a model
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # training
    history = model.fit(x_train_pad, y_train, batch_size=32, epochs=25, validation_data=(x_test_pad, y_test), verbose=1)

    # save trained model
    model.save('model')

# prediction
# read the (.csv) file for prediction
pred_pd = pd.read_csv('vaccination_all_tweets.csv')

# remove redundant columns
pred_pd = pred_pd[['text']]

# rename columns
pred_pd = pred_pd.rename(columns={'text': 'headline'})

def predict_sarcasm(dataframe):
    test_lines = CleanTokenize(dataframe)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    pred = model.predict(test_review_pad)
    pred = pred.squeeze()

    # threshold = 50%
    pred *= 100
    pred[pred < 50] = 0
    pred[pred >= 50] = 1
    
    # add a column for predictions
    dataframe['is_sarcastic'] = pred

predict_sarcasm(pred_pd)
print('Prediction result:')
print('{} tweets contain sarcasm.'.format(pred_pd.loc[pred_pd['is_sarcastic'] == 1].shape[0]))
pred_pd.to_csv('sarcasm_prediction.csv')
pass