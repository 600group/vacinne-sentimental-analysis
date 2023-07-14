# vaccine-term-project

The term project mainly is divided into five parts. 

- DataCollection
- DataProfile
- SentimentAnalysisModel
- SarcasmDetectModel
- Prediction



## Data Collection

- Use Tweepy4.8.0 to collect Twitter data about Covid-19 vaccine.



## DataProfile

Before you run the program, you need

- install pandas
- Put "Raw_Data_Labeled.csv" at the same folder.



## SentimentAnalysisModel

To run the model, you need to obey the following steps

- open jupyter notebook
- open sentimentalAnalysis.ipynb
- change tweetFile = pd.read_csv("") to your csv file directory
- align csv column name with examples
- run it



## SarcasmDetectModel

Verion information:

- Python: 3.8.3
- Tensorflow: 2.3.0
- Numpy: 1.18.5
- NLTK: 3.6.5

Step for running model

- Download glove model: https://nlp.stanford.edu/projects/glove/ and unzip it.
- If you want to train this model, delete the folder "model".
- Run the sarcasm_detection.py
- Run plot.py for accuracy and loss plot



## Prediction

Before running you need to 

- install pandas
- Put the sentiment_sarcasm_prediction.csv in the same fold

The Prediction.ipynb will generate the Final_prediction.csv, which contains the final results in the column of "Attitude"
