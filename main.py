# Import statements
import clean
import pandas as pd
import entity_recognition

if __name__ == "__main__":

     # Read-in prediction data
     df = pd.read_csv("articles.csv", encoding="latin1")

     # Set 'df' variable as first 10 rows of 'df'
     df = df.head(10)

     # Calls clean function in clean.py to clean 'df' dataframe
     df = clean(df)

     # Calls setup() function in entity_recognition.py
     entity_recognition.setup()

     # Applies predict_tags() function in entity_recognition.py on each row of df under ['text']
     df['Entities'] = df.apply(lambda row: entity_recognition.predict_tags(row['text']), axis=1)
