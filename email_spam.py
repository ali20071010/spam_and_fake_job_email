import pandas as pd 

df  = pd.read_csv("enron_spam.csv")

da = df.head(5)

de = df.tail(5)

dw = df.isna().sum()

dh = df.duplicated().sum()

dup_count = df["text"].duplicated().sum()
 
value_label = df["label"].value_counts(normalize = True) 

df["word_count"] = df["text"].str.split().str.len()

avg_words = df.groupby("label")["word_count"].mean()

median_words = df.groupby("label")["word_count"].median()

conflicting_count = (df.groupby("text")["label"].nunique() > 1).sum()



df = df.dropna(subset=["text"])


df = df.drop_duplicates(subset=["text"])


x = df["text"]
y = df["label"]


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size =0.2 , random_state = 42 , stratify = y )
vectorizer = TfidfVectorizer(max_features = 50000)

x_train_tfidf = vectorizer.fit_transform(x_train) 
x_test_tfidf = vectorizer.transform(x_test) 

print(x_train_tfidf.shape)
print(x_test_tfidf.shape)

model = MultinomialNB()
model.fit(x_train_tfidf , y_train)

y_pred = model.predict(x_test_tfidf)
print("Accuracy = " , accuracy_score(y_test , y_pred))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
print(cm) 


import joblib

joblib.dump(model, 'MultinomialNB.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')





