import pandas as pd 
df = pd.read_csv("fake_job_postings.csv")
"""
print(df.shape) 
print(df.head(1))
print(df.tail(1))
print(df.dtypes)

"""



""""
print(df["fraudulent"].value_counts())
print(df["fraudulent"].value_counts(normalize=True) * 100)

dup_row = df.drop(columns = "job_id").duplicated().sum()
dup_decs = df["description"].duplicated().sum()
print(dup_row)
print(dup_decs)

conflicting_desc = (df.groupby("description")["fraudulent"].nunique() > 1).sum()
print(conflicting_desc)

df["desc_words"] = df["description"].str.split().str.len()
print(df.groupby("fraudulent")["desc_words"].agg(["mean" , "median"]))

minus = df.isna().sum().sort_values(ascending = False)
print(minus)

"""

df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["company_profile"] = df["company_profile"].fillna("")
df["requirements"] = df["requirements"].fillna("")
df["benefits"] = df["benefits"].fillna("")

#print(df["title"].head(1))
#print(df["description"].head(1))
#print(df["company_profile"].head(1))
#print(df["requirements"].head(1))
#print(df["benefits"].head(1))

df["text"] = df["title"] + "  " + df["description"] + "  " + df["company_profile"] + "  " +  df["requirements"] + "  " + df["benefits"] 
df["text"] = df["text"].str.lower()

#print(df["text"].iloc[0])

#print(df.shape , df["fraudulent"].sum())
df = df.drop_duplicates(subset = "text")
#print(df.shape , df["fraudulent"].sum)


x = df["text"]
y = df["fraudulent"]


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 ,random_state = 42 , stratify = y  )   


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features = 100000)

x_train_tfidf = vectorizer.fit_transform(x_train) 
x_test_tfidf = vectorizer.transform(x_test) 

#print(x_train_tfidf.shape)
#print(x_test_tfidf.shape)


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier ( n_estimators = 100 ,class_weight = "balanced" , random_state = 42 , n_jobs = -1)
model.fit(x_train_tfidf , y_train)
y_pred = model.predict(x_test_tfidf)
#print("Accuracy = " , accuracy_score(y_test , y_pred))


from sklearn.metrics import confusion_matrix
metrics= confusion_matrix(y_test , y_pred)
#print(metrics) 

from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred))




import joblib


joblib.dump(model, "random_forest_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

