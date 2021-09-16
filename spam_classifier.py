import  pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
data = pd.read_csv("SPAM-210331-134237.csv",encoding="cp1252")



ws = WordNetLemmatizer()

preprocess_text = []

for i in range(0,len(data)):
    remove_punct = re.sub('[^a-zA-Z]', " ", data['text'][i])
    words = remove_punct.split()
    words = [word.lower() for word in words]
    remove_stopwords = [ws.lemmatize(word) for word in words if not word in set(stopwords.words('english'))]
    sentence = ' '.join(remove_stopwords)
    preprocess_text.append(sentence)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(preprocess_text).toarray()

y = pd.get_dummies(data['type']).iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB()
model = spam_model.fit(x_train,y_train)

y_pred = model.predict(x_test)


score1 = model.score(x_test, y_test)
print(score1)

'''from sklearn.linear_model import LogisticRegression
spam_model = LogisticRegression()
model = spam_model.fit(x_train, y_train)
y_pred = model.predict(x_test)'''
