from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

model_lr = LogisticRegression()
model_lr.fit(xv_train, y_train)
pred_lr = model_lr.predict(xv_test)

model_dt = DecisionTreeClassifier()
model_dt.fit(xv_train, y_train)
pred_dt = model_dt.predict(xv_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

print("Decision Tree Accuracy:", accuracy_score(y_test, pred_dt))
print(classification_report(y_test, pred_dt))

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.countplot(x='class', data=data)
plt.title('Distribution of Real and Fake News')
plt.show()
