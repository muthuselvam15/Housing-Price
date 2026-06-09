import pandas as pd
df = pd.read_csv('Housing.csv')
df.head()
df.tail()
df.describe()
df.info()
df.isna().sum()
List_1 = ['mainroad',	'guestroom', 'basement','hotwaterheating','airconditioning','prefarea']
df[List_1] = df[List_1].replace({'yes':1,'no':0})
df['furnishingstatus'].nunique()
df['furnishingstatus'] = df['furnishingstatus'].replace({'unfurnished':0,'semi-furnished':1,'furnished':2})
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
List_2 = ['price','area']
df[List_2] = scaler.fit_transform(df[List_2])
df.dtypes
corelation = df.corr()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.heatmap(corelation,annot=True,cmap='coolwarm')
plt.show()
df.hist(figsize=(10,10),bins = 10)
plt.show()
x = df.drop('price',axis=1)
y = df['price']
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)
from sklearn.metrics import r2_score
lr_accuracy = r2_score(y_test,y_predict)*100
lr_accuracy
