import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# CSVの読み込み（カンマ区切り）
df = pd.read_csv('data/wine.csv')

# 列名確認
print(df.columns)

# 説明変数
X = df[['alcohol', 'sulphates', 'volatile acidity']]

# 目的変数
y = df['quality']

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 線形回帰
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 決定係数
r2 = r2_score(y_test, y_pred)
print('R² =', r2)

# 回帰係数
print('\n回帰係数')
for name, coef in zip(X.columns, model.coef_):
    print(f'{name}: {coef:.4f}')

print('切片:', model.intercept_)

# 実測値と予測値の散布図
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()],
         [y.min(), y.max()],
         'r--')

plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Linear Regression: Actual vs Predicted Quality')
plt.grid()
plt.show()