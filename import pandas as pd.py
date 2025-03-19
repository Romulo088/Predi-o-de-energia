import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 🔍 1. Carregar o dataset
energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# 📊 2. Exploração dos dados
print("\n🔍 Primeiras linhas dos dados:")
print(X.head())
print("\n🎯 Primeiras linhas dos alvos:")
print(y.head())

# 🔍 3. Verificar se há valores nulos
print("\n📌 Valores nulos por coluna:")
print(X.isnull().sum())

# 🛠️ 4. Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 5. Treinar um modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 📊 6. Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# 📈 7. Avaliar o modelo com MSE
mse = mean_squared_error(y_test, y_pred)
print(f"\n🔢 Erro Quadrático Médio (MSE): {mse:.2f}")

# 🎯 1. Gráfico de comparação entre valores reais e previstos
plt.figure(figsize=(8, 6))
plt.scatter(y_test["Y1"], y_pred[:, 0], alpha=0.5, label="Y1")
plt.scatter(y_test["Y2"], y_pred[:, 1], alpha=0.5, label="Y2", color="red")
plt.plot([min(y_test["Y1"]), max(y_test["Y1"])], [min(y_test["Y1"]), max(y_test["Y1"])], linestyle="--", color="black")
plt.xlabel("Valores Reais")
plt.ylabel("Valores Preditos")
plt.title("Valores Reais vs. Preditos")
plt.legend()
plt.show()

# 📉 2. Gráfico de resíduos (diferença entre real e previsto)
residuos_Y1 = y_test["Y1"] - y_pred[:, 0]
residuos_Y2 = y_test["Y2"] - y_pred[:, 1]

plt.figure(figsize=(8, 6))
sns.histplot(residuos_Y1, bins=20, kde=True, label="Y1", color="blue", alpha=0.6)
sns.histplot(residuos_Y2, bins=20, kde=True, label="Y2", color="red", alpha=0.6)
plt.axvline(0, linestyle="--", color="black")
plt.xlabel("Erro (Resíduo)")
plt.ylabel("Frequência")
plt.title("Distribuição dos Resíduos")
plt.legend()
plt.show()

# 🔍 3. Matriz de correlação entre os recursos e os alvos
df_corr = pd.concat([X, y], axis=1)  # Junta os dados de entrada e saída
plt.figure(figsize=(10, 6))
sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()
