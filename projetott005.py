# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Criar diretório figures se não existir
os.makedirs('figures', exist_ok=True)

# Carregar dados
df = pd.read_excel("dados_simulados_RU.xlsx")

# Tratar a coluna Temperatura que contém valores mistos (números e datas)
def limpar_temperatura(valor):
    """Converte valores de temperatura para float, removendo datas inválidas"""
    try:
        # Se for uma string que contém data, extrair apenas números se possível
        if isinstance(valor, str) and '-' in valor:
            return np.nan  # Marcar como valor ausente
        # Tentar converter para float
        return float(valor)
    except (ValueError, TypeError):
        return np.nan

# Aplicar a limpeza na coluna Temperatura
df['Temperatura'] = df['Temperatura'].apply(limpar_temperatura)

# Preencher valores ausentes de temperatura com a média
df['Temperatura'] = df['Temperatura'].fillna(df['Temperatura'].mean())

# Remover colunas do tipo datetime (como 'Data'), se existirem
df = df.select_dtypes(exclude=['datetime'])

# Codificar variáveis categóricas (one-hot encoding para o dia da semana)
df_encoded = pd.get_dummies(df, columns=['Dia_da_Semana'], drop_first=True)

# Separar variáveis independentes (X) e variável alvo (y)
X = df_encoded.drop('Refeicoes_Servidas', axis=1)
y = df_encoded['Refeicoes_Servidas']

# Dividir os dados em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados para o MLP (apenas valores numéricos são permitidos)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(float))
X_test_scaled = scaler.transform(X_test.astype(float))

# Inicializar os modelos
modelos = {
    'Regressão Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'MLP Regressor': MLPRegressor(
        hidden_layer_sizes=(20, 10),
        max_iter=2000,
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        early_stopping=True,
        random_state=42
    )
}

# Avaliar os modelos
resultados = []

for nome, modelo in modelos.items():
    if nome == 'MLP Regressor':
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    resultados.append({
        'Modelo': nome,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    })

# Exibir os resultados em um DataFrame
df_resultados = pd.DataFrame(resultados)
print("\nAvaliação dos Modelos:\n")
print(df_resultados)

# Gerar gráfico de comparação dos erros
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_resultados.melt(id_vars='Modelo', value_vars=['MAE', 'RMSE']),
    x='Modelo', y='value', hue='variable'
)
plt.title('Comparação de Erros dos Modelos')
plt.ylabel('Erro')
plt.xlabel('Modelo')
plt.legend(title='Métrica')
plt.tight_layout()
plt.savefig("figures/comparacao_modelos.png")  # salvar figura (opcional)
plt.show()
