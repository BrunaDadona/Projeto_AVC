import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#carregar e limpar dados (colunas)

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Renomear colunas para portugues
df.rename(columns={
    'gender': 'sexo',
    'age': 'idade',
    'hypertension': 'hipertensao',
    'heart_disease': 'doenca_cardiaca',
    'ever_married': 'casado',
    'work_type': 'tipo_trabalho',
    'Residence_type': 'tipo_residencia',
    'avg_glucose_level': 'glicose_media',
    'bmi': 'imc',
    'smoking_status': 'tabagismo',
    'stroke': 'avc'
}, inplace=True)

df['sexo'] = df['sexo'].map({'Male': 'Masculino', 'Female': 'Feminino', 'Other': 'Outro'})
df['casado'] = df['casado'].map({'Yes': 'Sim', 'No': 'Nao'})
df['tipo_residencia'] = df['tipo_residencia'].map({'Urban': 'Urbana', 'Rural': 'Rural'})
df['tabagismo'] = df['tabagismo'].replace({
    'never smoked': 'Nunca fumou',
    'formerly smoked': 'Fumou anteriormente',
    'smokes': 'Fuma',
    'Unknown': 'Desconhecido'
})
df['tipo_trabalho'] = df['tipo_trabalho'].replace({
    'Private': 'Privado',
    'Self-employed': 'Autonomo',
    'Govt_job': 'Emprego publico',
    'Children': 'Criança',
    'Never_worked': 'Nunca trabalhou'
})
df['avc'] = df['avc'].map({0: 'Nao', 1: 'Sim' })

#mostra valores ausentes na coluna imc
df['imc'].fillna(df['imc'].median(), inplace=True)

#visualizar
sns.boxplot(x='avc', y='idade', data=df)
plt.title('Distribuição de Idade por Ocorrência de AVC')
plt.xlabel('AVC')
plt.ylabel('Idade')
plt.show()

#Histograma de idade dos pacientes com AVC
df_avc = df[df['avc'] == 'Sim']
plt.hist(df_avc['idade'], bins=20, color='tomato', edgecolor='black')
plt.title('Distribuição Etária dos Pacientes com AVC')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

#Mapa de calor de correlação
df_corr = df.copy()
df_corr['avc'] = df_corr['avc'].map({'Nao': 0, 'Sim': 1})
df_corr['casado'] = df_corr['casado'].map({'Sim': 1, 'Nao': 0})
df_corr['tipo_residencia'] = df_corr['tipo_residencia'].map({'Urbana': 1, 'Rural': 0})
df_corr['sexo'] = df_corr['sexo'].map({'Masculino': 0, 'Feminino': 1, 'Outro': 2})

#Remover colunas com texto para evitar erro
df_corr_numerico = df_corr.select_dtypes(include='number')

plt.figure(figsize=(12, 8))
sns.heatmap(df_corr_numerico.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Correlação entre Variáveis')
plt.show()

#odelagem simples
df_modelo = df.copy()
df_modelo['avc'] = df_modelo['avc'].map({'Nao': 0, 'Sim': 1})
df_modelo['casado'] = df_modelo['casado'].map({'Sim': 1, 'Nao': 0})
df_modelo['tipo_residencia'] = df_modelo['tipo_residencia'].map({'Urbana': 1, 'Rural': 0})
df_modelo['sexo'] = df_modelo['sexo'].map({'Masculino': 0, 'Feminino': 1, 'Outro': 2})
df_modelo = pd.get_dummies(df_modelo, columns=['tipo_trabalho', 'tabagismo'], drop_first=True)

X = df_modelo.drop(['id', 'avc'], axis=1)
y = df_modelo['avc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

#Fatores importantes
importances = pd.Series(modelo.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(10, 8), color='teal')
plt.title('Importância dos Fatores para AVC')
plt.xlabel('Importância')
plt.tight_layout()
plt.show()



#pd.set_option('display.max_columns', None)
#print(df.head())
#print(df.head(11))

#print(df.info())

#print(df.isnull().sum())

#print("Numero total de linhas:", len(df))



