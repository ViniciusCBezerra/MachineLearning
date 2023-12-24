import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('exercicios/ex009/arquivos_usados/births.csv')
x = df.drop('births',axis=1)
y = df['births']
colunas = x.columns

one_hot = make_column_transformer(
    (
        OneHotEncoder(drop='if_binary'),
        ['gender']
    ),
    remainder='passthrough',
    sparse_threshold=0
)
x = one_hot.fit_transform(x)
colunas_transformadas = one_hot.get_feature_names_out(colunas)
df_transformado = pd.DataFrame(x,columns=colunas_transformadas)

# TREINO & TESTE

x_treino,x_teste,y_treino,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

arvore = DecisionTreeClassifier(
    max_depth=3,
    random_state=5
)
arvore.fit(x_treino,y_treino)

