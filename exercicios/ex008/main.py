import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

dados = pd.read_csv('exercicios/ex008/arquivos_usados/president_heights.csv')
x = dados.drop('height(cm)',axis=1)
y = dados['height(cm)']
colunas = x.columns

one_hot = make_column_transformer(
    (
        OneHotEncoder(drop='if_binary'),
        ['name']
    ),
    remainder='passthrough',
    sparse_threshold=0
)
x = one_hot.fit_transform(x)
colunas_transformadas = one_hot.get_feature_names_out(colunas)
print(pd.DataFrame(x,columns=colunas_transformadas))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TREINO E TESTE

x_treino,x_teste,y_treino,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)


