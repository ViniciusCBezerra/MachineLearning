from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px

dados = pd.read_csv('aulas/parte_um/arquivos_usados/marketing_investimento.csv')

x = dados.drop('aderencia_investimento', axis=1)
y = dados['aderencia_investimento']
colunas = x.columns

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

one_hot = make_column_transformer(
    (OneHotEncoder(drop='if_binary'), ['estado_civil', 'escolaridade', 'inadimplencia', 'fez_emprestimo']),
    remainder='passthrough',
    sparse_threshold=0
)
x = one_hot.fit_transform(x)
colunas_transformadas = one_hot.get_feature_names_out(colunas)
dados_transformado = pd.DataFrame(x,columns=colunas_transformadas)

# TREINO E TESTE

x_treino,x_teste,y_treino,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

