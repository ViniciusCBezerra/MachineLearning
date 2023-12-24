from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import pandas as pd
import plotly.express as px

dados = pd.read_csv('aulas/parte_um/arquivos_usados/marketing_investimento.csv')

variaveis_explicativas = dados.drop('aderencia_investimento', axis=1)
variavel_alvo = dados['aderencia_investimento']

label_encoder = LabelEncoder()

variavel_alvo = label_encoder.fit_transform(variavel_alvo)

colunas = variaveis_explicativas.columns
one_hot = make_column_transformer(
    (OneHotEncoder(drop='if_binary'), ['estado_civil', 'escolaridade', 'inadimplencia', 'fez_emprestimo']),
    remainder='passthrough',
    sparse_threshold=0
)

variaveis_explicativas_transformadas = one_hot.fit_transform(variaveis_explicativas)
colunas_transformadas = one_hot.get_feature_names_out(colunas)

df_transformado = pd.DataFrame(variaveis_explicativas_transformadas, columns=colunas_transformadas)