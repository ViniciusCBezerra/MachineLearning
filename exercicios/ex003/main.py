import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer

dados = pd.read_csv('exercicios/ex003/arquivos_usados/marketing_investimento.csv')

variaveis_explicativas = dados.drop('aderencia_investimento',axis=1)
variavel_alvo = dados['aderencia_investimento']
colunas = variaveis_explicativas.columns

label_encoder = LabelEncoder()
variavel_alvo = label_encoder.fit_transform(variavel_alvo)

one_hot = make_column_transformer(
    (
        OneHotEncoder(drop='if_binary'),
        ['estado_civil','escolaridade','inadimplencia','fez_emprestimo']
    ),
    remainder='passthrough',
    sparse_threshold=0
)
variaveis_explicativas = one_hot.fit_transform(variaveis_explicativas)
colunas_transformadas = one_hot.get_feature_names_out(colunas)

pd.DataFrame(variaveis_explicativas,columns=colunas_transformadas)
variavel_alvo