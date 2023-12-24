import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


dados = pd.read_csv('exercicios/ex002/arquivos_usados/marketing_investimento.csv')

variaveis_explicativas = dados.drop('aderencia_investimento',axis=1)
colunas = variaveis_explicativas.columns

one_hot = make_column_transformer((
    OneHotEncoder(drop='if_binary'),
    ['estado_civil','escolaridade','inadimplencia','fez_emprestimo']
),remainder='passthrough',
sparse_threshold=0
)

variaveis_explicativas = one_hot.fit_transform(variaveis_explicativas)

pd.DataFrame(variaveis_explicativas,columns=one_hot.get_feature_names_out(colunas))
