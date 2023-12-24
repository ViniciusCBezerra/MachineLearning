import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

dados = pd.read_csv('aulas/parte_dois/arquivos_usados/marketing_investimento.csv')
x = dados.drop('aderencia_investimento',axis=1)
y = dados['aderencia_investimento']
colunas = x.columns

one_hot = make_column_transformer(
    (
        OneHotEncoder(drop='if_binary'),
        ['estado_civil','escolaridade','inadimplencia','fez_emprestimo']
    ),
    remainder='passthrough',
    sparse_threshold=0
)
x = one_hot.fit_transform(x)
colunas_transformadas = one_hot.get_feature_names_out(colunas)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#TREINO E TESTE

x_treino,x_teste,y_treino,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

dummy = DummyClassifier()
dummy.fit(x_treino,y_treino)
print(dummy.score(x_teste,y_teste))
