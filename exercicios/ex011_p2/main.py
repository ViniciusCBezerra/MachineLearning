import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree

dados = pd.read_csv('exercicios/ex011/arquivos_usados/marketing_investimento.csv')
x = dados.drop('aderencia_investimento',axis=1)
y = dados['aderencia_investimento']
colunas = x.columns

one_hot = make_column_transformer(
    (
        OneHotEncoder(drop='if_binary'),
        ['escolaridade','estado_civil','fez_emprestimo','inadimplencia']
    ),
    remainder='passthrough',
    sparse_threshold=0
)
x = one_hot.fit_transform(x)
colunas_transformadas = one_hot.get_feature_names_out(colunas)
df = pd.DataFrame(x,columns=colunas_transformadas)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TRAIN & TEST

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

novas_colunas = [
    'casado (a)',
    'divorciado (a)',
    'solteiro (a)',
    'fundamental',
    'medio',
    'superior',
    'inadimplencia',
    'fez_emprestimo',
    'idade',
    'saldo',
    'tempo_ult_contato',
    'numero_contatos'
]

plot_tree(arvore,filled=True,class_names=['nao','sim'],fontsize=7,feature_names=novas_colunas)

normalizacao = MinMaxScaler()
x_treino_normalizado = normalizacao.fit_transform(x_treino)
pd.DataFrame(x_treino_normalizado)
