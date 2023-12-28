import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import pickle

dados = pd.read_csv('exercicios/ex015/arquivos_usados/marketing_investimento.csv')
x = dados.drop('aderencia_investimento',axis=1)
y = dados['aderencia_investimento']
colunas = x.columns 

one_hot = make_column_transformer(
    (
        OneHotEncoder(drop='if_binary'),
        ['fez_emprestimo','inadimplencia','escolaridade','estado_civil']
    ),
    remainder='passthrough',
    sparse_threshold=0
)
x = one_hot.fit_transform(x)
colunas_transformadas = one_hot.get_feature_names_out(colunas)
df = pd.DataFrame(x,columns=colunas_transformadas)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TREINO E TESTE

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
nome_colunas = [
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

plot_tree(arvore,filled=True,class_names=['nao','sim'],fontsize=5,feature_names=nome_colunas)

with open('modelo_one_hot.pkl','wb') as arquivo:
    pickle.dump(one_hot,arquivo)

with open('modelo_arvore.pkl','wb') as arquivo:
    pickle.dump(arvore,arquivo)

modelo_one_hot = pd.read_pickle('modelo_one_hot.pkl')

novo_dado = {
    'idade': [45],
    'estado_civil':['solteiro (a)'],
    'escolaridade':['superior'],
    'inadimplencia': ['sim'],
    'saldo': [100],
    'fez_emprestimo': ['nao'],
    'tempo_ult_contato': [1],
    'numero_contatos': [4]
}
novo_dado = pd.DataFrame(novo_dado)
colunas = novo_dado.columns

novo_dado = modelo_one_hot.transform(novo_dado)
#novo_dado = pd.DataFrame(novo_dado,columns=modelo_one_hot.get_feature_names_out(colunas))

modelo_arvore = pd.read_pickle('modelo_arvore.pkl')
modelo_arvore.predict(novo_dado)
if modelo_arvore.predict(novo_dado) == 1:
    aderiu = 'sim'
elif modelo_arvore.predict(novo_dado) == 0:
    aderiu = 'nao'
else:
    aderiu = 'erro'

print(f'Aderiu: {aderiu}')