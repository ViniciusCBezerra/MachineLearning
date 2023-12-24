import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier ,plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

dados = pd.read_csv('exercicios/ex013/marketing_investimento.csv')
x = dados.drop('aderencia_investimento',axis=1)
y = dados['aderencia_investimento']
colunas = x.columns 

one_hot = make_column_transformer(
    (
        OneHotEncoder(drop='if_binary'),
        ['fez_emprestimo','estado_civil','inadimplencia','escolaridade']
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

# ALGORITMO - DUMMY
#dummy = DummyClassifier()
#dummy.fit(x_treino,y_treino)


# ALGORITMO - TREE(ARVORE)
arvore = DecisionTreeClassifier(
    max_depth=3,
    random_state=5
)
arvore.fit(x_treino,y_treino)
nova_coluna = [
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
plot_tree(arvore,filled=True,class_names=['nao','sim'],fontsize=5,feature_names=nova_coluna)

# ALGORITMO - KNN
#normalizacao = MinMaxScaler()
#x_treino_normalizado = normalizacao.fit_transform(x_treino)
#x_teste_normalizado = normalizacao.fit_transform(x_teste)
#knn = KNeighborsClassifier()
#knn.fit(x_treino,y_treino)
#print(knn.score(x_teste,y_teste))

with open('modelo_one_hot.pkl','wb') as arquivo:
    pickle.dump(one_hot,arquivo)

with open('modelo_arvore.pkl','wb') as arquivo:
    pickle.dump(arvore,arquivo)    

novo_dado = {
    'idade': [45],
    'estado_civil':['solteiro (a)'],
    'escolaridade':['superior'],
    'inadimplencia': ['nao'],
    'saldo': [23040],
    'fez_emprestimo': ['nao'],
    'tempo_ult_contato': [800],
    'numero_contatos': [4]
}
novo_dado = pd.DataFrame(novo_dado)

modelo_one_hot = pd.read_pickle('modelo_one_hot.pkl')
novo_dado = modelo_one_hot.transform(novo_dado)

modelo_arvore = pd.read_pickle('modelo_arvore.pkl')
print(modelo_arvore.predict(novo_dado))
