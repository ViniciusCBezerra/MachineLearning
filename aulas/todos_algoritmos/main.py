import pandas as pd 
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.neighbors import KNeighborsClassifier


dados = pd.read_csv('aulas/todos_algoritmos/arquivos_usados/marketing_investimento.csv')
x = dados.drop('aderencia_investimento',axis=1)
y = dados['aderencia_investimento']
colunas = x.columns

one_hot = make_column_transformer(
    (
        OneHotEncoder(drop='if_binary'),
        ['fez_emprestimo','escolaridade','estado_civil','inadimplencia']
    ),
    remainder='passthrough',
    sparse_threshold=0
)
x = one_hot.fit_transform(x)
colunas_transformadas = one_hot.get_feature_names_out(colunas)

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
#print(dummy.score(x_teste,y_teste))



# ALGORITMO - TREE
#arvore = DecisionTreeClassifier(max_depth=3,random_state=5)
#arvore.fit(x_treino,y_treino)
#print(arvore.score(x_teste,y_teste))
#novas_colunas = [
#    'casado (a)',
#    'divorciado (a)',
#    'solteiro (a)',
#    'fundamental',
#    'medio',
#    'superior',
#    'inadimplencia',
#    'fez_emprestimo',
#    'idade',
#    'saldo',
#   'tempo_ult_contato',
#   'numero_contatos'
#]
#plot_tree(arvore,filled=True,class_names=['nao','sim'],fontsize=5,feature_names=novas_colunas)



# ALGORITMO - KNN
#normalizacao = MinMaxScaler()
#x_treino_normalizado = normalizacao.fit_transform(x_treino)
#x_teste_normalizado = normalizacao.fit_transform(x_teste)
#knn = KNeighborsClassifier()
#knn.fit(x_treino_normalizado,y_treino)
#print(knn.score(x_teste_normalizado,y_teste))