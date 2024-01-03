import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
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
df_one_hot = pd.DataFrame(x,columns=colunas_transformadas)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


arvore = DecisionTreeClassifier(max_depth=3)
arvore.fit(x,y)

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

plot_tree(arvore,filled=True,class_names=['não','sim'],fontsize=4,feature_names=nome_colunas)


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
colunas_novo_dado = novo_dado.columns
novo_dado = modelo_one_hot.transform(novo_dado)
#df_novo_dado = pd.DataFrame(novo_dado,columns=modelo_one_hot.get_feature_names_out(colunas_novo_dado))


modelo_arvore = pd.read_pickle('modelo_arvore.pkl')
print(modelo_arvore.predict(novo_dado))