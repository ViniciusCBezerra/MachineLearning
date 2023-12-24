import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree

df = pd.read_csv('exercicios/ex011/arquivos_usados/marketing_investimento.csv')
x = df.drop('aderencia_investimento',axis=1)
y = df['aderencia_investimento']
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
transformed_df = pd.DataFrame(x,columns=colunas_transformadas)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TRAIN & TEST

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=5
)
tree.fit(x_train,y_train)

new_columns = [
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

plot_tree(tree,filled=True,class_names=['no','yes'],fontsize=7,feature_names=new_columns)
