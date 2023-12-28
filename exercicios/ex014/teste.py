import pandas as pd

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
