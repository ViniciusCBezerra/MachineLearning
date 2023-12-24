import pandas as pd
import plotly.express as px

def criar_grafico(dados,coluna,segunda_coluna=0):   
    if verificar_tipo(dados,coluna) != str:
        if verificar_se_segcoluna_nao_eh_zero(segunda_coluna):
            return px.box(dados,x=coluna,color=segunda_coluna)
        else:
            return px.box(dados,x=coluna)
    else:
        if verificar_se_segcoluna_nao_eh_zero(segunda_coluna):
            return px.histogram(dados,x=coluna,text_auto=True,barmode='group',color=segunda_coluna)
        else:
            return px.histogram(dados,x=coluna,text_auto=True,barmode='group')

def verificar_tipo(dados,coluna):
    tipo = type(dados[coluna][0])
    return tipo

def verificar_se_segcoluna_nao_eh_zero(segunda_coluna):
    if segunda_coluna != 0:
        return True
    else:
        return False
    
dados = pd.read_csv('exercicios/ex001/arquivos_usados/marketing_investimento.csv')
criar_grafico(dados,'escolaridade','fez_emprestimo')