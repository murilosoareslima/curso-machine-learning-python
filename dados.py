#importando dados de uma planilha csv que demonstra acessos em determinadas paginas de um site que vende cursos
#chamaremos de X os dados que importaremos e Y os que queremos descobrir, prever.
import csv

import pandas as pd
from collections import Counter

#parametros serão [acessou_home, acessou_como_funciona, acessou_contato, comprou]
#0 para entrou na pagina e 1 para nao entrou na pagina

#funcao que ira acessar a planilha csv e ira, a cada linha, associar os valores para os arrays
#os : no final indica que algo sera feito
def carregar_acessos():
    #dados (quais paginas entrou)
    X = []
    #marcacoes (se comprou ou nao)
    Y = []
    
    #estamos abrindo o arquivo especificado e o segundo parametro, indica que vamos ler ele (read)
    arquivo = open('teste.csv', 'r')
    leitor = csv.reader(arquivo)
    #pulando a primeira linha que é do cabeçalho
    next(leitor)
    #os : no final indica que algo sera feito
    for home, como_funciona, contato, comprou  in leitor:
        #adicionando aos dados as colunas referentes a dados
        dado = [int(home), int(como_funciona), int(contato)];
        X.append(dado)
        #adicionando as marcacoes as colunas referentes a marcacoes, que é o comprou, pos comprou ou nao, servir de marcacao, para o ML aprender que determinado comportamento
        #entrando em determinadas paginas, deu o resultado de comprar ou nao
        Y.append(int(comprou))
    return X, Y
    
def carregar_buscas():
    X = []
    Y = []
    teste_dados = []
    teste_marcacoes = []
    treino_dados = []
    treino_marcacoes = []
    #df é sigla para data frame, que é o formato dos dados que o Panda devolve
    #df = pd.read_csv('buscas.csv')
    df = pd.read_csv('buscas2.csv')
    X_df = df[['home', 'busca', 'logado']]
    Y_df = df['comprou']
    Xdummies_df = pd.get_dummies(X_df)
    Ydummies_df = Y_df
    X = Xdummies_df.values
    Y = Ydummies_df.values
    
    #quantidade a ser treinada
    porcentagem_treino = 0.8
    porcentagem_teste = 0.1
    tamanho_de_treino = int(porcentagem_treino * len(Y))
    tamanho_de_teste = int(porcentagem_teste * len(Y))
    tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste
    #pegando os dados das 800 primeiras linhas
    #	0	até	799
    treino_dados = X[:tamanho_de_treino]
    treino_marcacoes = Y[:tamanho_de_treino]
    
    fim_de_treino = tamanho_de_treino + tamanho_de_teste
    #	800	até	899  
    teste_dados = X[tamanho_de_treino:fim_de_treino]
    teste_marcacoes = Y[tamanho_de_treino:fim_de_treino]
    
    validacao_dados = X[fim_de_treino:]
    validacao_marcacoes = Y[fim_de_treino:]
        
    return teste_dados, teste_marcacoes, treino_dados, treino_marcacoes, validacao_dados, validacao_marcacoes, X, Y
    
    