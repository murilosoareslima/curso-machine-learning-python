import pandas as pd
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


df = pd.read_csv('situacao_do_cliente.csv')
X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
# tamanho_de_validacao = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]

resultados = {}

def fit_and_predict(nome, modeloOneVsRest, treino_dados, treino_marcacoes):
    #K vai guardar a quantidade de pedados que será dividido os nossos dados e por sua vez, a quantidade de formas diferentes que o algoritimo irá testar
    #a cada valor de k diferente, um resultado (média) é gerado, e para nao viciarmos e ficarmos caçando um que de o melhor valor, o ideal é definir o valor de k e nao altera-lo
    k = 10
    #cross_val_score é o algoritmo que testa os dados de acordo com a quantidade de possibilitades que forem passadas para ele, como parametro (k)
    scores = cross_val_score(modeloOneVsRest, treino_dados, treino_marcacoes, cv = k)    

    # funcao da biblioteca numpy, que calcula a media
    taxa_de_acerto = np.mean(scores)
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto
    
def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes
    
    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)
    
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    
    msg = "Taxa	de	acerto	do	vencedor	entre	os	dois	algoritmos	no	mundo	real:	{0}".format(taxa_de_acerto)
    print(msg)

modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0, dual=False))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)

resultados[resultadoOneVsRest] = modeloOneVsRest


modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0, dual=False))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)

resultados[resultadoOneVsOne] = modeloOneVsOne

modeloMultinomialNB = MultinomialNB()
resultadoMultinomialNB = fit_and_predict("MultinomialNB", modeloMultinomialNB, treino_dados, treino_marcacoes)

resultados[resultadoMultinomialNB] = modeloMultinomialNB

modeloAdaBoost = AdaBoostClassifier()
resultadomodeloAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)

resultados[resultadomodeloAdaBoost] = modeloAdaBoost

print(resultados)
    
#pegando o maior resultado dentre os testados (maior chave do dicionario). Como o dicionario guarda o resultado do fit_and_predict
#que é a taxa de acerto, ele fica sendo a chave do dicionario, e o valor fica sendo o modelo, que possui o nome do algoritmo
maximo = max(resultados)
vencedor = resultados[maximo]

#para o teste real, como sera utilizado o predict, primeiramente o modelo precisa ter sido fit, senao, nao irá rodar o algoritmo
vencedor.fit(treino_dados, treino_marcacoes)

teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa	de	acerto	base:	%f"	%	taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste:	%d"	%	total_de_elementos)



