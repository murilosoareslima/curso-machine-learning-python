#criando classificacao de textos, separando cada palavra em um dicionario para que possamos reconhecer as palavras, quantas vezes
#elas foram ditas em uma frase, e tentarmos prever de qual assunto ou para qual departamento deveriamos encaminhar a mensagem.
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import numpy as np
from collections import Counter

def vetorizar_texto(texto, tradutor):    
    #vetor vai ser do tamanho do array que guarda as palavras distintas e ele vai guardar a quantidade de vezes em que cada palavra aparece em uma frase
    vetor = [0] * len(tradutor)    
    
    for palavra in texto:
        if palavra in tradutor:
           #pegando a posicao da palavra no mapa
           posicao = tradutor[palavra]
           #somando 1 no vetor, exatamente na posicao que a palavra é encontrada no mapa tradutor
           #e somando 1 para marcar a quantidade de ocorrencia daquela palavra, no vetor
           vetor[posicao] += 1
       
    return vetor
    
#tive que colocar o encoding senao nao lia a planilha
classificacoes = pd.read_csv('emails.csv', encoding='latin-1')

textosPuros = classificacoes['email']
textosQuebrados = textosPuros.str.lower().str.split(' ')

#set sendo utilizado no lugar de [] pois nao queremos dados repetidos no nosso dicionario, o set faz esse controle.
dicionario = set()

#atualizando o dicionario conforme novas palavras aparecem, e deixando tudo em um array só.
for lista in textosQuebrados:
    dicionario.update(lista)

totalDePalavras = len(dicionario)  

#o zip serve para conseguirmos definir um array com 2 valores em cada posicao, definindo assim que valor vai na esquerda (a palavra) e 
#qual vai na direita (o numero de 0 ao tamanho do array)
#para imprimir a lista que retorna do zip, precisamos passar o zip para o list()  
print(list(zip(dicionario, range(totalDePalavras))))

tuplas = zip(dicionario, range(totalDePalavras))

#transformando tuplas em dicionario (mapa) para podermos buscar o numero que representa uma determinada palavra, apenas passando a palavra como parametro para o mapa. (chave e valor)
#isso porque o zip nao funciona como chave valor, entao direto em nao conseguiriamos passar um valor (chave) para ter o valor Ex: tuplas['se']
tradutor = {palavra:indice for palavra, indice in tuplas}
    
#transformando em um array o resultado da vetorizacao de cada frase que a lista de textosQuebrados possui    
vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]
    
print(vetoresDeTexto)

marcas = classificacoes['classificacao']

#sao os dados que foram transformados em numeros
X = vetoresDeTexto
#sao as marcacoes que vieram na planilha
Y = marcas

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]

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

resultados = {}

modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0, dual=False))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0, dual=False))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

modeloMultinomialNB = MultinomialNB()
resultadoMultinomialNB = fit_and_predict("MultinomialNB", modeloMultinomialNB, treino_dados, treino_marcacoes)
resultados[resultadoMultinomialNB] = modeloMultinomialNB

#passando o random_state = 0 para o AdaBoostClassifier que é o valor do seed, que sempre aparece em algoritmos que se baseiam
#em valores aleatórios, porém isso pode prejudicar, caso não seja atribuido o valor 0 para esse parametro, o algoritmo utilizará
#valores aleatórios, fazendo que com os mesmos dados, possamos ter resultados diferentes.
modeloAdaBoost = AdaBoostClassifier(random_state = 0)
resultadomodeloAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadomodeloAdaBoost] = modeloAdaBoost

#pegando o maior resultado dentre os testados (maior chave do dicionario). Como o dicionario guarda o resultado do fit_and_predict
#que é a taxa de acerto, ele fica sendo a chave do dicionario, e o valor fica sendo o modelo, que possui o nome do algoritmo
maximo = max(resultados)
vencedor = resultados[maximo]

#para o teste real, como sera utilizado o predict, primeiramente o modelo precisa ter sido fit, senao, nao irá rodar o algoritmo
vencedor.fit(treino_dados, treino_marcacoes)

#aqui pegamos o algoritmo vencedor e o utilizamos para fazer o teste real com os 20% dos dados nao utilizados no treino.
teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa	de	acerto	base:	%f"	%	taxa_de_acerto_base)

total_de_elementos = len(validacao_marcacoes)
print("Total de teste:	%d"	%	total_de_elementos)
