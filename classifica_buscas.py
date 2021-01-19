from dados import carregar_buscas
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

from collections import Counter

teste_dados, teste_marcacoes, treino_dados, treino_marcacoes, validacao_dados, validacao_marcacoes, X, Y = carregar_buscas()

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    #dados pra treinar
    modelo.fit(treino_dados, treino_marcacoes)

    #testando se vai descobrir o resultado com esses dados que nao foram utilizados no treino
    resultado = modelo.predict(teste_dados)


    if resultado[0]=='sim' or resultado[0]=='nao':
       #dessa forma transformamos os sim e nao em true e false que é o mesmo que 0 e 1
       #se nao for dessa forma, nao tem como fazer as contas abaixo, pois sim e nao são Strings
       acertos = resultado == teste_marcacoes   
       total_de_acertos = sum(acertos)
       total_de_elementos = len(teste_dados)
    else: 
       diferencas = resultado - teste_marcacoes
       acertos = [d for d in diferencas if d == 0]
       total_de_acertos = len(acertos)
       total_de_elementos = len(teste_dados)



    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    msg = "Taxa	de	acerto	do	algoritmo complexo {0}: {1}".format(nome, taxa_de_acerto)
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
    
    
modeloMultinomialNB = MultinomialNB()

resultadoMultinomialNB = fit_and_predict("MultinomialNB", modeloMultinomialNB, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modeloAdaBoostClassifier = AdaBoostClassifier()
resultadoAdaBoostClassifier = fit_and_predict("AdaBoostClassifier", modeloAdaBoostClassifier, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoMultinomialNB > resultadoAdaBoostClassifier:
   vencedor = modeloMultinomialNB
else:
   vencedor = modeloAdaBoostClassifier
   
teste_real(vencedor, validacao_dados, validacao_marcacoes)


#--------------------------------------------MODO BURRO

#se formos pelo modo burro, somamos a qtd de respostas 1 (comprou) e somamos as respostas 0 (não comprou)
#ai usamos o de maior ocorrência como taxa de acerto
if Y[0]=='sim' or Y[0]=='nao':
   acerto_de_um = list(Y).count('sim')
   acerto_de_zero = list(Y).count('nao')
   caracteristicas_quem_nao_compra = X[Y=='nao']
else:
   acerto_de_um = list(Y).count(1)
   acerto_de_zero = list(Y).count(0)
   caracteristicas_quem_nao_compra = X[Y==0]
   
taxa_de_acerto_base_burro = max(acerto_de_um, acerto_de_zero)
taxa_de_acerto_base_burro = 100.0 * taxa_de_acerto_base_burro / len(Y)
#o %f é para indicar que queremos decimais, se fosse inteiro deveria ser %d
print("Taxa	de	acerto	base burro:	%f"	%	taxa_de_acerto_base_burro)
   
#o max retorna a maior o correncia dentre os valores encontrados e contados pelo Counter
acerto_base = max(Counter(validacao_marcacoes).values())

taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
#o %f é para indicar que queremos decimais, se fosse inteiro deveria ser %d
print("Taxa	de	acerto	base:	%f"	%	taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total	de	teste:	%d"	% total_de_elementos)

#para pegarmos as caracterisicas de quem nao comprou por exemplo, podemos fazer X[Y==0] onde X são as características e Y==0, queremos apenas 
#quem não comprou, e como Y guarda 1 para comprou e 0 para nao comprou, esse código Y==0 retorna true pra tudo que for 0 e pega em X tudo que 
#confere com Y true
print("Características de quem não compra")
print(caracteristicas_quem_nao_compra)
    


