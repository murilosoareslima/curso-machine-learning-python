from sklearn.naive_bayes import MultinomialNB

#parametros a serem definidos para cada array[e gordinho?, tem perna curta?, late?]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

#1 pra porco e -1 pra cachorro
marcacoes = [1, 1, 1, -1, -1, -1]

#pra nao dar o erro Expected 2D array, got 1D array instead: usamos o [[]] entre os valores ou colocamos 
# os misteriosos dentro de um outro array, como foi feito no teste = 
misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

teste = [misterioso1, misterioso2, misterioso3]

#estou definindo aqui o que eu sei que é cada um dos misteriosos. primeiro e terceiro cachorro, e o segundo porco.
marcacoes_teste = [-1, 1, -1]
#responsavel pelo treinamento
modelo = MultinomialNB()

#este metodo faz a funcao de se adequar, de aprender os dados e as regras
modelo.fit(dados, marcacoes)

#pedindo para o modelo prever o que vem a ser o misterioso, usando o metodo predict
resultado = modelo.predict(teste)

#subtraindo os 2 arrays, para mostrar o percentual de acerto
#que, como diz no livro, nem sempre vai ser 100% mas temos que ter um percentual aceitável de acerto.
diferencas = resultado - marcacoes_teste


#pegando em quantidade, os acertos, dentro de um array de acertos
acertos = [d for d in diferencas if d == 0]

#pegando o tamanho do array contendo os acertos
total_acertos = len(acertos)

#agora pegamos o tamanho de elementos testados
total_elementos = len(teste)

#pegamos a porcentagem de acerto
taxa_acertos = 100.0 * total_acertos / total_elementos

print(resultado)
print(diferencas)
print(taxa_acertos)