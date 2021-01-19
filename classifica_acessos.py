from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acessos()

#para o treinamento utilizamos 90% dos dados que temos na planiha, e os 10% faltantes utilizamos como testes

#aqui estamos pegando as 90 linhas ta planilha que serao utilizadas para treinar
treino_dados = X[:90]
treino_marcacoes = Y[:90]

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

#aqui estamos pegando as ultimas 9 linhas da planilha para usarmos de teste
teste_dados = X[-9:]
teste_marcaoes = Y[-9:]

#tentando definir dados de teste, que são os não treinados
resultado = modelo.predict(teste_dados);
diferencas = resultado - teste_marcaoes

acertos = [d for d in diferencas if d == 0]

total_acertos = len(acertos)
#o total de elementos que esta sendo testado.
total_elementos = len(teste_dados)

taxa_acerto = 100.0 * total_acertos / total_elementos

print(taxa_acerto)
print(total_elementos)