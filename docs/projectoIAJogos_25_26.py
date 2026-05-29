# %% [markdown]
# # Inteligência Artificial Em Jogos 25/26
# ## Projecto - Aplicação do NEAT ao jogo Racing Cars
# 
# <img src="figs/rn.png" alt="Drawing" style="width: 600px;"/>
# %% [markdown]
# ## 1. Introdução
# Em aulas de laboratório anteriores apresentámos o jogo Racing Cars em `pygame` em que podíamos controlar os carros de corrida manualmente e em que apresentámos carros autónomos com navegação entre waypoints e também carros controlados através de árvores de decisão na secção de exercícios.
# 
# No caso dos carros controlados por árvores de decisão, utilizámos os 4 vértices do carro como sensores que indicavam se o carro estava dentro ou fora da pista, neste caso na borda ou na relva. Estes 4 sensores são muito limitados porque só funcionam quando os carros estão efectivamente já a sair da pista e não se adaptam bem ao padrão do circuito, não atencipando o formato da estarda. Os carros são também controlados através de acções discretas tais como rotação à direita e à esquerda e a alteração da magnitude da velocidade. No fundo, essas acções discretas alteram a velocidade angular e também a sua magnitude. Existe contudo limtações em termos da velocidade máxima a que o carro se pode mover e também a máxima alteração tanto em termos angulares como em termos de magnitude.
# %% [markdown]
# ## 2. Objectivo
# 
# O objectivo deste projecto é utilizar redes neuronais para controlar os carros de corrida e que os controladores sejam evoluídos através de neuro-evolução, i.e a aplicação de evolução artificial para a geração das redes neuronais. A evolução de redes neuronais pode ser de dois tipo: 
# * de topologia fixa em que o designer decide qual a topologia da rede e a evolução dá-se apenas ao nível dos pesos das sinapses e
# * de topologia variável em que a própria arquitectura das redes é evoluida juntamente com os pesos das ligações entre os neurónios.
# 
# Este segundo tipo de neuro-evolução é mais desafiante e por isso escolhida para este projecto.
# 
# Um dos algoritmos mais famosos e de maior sucesso na neuro-evolução é o NEAT (NeuroEvolution of Augmenting Topologies) que evolui simultaneamente tanto os pesos como a estrutura (topologia) das redes neuronais, pertencendo à classe dos TWEANNS (Topology and Weight Evolving Artificial Neural Networks). 
# 
# <img src="figs/fully.png" alt="Drawing" style="width: 300px;"/>
# 
# Este algoritmo começa por uma arquitectura mínima, apenas formada pelos neurónios de entrada e de saída, completamente ligada, com pesos aleatórios nas ligações e aumenta de complexidade com o tempo tendo sido usada com sucesso em problemas de robótica, de controle e do mundo dos jogos.
# 
# Podem consultar o capítulo inicial do livro *Hands-on Neuroevolution with python* de Iaroslav Omelianenko que está na pasta com este enunciado  que poderão ler como introdução ao NEAT. Uma das livrarias do python que implementa o NEAT é a NEAT-python que também é introduzida no livro juntamente com o PyTorch-NEAT e o MultiNEAT. Podem escolher qualquer delas para ser usada neste projecto.
# %% [markdown]
# ## 3. Especificações
# Este projecto é aberto a alguma variação e experimentação mas há certos requisitos que são exigidos. 
# * Terão que utilizar o jogo Racing Cars que usaram nos laboratórios e não qualquer outro jogo semelhante.
# * Queremos que façam duas versões de controladores: uma que utiliza os wayspoints e uma que utiliza radares.
# %% [markdown]
# ### 3.1 WayPoints
# Como se lembram do laboratório onde foi introduzido o Racing Car, o carro verde autónomo depende para a sua condução de uma sequência de pontos marcados ao longo da pista, os waypoints, normalmente na zona das curvas. Esses pontos podem ser marcados manualmente fazendo uso do rato sendo guardados numa lista.
# 
# O programa deve permitir a introdução pelo utilizador da sequência de waypoints e a consequente evolução.
# 
# #### Inputs
# Desejamos neste projecto que o controlador do carro verde passe a ser uma rede neuronal que terá como inputs informação relativa ao próximo waypoint, ou ao waypoint corrente, a qual pode ser o ângulo entre o centro do carro e o waypoint corrente e a distância, por exemplo. Podem também ter como input o tipo de curva, se no sentido horário ou no sentido anti-horário ou o ângulo entre o próximo waypoint e o waypoint seguinte ou ambos os casos, sendo este último mais aconselhado porque não depende do utilizador.
# 
# #### Outputs
# Os outputs da navegação serão sempre a aceleração em termos angulares e em termos de magnitude tendo sempre em atenção os seus limites. A decisão quanto à mudança de waypoint pode também opcionalmente ser inclu+ida como output do controlador, e nesse caso, é a rede neurinal que decide quando é que é a leitura de mudar o próximo waypoint.  
# %% [markdown]
# ### 3.2 Radares (sensores)
# <img src="figs/radarsCar.png" alt="Drawing" style="width: 200px;"/>
# Em claro contraste com os carros usados nas aulas de laboratório os quais possuem sensores em posições fixas do carro e que detectam $3$ tipos de materiais, i.e., píxeis: estrada, borda e relva grass, os carros devem possuir um conjunto de radares de colocados num número e disposição livres, i.e. que são deixados ao vosso critério. Na figura em cima temos um carro com $5$ radares dispostos de forma simétrica com um alcance limitado e que medem a distância à borda da estrada. Esses radares poderão não ter uma distância máxima de alcance mas certamente que a existência de um alcance máximo pode tornar mais desafiante a navegação autónoma quanto mais esquena for essa distância máxima.
# 
# #### Inputs
# Notem que os valores das distâncias medidas pelos radares serão valores "float" em claro contraste com os valores das 3 cores dos carros que usámos nos sensores de tamanho fixo na aulas de Laboratório. Esses valores dos sonares serão os inputs das redes neuronais usadas como controladore dos carros.
# 
# #### Outputs ou acções dos carros
# Os outputs da navegação serão sempre a aceleração em termos angulares e em termos de magnitude tendo sempre em atenção os seus limites.
# %% [markdown]
# ### 3.3 Ruído
# Devem introduzir algum ruído na medição dos sensores e também nos valores dos actuadores porque o mundo físico é um mundo de incerteza, sem uma precisão abosluta tanto nos inputs como nos outputs.
# %% [markdown]
# ### 3.4 Corridas diferentes
# Ao evoluirem os carros poderão fazer experiências em que a linha de meta é colocada em diferentes pontos do circuito ou mantendo-se no mesmo lugar, o sentido do circutio é invertido, os carros percorrem-no no sentido horário em vez de ser no sentido anti-horário que é o modo Lab.
# %% [markdown]
# ## 4 A Função de Fitness
# Para qualquer das versões seria interessante experimentarem diferentes funções de fitness para avaliarem o desempenho dos carros. A função de fitness pode ser avaliada num só circuito numa só corrida ou como desempenho médio de uma uma série de corridas em que se varia a posição inicial do carro para a mesma linha de meta, o sentido horário ou anti-horário da corrida ou mesmo se experimentam diferentes poições para as linhas de meta.
# 
# Será relevante distinguir conduções que sejam capazes de atacar melhor as curvas por dentro e não por fora, reflectindo-se no valor de desempenho.
# É importante penalizar os carros que saiam da pista e que façam corta mato e que não façam batota porque uma boa condução implica que o carro se mantenha dentro da pista sempre e cumprar todo o circuito.
# %% [markdown]
# ## 5 Avaliação e Comparação da Performance da Evolução
# Será interessante avaliar e comparar as performances dos dois tipos de carros evoluídos considerando as variações de parâmetros que possam utilizar.
# Durante a evolução, o NEAT-python fornece informação que pode ser apresentada graficamente sobre a evolução do fitness médio e máximo ao longo de gerações. Será também interessante conhecer a topologia e pesos das melhores redes neuronais, que pode ser visualizada graficamente.
# 
# Convém que se indiquem quais os parâmetros fixos do carro que foram utilizados e quais os valores dos parâmetros NEAT do ficheiro de configuração e se foram experimentadas variações nesses parâmetros ou os parâmetros standard.
# %% [markdown]
# ## 6. Extras
# Poderão ir para além do que foi pedido através por exemplo de uma modelização mais complexa da física do movimento dos carros
# %% [markdown]
# ## 7. Recursos
# Poderão usar todo o código fornecido durante as aulas de laboratório e o código associado ao livro de referência para este trabalho, livro esse que se encontra numa pasta em conjunto com o código.
# %% [markdown]
# ## 8. Relatório
# Juntamente com o código é preciso entregar um relatório que deve:
# * Identificar os elementos do grupo pelos seus nomes e números.   
# * Descrever sumariamente o trabalho realizado
# * Descrever os inputs utilizados para os controladores, os radares e outra informação caso exista
# * Descrever as experiências feitas e todos os parâmetros usados bem como os a performance dos controladores ao longo das evoluções e também a performance comparativa entre as duas versões.
# %% [markdown]
# ## 9. Submissão
# O zip com o código e relatório deve ser submetido no link específico para esse efeito na página da disciplina no Moodle. O ficheiro zip deve ser nomeado de acordo com o seguinte formato: `25_26_IIAJogosProjecto_XX` onde XX identifica o nome do grupo ou ids dos alunos.
# %% [markdown]
# ## 10. Limite do Prazo de Entrega
# O ficheiro zip tem de ser submetido até 4 de Junho, 1m antes da meia noite.
# Na sexta, 5 de Junho os projectos serão apresentados, grupo a grupo, e mais tarde iremos disponibilzar no Moodle uma ligação para a marcação de "slots" para essa apresentação.