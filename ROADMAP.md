
# ROADMAP — Visualizações de Algoritmos de IA

Este documento lista futuras implementações de visualizações animadas de algoritmos clássicos de Inteligência Artificial.
O objetivo é expandir gradualmente a coleção de GIFs educativos gerados automaticamente pelo projeto.

---

# Prioridade Alta

## Q-Learning (GridWorld)
Visualização de aprendizado por reforço onde um agente aprende a navegar em um grid.

Elementos visuais:
- agente explorando o ambiente
- política emergindo
- caminho ótimo sendo descoberto

Métricas:
- episódio
- reward médio
- epsilon
- convergência da política

---

## DBSCAN
Clusterização baseada em densidade.

Visualização:
- expansão de clusters
- identificação de pontos de ruído
- raio epsilon mostrado graficamente

---

## PageRank
Algoritmo clássico de ranking usado pelo Google.

Visualização:
- grafo de páginas
- fluxo de importância entre nós
- tamanho dos nós proporcional ao rank

---

## Gaussian Mixture Model (EM)
Clusterização probabilística via Expectation-Maximization.

Visualização:
- gaussianas elípticas
- centros movendo
- covariâncias mudando

---

## Simulated Annealing
Otimização global inspirada em processos termodinâmicos.

Visualização:
- ponto explorando superfície de custo
- grandes saltos no início
- refinamento conforme temperatura diminui

---

# Prioridade Média

## Particle Swarm Optimization
Partículas se movendo em busca de um ótimo global.

Visualização:
- movimento coletivo
- melhores posições locais e globais

---

## Monte Carlo Tree Search (MCTS)
Busca em árvore usada em jogos como Go.

Visualização:
- expansão progressiva da árvore
- caminhos mais promissores destacados

---

## Kalman Filter
Filtro de estimação de estado.

Visualização:
- trajetória real
- medições ruidosas
- estimativa filtrada
- elipse de covariância

---

## Ant Colony Optimization
Metaheurística inspirada em colônias de formigas.

Visualização:
- formigas explorando caminhos
- trilhas de feromônio se fortalecendo

---

# Prioridade Baixa

## LLE (Locally Linear Embedding)
Redução de dimensionalidade baseada em reconstrução local.

---

## Isomap
Redução de dimensionalidade baseada em distâncias geodésicas.

---

## Boltzmann Machine / RBM
Modelo energético estocástico relacionado às Hopfield Networks.

---

## Hidden Markov Model (Viterbi)
Inferência de sequência mais provável.

Visualização:
- treliça temporal
- caminho ótimo sendo descoberto

---

# Observação

Novas visualizações devem seguir o mesmo padrão visual do projeto:

- layout consistente
- painel de métricas à esquerda
- visualização principal à direita
- animação progressiva do algoritmo
- geração automática de GIFs via scripts Python
