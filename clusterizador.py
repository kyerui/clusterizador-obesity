import pandas as pd
from sklearn.cluster import KMeans #Clusterizador
import matplotlib.pyplot as plt #Para gráficos
import math #Matemática
from scipy.spatial.distance import cdist #para calcular as distâncias e distorções
import numpy as np #Para procedimentos numéricos
from sklearn.preprocessing import  MinMaxScaler #Classe normalizadora
from pickle import dump

pd.set_option('display.max_columns', None)

obesity = pd.read_csv('dados/ObesityDataSet_raw_and_data_sinthetic.csv', sep = ',')

dados_categoricos = obesity[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']]
dados_categoricos_normalizados = pd.get_dummies(dados_categoricos).astype(int)
dados_numericos = obesity.drop(columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'])

# Salvar as colunas categoricas
colunas_categoricas = dados_categoricos_normalizados.columns
dump(colunas_categoricas, open("./dados/colunas_categoricas.pkl", "wb"))



# Normalizar os dados

normalizador = MinMaxScaler()
neo_normalizador = normalizador.fit(dados_numericos)

dump(neo_normalizador, open('./dados/normalizador.pkl','wb'))

dados_numericos_normalizados = normalizador.fit_transform(dados_numericos)
dados_numericos_normalizados = pd.DataFrame(dados_numericos_normalizados, columns=dados_numericos.columns)
dados_finais = pd.concat([dados_numericos_normalizados, dados_categoricos_normalizados], axis=1)

distortions = []
K = range(1, 200)
dados_finais.columns = dados_finais.columns.astype(str)

for i in K:
    obesity_kmeans_model = KMeans(n_clusters = i).fit(dados_finais)
    distortions.append(sum(np.min(cdist(dados_finais,obesity_kmeans_model.cluster_centers_,'euclidean'), axis=1) /dados_finais.shape[0]))


fig, ax = plt.subplots()
ax.plot(K, distortions)
ax.set(xlabel = 'n Clusters', ylabel = 'Distorção', title = 'Elbow pela distorção')
ax.grid()
fig.savefig('./dados/elbow_distorcao.png')
plt.show()

# Calcular o número ótimo de clusters
x0 = K[0]
y0 = distortions[0]
xn = K[len(K) - 1]
yn = distortions[len(distortions)-1]
# Iterar nos pontos gerados durante os treinamentos preliminares
distancias = []
for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)

# Maior distância
n_clusters_otimo = K[distancias.index(np.max(distancias))]

telescope_kmeans_model = KMeans(n_clusters = n_clusters_otimo, random_state=42).fit(dados_finais)

dump(telescope_kmeans_model, open("./dados/obesity_cluster.pkl", "wb"))