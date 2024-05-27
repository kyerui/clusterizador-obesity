import pandas as pd
from pickle import load
import numpy as np
import warnings
warnings.filterwarnings('ignore')

obesity_clusters_kmeans = load(open('./dados/obesity_cluster.pkl', "rb"))
normalizador = load(open("./dados/normalizador.pkl", "rb"))
colunas_categoricas = load(open("./dados/colunas_categoricas.pkl", "rb"))
dados_categoricos_colunas = pd.DataFrame(columns=colunas_categoricas)
colunas = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']

nova_instancia = ['Female', 21, 1.62, 64, 'yes', 'no', 2, 3, 'Sometimes', 'no', 2, 'no', 0, 1, 'no', 'Public_Transportation', 'Normal_Weight']
nova_instancia_ds = pd.DataFrame([nova_instancia],  columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'])

dados_categoricos = nova_instancia_ds[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']]
dados_numericos = nova_instancia_ds.drop(columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'])

dados_numericos_normalizados = normalizador.transform(dados_numericos)
dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'])

dados_categoricos_normalizados = pd.get_dummies(data=dados_categoricos, prefix_sep='_', dtype=int)

dados_completos = pd.concat([dados_categoricos_colunas, dados_categoricos_normalizados], axis=0)
dados_completos = dados_completos.where(pd.notna(dados_completos), other=0)
dados_completos = dados_numericos_normalizados.join(dados_completos, how='left')

# EXECUÇÃO DO PREDICT
predict = obesity_clusters_kmeans.predict(dados_completos)
centroide = obesity_clusters_kmeans.cluster_centers_[predict]

centroide_list = np.array(centroide)
centroide_num = centroide_list[0][:8]
dados_numericos_centroide = pd.DataFrame([centroide_num], columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'])
centroide_cat = centroide_list[0][8:]

dados_norm_legiveis_num = normalizador.inverse_transform(dados_numericos_centroide)
dados_numericos_centroide = pd.DataFrame(dados_norm_legiveis_num, columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'])

dados_numericos_centroide_rounded = dados_numericos_centroide.round({'Age': 0, 'Height' : 4, 'Weight': 1, 'FCVC': 0, 'NCP': 0, 'CH2O': 0, 'FAF': 0, 'TUE': 0})
dados_numericos_centroide_rounded = dados_numericos_centroide_rounded.applymap(lambda x: int(x) if round(x) == x else x)

dados_categoricos_centroide = pd.DataFrame(columns=colunas_categoricas)
dados_categoricos_centroide.loc[0] = np.round(centroide_cat)
dados_categoricos_centroide = pd.from_dummies(dados_categoricos_centroide, sep='_')


dados_finais = pd.concat([dados_numericos_centroide_rounded, dados_categoricos_centroide], axis=1).reindex(columns=colunas)



print(f"\nÍndice do grupo da nova instância: {predict[0]}")
print(f"\nCentroide da nova instância: \n{centroide}")
print(f"\nDados legiveis da centroid: \n{dados_finais}")
