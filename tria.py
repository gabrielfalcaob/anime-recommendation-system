import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

print("--- [1] INICIALIZANDO O PIPELINE DO SISTEMA ---")

# 1. Carregar os dados
try:
    anime_df = pd.read_csv('anime.csv')
    rating_df = pd.read_csv('rating.csv')
    print("✓ Arquivos 'anime.csv' e 'rating.csv' carregados com sucesso!")
except FileNotFoundError as e:
    print(f"❌ Erro: Certifique-se de que os arquivos CSV estão na mesma pasta do script.\n{e}")
    exit()

# 2. Limpeza inicial de ruídos (Removendo animes assistidos mas não avaliados: nota -1)
rating_df = rating_df[rating_df['rating'] != -1]


print("\n--- [2] TRATAMENTO DE DADOS E MATRIZ ESPARSA ---")

# 1. Filtro de Usuários: Mantemos apenas usuários minimamente ativos (>= 20 avaliações válidas)
# Isso reduz o ruído de contas fantasmas, mas não estrangula a base de dados.
filtro_usuarios = rating_df['user_id'].value_counts() >= 20
usuarios_ativos = filtro_usuarios[filtro_usuarios].index
rating_filtrado = rating_df[rating_df['user_id'].isin(usuarios_ativos)]

# 2. Filtro de Animes: Reduzimos a exigência para 10 avaliações válidas.
# Como animes longos (One Piece/Shippuuden) sofrem com muitas notas -1, esse ajuste 
# permite que eles sobrevivam ao corte se tiverem pelo menos um punhado de notas reais.
filtro_animes = rating_filtrado['anime_id'].value_counts() >= 10
animes_populares = filtro_animes[filtro_animes].index
rating_final = rating_filtrado[rating_filtrado['anime_id'].isin(animes_populares)]

# Remover duplicatas críticas para o pivot funcionar sem erro
rating_final = rating_final.drop_duplicates(subset=['user_id', 'anime_id'])

print(f"✓ Base recalculada para {len(rating_final)} avaliações.")

# Construção da tabela Pivot
pivot_matriz = rating_final.pivot(index='anime_id', columns='user_id', values='rating').fillna(0)

# Conversão para Matriz Esparsa CSR
matriz_esparsa = csr_matrix(pivot_matriz.values)
print("✓ Matriz esparsa Usuário-Item gerada com sucesso.")


print("\n--- [3] TREINAMENTO DA CLUSTERIZAÇÃO (K-MEANS) ---")

# Configuração do K=10 conforme definido na metodologia do relatório
K = 10
modelo_kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42, n_init=10)

# Predição e acoplamento dos clusters de volta ao DataFrame pivotado
pivot_matriz['cluster'] = modelo_kmeans.fit_predict(matriz_esparsa)
print(f"✓ Catálogo segmentado com sucesso em {K} clusters (nichos comportamentais).")


print("\n--- [4] MOTOR DE RECOMENDAÇÃO (FILTRAGEM COLABORATIVA HÍBRIDA) ---")

def recomendar_animes_interativo():
    """
    Interface iterativa que busca correspondências, apresenta um menu baseado 
    na popularidade das obras e executa o cálculo híbrido após a confirmação.
    """
    print("\n" + "="*50)
    print("SISTEMA DE RECOMENDAÇÃO DE ANIMES INTERATIVO")
    print("="*50)
    
    while True:
        print("\nDigite o nome (ou parte do nome) do anime (ou 'sair'):")
        termo_busca = input(">> ").strip()
        
        if termo_busca.lower() == 'sair':
            print("Encerrando o sistema. Até logo!")
            break
        if not termo_busca:
            continue
            
        # 1. Buscar correspondências textuais na base de metadados [cite: 69, 70]
        matches = anime_df[anime_df['name'].str.contains(termo_busca, case=False, na=False)].copy()
        
        if matches.empty:
            print(f"❌ Nenhum título encontrado com o termo '{termo_busca}'. Tente novamente.")
            continue
            
        # 2. Ordenar os candidatos por POPULARIDADE (coluna 'members') para exibição [cite: 71, 72]
        matches_selecao = matches.sort_values(by='members', ascending=False).head(5)
        
        print(f"\nEncontramos estes títulos relacionados a '{termo_busca}':")
        opcoes = {}
        for i, (idx, row) in enumerate(matches_selecao.iterrows(), start=1):
            opcoes[i] = (row['anime_id'], row['name'])
            print(f" [{i}] {row['name']} (Membros: {int(row['members'])})")
            
        # 3. Capturar a confirmação do usuário
        try:
            escolha = input("\nSelecione o número do anime correto (or '0' para cancelar): ").strip()
            num_escolha = int(escolha)
            
            if num_escolha == 0:
                print("Busca cancelada.")
                continue
            if num_escolha not in opcoes:
                print("❌ Opção inválida.")
                continue
        except ValueError:
            print("❌ Por favor, digite um número válido.")
            continue
            
        # Resgatar o ID e Nome confirmados [cite: 69, 70]
        anime_id_escolhido, nome_confirmado = opcoes[num_escolha]
        
        # 4. Verificar se a obra específica resistiu aos filtros da matriz esparsa
        if anime_id_escolhido not in pivot_matriz.index:
            print(f"\n💡 [Nota de Engenharia de Dados]: '{nome_confirmado}' é altamente popular,")
            print("   mas grande parte de suas entradas no arquivo original rating.csv possuem valor nulo (-1).")
            print("   Como o pipeline purga esses dados omissos para preservar o cálculo de cosseno,")
            print("   o título foi desconsiderado por falta de notas reais de 1 a 10.")
            continue
        
        # 5. Descobrir o cluster do item escolhido [cite: 26, 27]
        cluster_do_item = pivot_matriz.loc[anime_id_escolhido, 'cluster']
        print(f"\n🚀 Processando... '{nome_confirmado}' está no Cluster: {cluster_do_item}")
        
        # 6. Filtrar o espaço vetorial para o cluster correspondente [cite: 55, 56]
        animes_no_mesmo_cluster = pivot_matriz[pivot_matriz['cluster'] == cluster_do_item]
        dados_votos_cluster = animes_no_mesmo_cluster.drop(columns=['cluster'])
        
        # 7. Similaridade de Cossenos [cite: 58, 61]
        matriz_similaridade = cosine_similarity(dados_votos_cluster)
        df_similaridade = pd.DataFrame(matriz_similaridade, index=dados_votos_cluster.index, columns=dados_votos_cluster.index)
        
        # Gerar o vetor ordenado excluindo a si mesmo
        vetor_similaridade = df_similaridade[anime_id_escolhido].drop(anime_id_escolhido).sort_values(ascending=False)
        top_ids = vetor_similaridade.head(5).index
        
        # 8. Estruturar saída de resultados
        resultados = []
        for idx in top_ids:
            nome_rec = anime_df[anime_df['anime_id'] == idx]['name'].values[0]
            score = vetor_similaridade[idx]
            resultados.append({"Anime Recomendado": nome_rec, "Similaridade (Cosseno)": round(score, 4)})
            
        print("\n🎯 RECOMENDAÇÕES ENCONTRADAS:")
        print(pd.DataFrame(resultados))
        print("-" * 50)

if __name__ == "__main__":
    recomendar_animes_interativo()