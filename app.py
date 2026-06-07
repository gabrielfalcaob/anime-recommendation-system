import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página web
st.set_page_config(page_title="IA - Sistema de Recomendação", page_icon="🎬", layout="centered")

st.title("🎬 Sistema de Recomendação Híbrido")
st.markdown("---")

# O Streamlit usa esse decorador para salvar o modelo na memória. 
# Assim, o K-Means treina apenas UMA vez quando a página abre e não trava a navegação.
@st.cache_data
def inicializar_ia():
    # 1. Carregamento dos dados
    anime_df = pd.read_csv('anime.csv')
    rating_df = pd.read_csv('rating.csv')
    rating_df = rating_df[rating_df['rating'] != -1]
    
    # 2. Pipeline de Tratamento e Matriz Esparsa
    filtro_usuarios = rating_df['user_id'].value_counts() >= 50
    usuarios_ativos = filtro_usuarios[filtro_usuarios].index
    rating_filtrado = rating_df[rating_df['user_id'].isin(usuarios_ativos)]

    filtro_animes = rating_filtrado['anime_id'].value_counts() >= 100
    animes_populares = filtro_animes[filtro_animes].index
    rating_final = rating_filtrado[rating_filtrado['anime_id'].isin(animes_populares)].drop_duplicates(subset=['user_id', 'anime_id'])

    pivot_matriz = rating_final.pivot(index='anime_id', columns='user_id', values='rating').fillna(0)
    matriz_esparsa = csr_matrix(pivot_matriz.values)
    
    # 3. Treinamento do K-Means
    modelo_kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42, n_init=10)
    pivot_matriz['cluster'] = modelo_kmeans.fit_predict(matriz_esparsa)
    
    return anime_df, pivot_matriz

# Mensagem visual de carregamento na interface
with st.spinner("Treinando os centroides do K-Means e estruturando matrizes..."):
    anime_df, pivot_matriz = inicializar_ia()
st.success("Sistema pronto para operação!")

# --- INTERFACE GRÁFICA DE BUSCA ---
st.subheader("Pesquisar Obra de Referência")
termo_busca = st.text_input("Digite o nome (ou parte do nome) de um anime:", placeholder="Ex: Death Note, Naruto, Bleach...")

if termo_busca:
    # 1. Varredura textual
    matches = anime_df[anime_df['name'].str.contains(termo_busca, case=False, na=False)].copy()
    
    if matches.empty:
        st.error(f"Nenhum título encontrado com o termo '{termo_busca}'.")
    else:
        # Ordenar os candidatos por popularidade para facilitar a escolha
        matches_selecao = matches.sort_values(by='members', ascending=False).head(5)
        
        # Criar uma lista amigável para o componente de seleção da tela (Selectbox)
        lista_opcoes = []
        mapeamento = {}
        for _, row in matches_selecao.iterrows():
            label = f"{row['name']} ({int(row['members'])} membros)"
            lista_opcoes.append(label)
            mapeamento[label] = row['anime_id']
            
        st.write("Selecione a versão exata abaixo:")
        opcao_escolhida = st.selectbox("Títulos relacionados encontrados:", lista_opcoes)
        
        if st.button("Gerar Recomendações 🚀"):
            anime_id_escolhido = mapeamento[opcao_escolhida]
            nome_confirmado = anime_df[anime_df['anime_id'] == anime_id_escolhido]['name'].values[0]
            
            # Validação de dados consistentes
            if anime_id_escolhido not in pivot_matriz.index:
                st.warning(f"O título '{nome_confirmado}' não possui avaliações válidas suficientes na matriz.")
            else:
                # Recuperar o cluster correspondente
                cluster_do_item = pivot_matriz.loc[anime_id_escolhido, 'cluster']
                st.info(f"**Obra Identificada:** {nome_confirmado} | **Alocada no Cluster:** {cluster_do_item}")
                
                # Filtrar espaço vetorial e rodar Cosseno
                animes_no_mesmo_cluster = pivot_matriz[pivot_matriz['cluster'] == cluster_do_item]
                dados_votos_cluster = animes_no_mesmo_cluster.drop(columns=['cluster'])
                
                matriz_similaridade = cosine_similarity(dados_votos_cluster)
                df_similaridade = pd.DataFrame(matriz_similaridade, index=dados_votos_cluster.index, columns=dados_votos_cluster.index)
                
                vetor_similaridade = df_similaridade[anime_id_escolhido].drop(anime_id_escolhido).sort_values(ascending=False)
                top_ids = vetor_similaridade.head(5).index
                
                # Gerar tabela visual de resultados
                resultados = []
                for idx in top_ids:
                    nome_rec = anime_df[anime_df['anime_id'] == idx]['name'].values[0]
                    score = vetor_similaridade[idx]
                    resultados.append({"Anime Recomendado": nome_rec, "Similaridade (Cosseno)": round(score, 4)})
                    
                st.subheader("🎯 Sugestões de Afinidade Vetorial:")
                st.dataframe(pd.DataFrame(resultados), use_container_width=True)