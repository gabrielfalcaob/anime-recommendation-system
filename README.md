# Sistema Híbrido de Recomendação de Animes 🎬

Este projeto consiste em um **Sistema de Recomendação de Itens de Entretenimento** desenvolvido como trabalho prático para a disciplina de Inteligência Artificial.

A solução adota uma abordagem arquitetural híbrida e não supervisionada, combinando **Clusterização Espacial (K-Means)** para segmentação do catálogo em nichos comportamentais e **Filtragem Colaborativa Baseada em Itens (Similaridade de Cossenos)** para o ranqueamento refinado de afinidade dentro de cada grupo.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Python 3.12+**
- **Pandas**: Limpeza e estruturação matricial dos dados brutos.
- **NumPy**: Suporte a operações matemáticas vetoriais.
- **Scipy**: Armazenamento e otimização da Matriz Esparsa Usuário-Item (CSR).
- **Scikit-Learn**: Execução do algoritmo K-Means e cálculo da Similaridade de Cossenos.

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

Certifique-se de fazer o download do dataset público do Kaggle (**Anime Recommendations Database - MyAnimeList**) e extrair os arquivos `anime.csv` e `rating.csv` na raiz da pasta deste projeto.

### 2. Configurar o Ambiente Virtual (Linux)

Para isolar as dependências e executar o pipeline com segurança, configure o ambiente virtual:

```bash
# Criar o ambiente virtual
python3 -m venv venv

# Ativar o ambiente
source venv/bin/activate

# Instalar os pacotes necessários
pip install pandas scikit-learn scipy
```

### 3. Rodar a Interface interativa

Com o ambiente ativo e os arquivos CSV na pasta, execute o motor de recomendação:

```bash
# Executar
python3 tria.py
```
