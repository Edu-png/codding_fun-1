"""
matcher.py

Responsabilidade:
    - Receber um CV em PDF
    - Receber uma descrição de vaga
    - Transformar ambos em representações vetoriais (TF-IDF)
    - Calcular similaridade semântica via Cosine Similarity
    - Retornar um score percentual

Arquitetura atual:
    PDF -> Texto -> Preprocessamento -> TF-IDF -> Cosine Similarity -> Score
"""

# IMPORTS DE MACHINE LEARNING

# TF-IDF transforma texto em vetor numérico ponderado por relevância
from sklearn.feature_extraction.text import TfidfVectorizer

# Função para calcular similaridade entre vetores
from sklearn.metrics.pairwise import cosine_similarity


# IMPORTS INTERNOS

# Função responsável exclusivamente por extrair texto do PDF
# Boa prática: manter parsing fora do matcher (Single Responsibility Principle)
from utils import extract_text_from_pdf


# IMPORTS DE INTERFACE (DESKTOP)

# Tkinter é usado apenas para abrir o explorador de arquivos
import tkinter as tk
from tkinter import filedialog


# IMPORTS DE NLP

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# CONFIGURAÇÃO DE NLP

# Stopwords removem palavras muito frequentes e pouco informativas
# Ex: "the", "and", "is"
stop_words = set(stopwords.words("english"))

# Lematizador reduz palavras à forma base
# Ex: "running" -> "run"
lemmatizer = WordNetLemmatizer()


# FUNÇÃO DE PREPROCESSAMENTO

def preprocess_text(text: str) -> str:
    """
    Objetivo:
        Normalizar o texto para reduzir ruído antes da vetorização.

    Etapas:
        1. Lowercase (evita tratar Python e python como diferentes)
        2. Remoção de caracteres especiais
        3. Tokenização simples (split)
        4. Remoção de stopwords
        5. Lematização

    Retorna:
        Texto limpo e padronizado
    """

    # Normalização de caixa
    text = text.lower()

    # Remove tudo que não for letra ou espaço
    # Evita pontuação interferindo na vetorização
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenização simples (em produção pode-se usar nltk.word_tokenize)
    tokens = text.split()

    # Remove palavras muito comuns
    tokens = [t for t in tokens if t not in stop_words]

    # Lematiza palavras (reduz variações morfológicas)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# FUNÇÃO DE SELEÇÃO DE PDF

def select_pdf():
    """
    Abre o explorador de arquivos do sistema operacional.

    Boa prática:
        Separar interface da lógica de negócio.

    Retorna:
        Caminho do PDF selecionado
    """

    root = tk.Tk()
    root.withdraw()  # Esconde janela principal do Tkinter
    root.attributes('-topmost', True)  # Garante que a janela fique em foco

    file_path = filedialog.askopenfilename(
        title="Selecione o CV em PDF",
        filetypes=[("PDF files", "*.pdf")]
    )

    root.destroy()
    return file_path

# FUNÇÃO PRINCIPAL DE MATCHING

def calculate_job_fit(cv_pdf_path: str, job_text: str) -> float:
    """
    Responsável por:
        - Extrair texto do CV
        - Preprocessar ambos os textos
        - Vetorizar via TF-IDF
        - Calcular similaridade angular
        - Retornar score percentual

    Estratégia:
        TF-IDF + Cosine Similarity
        -> abordagem clássica de Information Retrieval
    """

    # 1. Extração de texto
    cv_text = extract_text_from_pdf(cv_pdf_path)

    # 2. Preprocessamento
    cv_text = preprocess_text(cv_text)
    job_text = preprocess_text(job_text)

    # 3. Vetorização
    # ngram_range=(1,2):
    #   Considera unigramas e bigramas (ex: "machine learning")
    # max_features:
    #   Limita dimensionalidade (controle de memória)
    vect = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000
)

    # Gera matriz 2 x N_features
    X = vect.fit_transform([cv_text, job_text])

    # 4. Similaridade angular
    # Retorna matriz 1x1
    similarity = cosine_similarity(X[0:1], X[1:2])[0][0]

    # Converte para percentual
    return similarity * 100


# ENTRY POINT

if __name__ == "__main__":

    # Seleção do arquivo
    cv_path = select_pdf()

    if not cv_path:
        print("Nenhum arquivo selecionado!")
        exit()

    # Entrada da vaga via terminal
    print("\nCole a descrição da vaga:")
    import sys
    job_description = sys.stdin.read()

    if not job_description.strip():
        print("Descrição da vaga vazia!")
        exit()

    # Cálculo final
    score = calculate_job_fit(cv_path, job_description)

    print(f"\n🎯 Job Fit Score: {score:.2f}%")