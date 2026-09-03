import re
import nltk
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Descargar recursos requeridos de NLTK en segundo plano
nltk.download("punkt", quiet=True)

# 1. Configuración de página
st.set_page_config(
    page_title="Motor de Búsqueda TF-IDF", page_icon="🔍", layout="wide"
)

# Estilos CSS personalizados para la parte gráfica
st.markdown(
    """
    <style>
    .highlight-title {
        color: #1e3a8a;
        font-size: 28px;
        font-weight: 800;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 5px;
        margin-bottom: 20px;
    }
    .custom-card {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1 class='highlight-title'>⚡ Demo TF-IDF en Español</h1>", unsafe_allow_html=True)

# Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

stemmer = SnowballStemmer("spanish")


def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]


# Inicialización de la pregunta en la sesión
if "question" not in st.session_state:
    st.session_state.question = "¿Dónde juegan el perro y el gato?"

# --- BARRA LATERAL: Preguntas Sugeridas ---
with st.sidebar:
    st.markdown("### 💡 Preguntas Sugeridas")
    st.write("Haz clic en una opción para cargar la pregunta:")

    if st.button(
        "🐶 ¿Dónde juegan el perro y el gato?", use_container_width=True
    ):
        st.session_state.question = "¿Dónde juegan el perro y el gato?"
        st.rerun()

    if st.button(
        "🏃 ¿Qué hacen los niños en el parque?", use_container_width=True
    ):
        st.session_state.question = "¿Qué hacen los niños en el parque?"
        st.rerun()

    if st.button("🐦 ¿Cuándo cantan los pájaros?", use_container_width=True):
        st.session_state.question = "¿Cuándo cantan los pájaros?"
        st.rerun()

    if st.button("🎵 ¿Dónde suena la música alta?", use_container_width=True):
        st.session_state.question = "¿Dónde suena la música alta?"
        st.rerun()

    if st.button(
        "🌙 ¿Qué animal maúlla durante la noche?", use_container_width=True
    ):
        st.session_state.question = "¿Qué animal maúlla durante la noche?"
        st.rerun()

# --- ESTRUCTURA PRINCIPAL: Organización por Pestañas (Tabs) ---
tab1, tab2 = st.tabs(
    ["📝 Entrada de Datos", "📊 Analizar y Resultados"]
)

with tab1:
    col_a, col_b = st.columns([1, 1], gap="medium")

    with col_a:
        st.markdown("#### 📄 Base de Conocimiento (Documentos)")
        text_input = st.text_area(
            "Un documento por línea:", default_docs, height=220
        )

    with col_b:
        st.markdown("#### ❓ Pregunta a Evaluar")
        question_input = st.text_input(
            "Tu consulta:", value=st.session_state.question
        )
        st.info(
            "👉 Puedes cambiar la pregunta escribiendo arriba o seleccionando una propuesta desde la barra lateral."
        )

with tab2:
    st.markdown("### 🚀 Procesamiento y Similitud")
    btn_analizar = st.button("🔥 Ejecutar Análisis", type="primary")

    if btn_analizar or st.session_state.get("analizado", False):
        st.session_state.analizado = True
        documents = [d.strip() for d in text_input.split("\n") if d.strip()]

        if len(documents) < 1:
            st.error("⚠️ Ingresa al menos un documento.")
        elif not question_input.strip():
            st.error("⚠️ Escribe una pregunta válida.")
        else:
            # Procesamiento TF-IDF
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem, min_df=1
            )
            X = vectorizer.fit_transform(documents)

            question_vec = vectorizer.transform([question_input])
            similarities = cosine_similarity(question_vec, X).flatten()

            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            # Muestra de Resultados con Tarjetas
            st.write("---")

            col_res1, col_res2 = st.columns([2, 1])

            with col_res1:
                st.markdown("#### 🎯 Respuesta Relevante Encontrada")
                if best_score > 0.01:
                    st.success(f"🏆 **Respuesta:** {best_doc}")
                else:
                    st.warning(
                        f"⚠️ **Respuesta (Baja Confianza):** {best_doc}"
                    )

            with col_res2:
                st.markdown("#### 📈 Métrica de Similitud")
                st.metric(
                    label="Puntuación Coseno",
                    value=f"{best_score:.3f}",
                    delta="Coincidencia Alta"
                    if best_score > 0.3
                    else "Coincidencia Baja",
                )

            # Matriz TF-IDF Desplegable
            st.write("---")
            with st.expander("📋 **Ver Matriz TF-IDF de los Documentos**"):
                df_tfidf = pd.DataFrame(
                    X.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=[f"Doc {i+1}" for i in range(len(documents))],
                )
                st.dataframe(
                    df_tfidf.round(3), use_container_width=True
                )
