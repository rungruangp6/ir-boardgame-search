import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Boardgame Search", page_icon="🎲", layout="wide")

@st.cache_resource
def load_models():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data():
    df = pd.read_excel('bgg_top500_all.xlsx').fillna(0)
    df.columns = df.columns.str.strip().str.lower()
    
    rename_dict = {
        'min_players': 'minplayers',
        'max_players': 'maxplayers',
        'playing_time': 'playingtime'
    }
    df = df.rename(columns=rename_dict)
    
    df['content'] = (
        df['name'].astype(str) + " " + 
        df['description'].astype(str) + " " + 
        df['categories'].astype(str)
    )
    return df

model = load_models()
df = load_data()

@st.cache_data
def get_embeddings(_df_content):
    return model.encode(_df_content.tolist(), convert_to_tensor=True)

embeddings = get_embeddings(df['content'])

st.sidebar.title("🎯 Filters")
num_players = st.sidebar.number_input("Players", min_value=1, max_value=10, value=2)
max_time = st.sidebar.slider("Max Playing Time (min)", min_value=15, max_value=240, value=90, step=15)

st.title("🎲 Boardgame Search")

query = st.text_input("🔍 Search for games:")

if query:
    with st.spinner('Searching...'):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['content'])
        query_vec = tfidf.transform([query])
        lexical_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        query_embedding = model.encode(query, convert_to_tensor=True)
        semantic_scores = util.cos_sim(query_embedding, embeddings).cpu().numpy().flatten()

        df['final_score'] = (lexical_scores * 0.3) + (semantic_scores * 0.7)
        
        mask = (df['minplayers'] <= num_players) & \
               (df['maxplayers'] >= num_players) & \
               (df['playingtime'] <= max_time)
        
        filtered_df = df[mask].copy()
        results = filtered_df.sort_values(by='final_score', ascending=False).head(10)

    if not results.empty:
        for i, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    img_url = row.get('thumbnail', "https://via.placeholder.com/150")
                    st.image(img_url)
                    st.metric("Match", f"{int(row['final_score']*100)}%")
                with col2:
                    st.subheader(row['name'])
                    st.write(f"👥 Players: {int(row['minplayers'])}-{int(row['maxplayers'])} | ⏳ Time: {int(row['playingtime'])} min")
                    st.write(f"🎭 Categories: {row['categories']}")
                    with st.expander("More Details"):
                        st.write(row['description'])
                st.markdown("---")
    else:
        st.warning("No games found matching your criteria. Please try adjusting the filters.")