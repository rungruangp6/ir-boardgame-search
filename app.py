import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, utils

st.set_page_config(page_title="Boardgame Search Engine", page_icon="🎲", layout="wide")

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

st.sidebar.title("🎯 Search Filters")
num_players = st.sidebar.number_input("Target Players", min_value=1, max_value=10, value=2)
max_time = st.sidebar.slider("Max Time (min)", min_value=15, max_value=240, value=90, step=15)

st.title("🎲 Boardgame Smart Search")
st.markdown("Find your next game using **AI-powered Hybrid Search**")

query = st.text_input("🔍 What kind of game are you looking for?", placeholder="e.g., Bird game, War in space, เกมวางแผน...")

if query:
    game_names = df['name'].tolist()
    suggestion = process.extractOne(query, game_names, processor=utils.default_process)
    
    if suggestion and 70 < suggestion[1] < 95:
        st.info(f"💡 Did you mean: **{suggestion[0]}**?")

    with st.spinner('Analyzing query and ranking results...'):
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
        
        results = df[mask].sort_values(by='final_score', ascending=False).head(10)

    if not results.empty:
        st.success(f"Found {len(results)} games matching your criteria.")
        for i, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    img_url = row.get('thumbnail', "https://via.placeholder.com/150")
                    st.image(img_url)
                    score_pct = int(row['final_score'] * 100)
                    st.metric("Relevance", f"{score_pct}%")
                with col2:
                    st.subheader(row['name'])
                    st.write(f"👥 **Players:** {int(row['minplayers'])}-{int(row['maxplayers'])} | ⏳ **Time:** {int(row['playingtime'])} min")
                    st.write(f"🎭 **Categories:** {row['categories']}")
                    with st.expander("Read Description"):
                        st.write(row['description'])
                st.markdown("---")
    else:
        st.warning("No games found. Try adjusting your filters or search terms.")