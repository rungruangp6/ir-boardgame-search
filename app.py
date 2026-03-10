import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, utils
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Boardgame Search Engine", page_icon="🎲", layout="wide")

@st.cache_resource
def load_models():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data():
    df = pd.read_excel('bgg_top500_all.xlsx').fillna(0)
    df.columns = df.columns.str.strip().str.lower()
    rename_dict = {'min_players': 'minplayers', 'max_players': 'maxplayers', 'playing_time': 'playingtime'}
    df = df.rename(columns=rename_dict)
    df['content'] = df['name'].astype(str) + " " + df['description'].astype(str) + " " + df['categories'].astype(str)
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
st.markdown("AI-powered Hybrid Search (Support Thai & English)")

query = st.text_input("🔍 Search for board games", placeholder="e.g., War game, Farming, ซอมบี้, เกมการ์ด...")

if query:
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(query)
        expanded_query = f"{query} {translated}"
    except:
        expanded_query = query

    game_names = df['name'].tolist()
    suggestion = process.extractOne(query, game_names, processor=utils.default_process)
    if suggestion and 75 < suggestion[1] < 95:
        st.info(f"💡 Did you mean: **{suggestion[0]}**?")

    with st.spinner('Ranking results...'):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['content'])
        query_vec = tfidf.transform([expanded_query])
        lexical_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        query_embedding = model.encode(expanded_query, convert_to_tensor=True)
        semantic_scores = util.cos_sim(query_embedding, embeddings).cpu().numpy().flatten()
        
        df['final_score'] = (lexical_scores * 0.4) + (semantic_scores * 0.6)
        mask = (df['minplayers'] <= num_players) & (df['maxplayers'] >= num_players) & (df['playingtime'] <= max_time)
        
        all_results = df[mask].sort_values(by='final_score', ascending=False).head(15)

    if not all_results.empty:
        main_col, side_col = st.columns([8.5, 1.5])

        with main_col:
            st.success(f"✅ ผลลัพธ์การค้นหาสำหรับ: '{query}'")
            for i, row in all_results.head(10).iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.image(row.get('thumbnail', "https://via.placeholder.com/150"))
                        st.caption(f"Match: {int(row['final_score'] * 100)}%")
                    with col2:
                        st.subheader(row['name'])
                        st.write(f"👥 {int(row['minplayers'])}-{int(row['maxplayers'])} คน | ⏳ {int(row['playingtime'])} นาที")
                        with st.expander("รายละเอียด"):
                            st.write(row['description'])
                    st.markdown("---")

        with side_col:
            st.markdown("##### ✨ อาจถูกใจ")
            for i, row in all_results.iloc[10:15].iterrows():
                st.image(row.get('thumbnail', "https://via.placeholder.com/150"), use_container_width=True)
                st.caption(f"**{row['name']}**")
                st.divider()
    else:
        st.warning("ไม่พบเกมที่ตรงตามเงื่อนไข กรุณาลองปรับฟิลเตอร์ใหม่")