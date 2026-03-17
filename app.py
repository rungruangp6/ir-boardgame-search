import streamlit as st
import pandas as pd
import numpy as np
import cloudscraper
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

st.set_page_config(page_title="Boardgame IR System", page_icon="🎲", layout="wide")

@st.cache_resource
def load_models():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_master_data():
    df = pd.read_excel('bgg_top500_all.xlsx').fillna("")
    df.columns = df.columns.str.strip().str.lower()
    rename_dict = {
        'min_players': 'minplayers', 
        'max_players': 'maxplayers', 
        'playing_time': 'playingtime'
    }
    df = df.rename(columns=rename_dict)
    return df

model = load_models()
master_df = load_master_data()

def get_live_bgg_data():
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    live_list = []
    status = {"success": False, "count": 0, "error": ""}
    try:
        url = "https://boardgamegeek.com/browse/boardgame"
        resp = scraper.get(url, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            rows = soup.find_all("tr", id=lambda x: x and x.startswith('row_'))
            for row in rows:
                obj_cell = row.find("td", class_="collection_objectname")
                if obj_cell:
                    name_tag = obj_cell.find("a", class_="primary")
                    if name_tag:
                        clean_name = name_tag.get_text().strip().split(' (')[0]
                        all_text = obj_cell.get_text(separator="|").split("|")
                        potential_desc = [
                            t.strip() for t in all_text 
                            if len(t.strip()) > 15 and clean_name not in t
                        ]
                        live_desc = potential_desc[0] if potential_desc else ""
                        live_list.append({"name": clean_name, "desc": live_desc})
            status["success"] = True
            status["count"] = len(live_list)
    except Exception as e:
        status["error"] = str(e)
    return live_list, status

if 'live_data' not in st.session_state:
    with st.spinner('🔄 Syncing...'):
        data, stat = get_live_bgg_data()
        st.session_state.live_data = data
        st.session_state.sync_status = stat

st.sidebar.title("📊 Control Panel")

if st.session_state.sync_status["success"]:
    st.sidebar.success("✅ Sync Successful")
else:
    st.sidebar.error("❌ Sync Failed")

num_players = st.sidebar.number_input("👤 Number of Players", min_value=1, max_value=10, value=4)
max_time = st.sidebar.slider("⏳ Max Time (Minutes)", 15, 240, 120, 15)

st.title("🎲 Boardgame Smart IR Engine")

query = st.text_input(
    "🔍 Search for board games (Supports Thai & English)", 
    placeholder="e.g. 'Build networks' or 'เกมวางแผนอวกาศ'..."
)

if query:
    is_thai = bool(re.search('[ก-๙]', query))
    if is_thai:
        th_stops = list(thai_stopwords()) + ['เกม', 'เล่น', 'อยาก', 'ที่มี', 'แบบ', 'เอา']
        tokens = word_tokenize(query, engine='newmm')
        clean_query_th = " ".join([w for w in tokens if w not in th_stops and w.strip() != ""])
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(clean_query_th)
        except:
            translated = query
    else:
        translated = query

    mask = (master_df['minplayers'] <= num_players) & \
           (master_df['maxplayers'] >= num_players) & \
           (master_df['playingtime'] <= max_time)
    filtered_df = master_df[mask].copy()

    matched_names = [
        item['name'] for item in st.session_state.live_data 
        if translated.lower() in item['desc'].lower() or query.lower() in item['desc'].lower()
    ]

    if not filtered_df.empty:
        with st.spinner('🚀 Searching...'):
            filtered_df['content'] = filtered_df['name'] + " " + filtered_df['description']
            
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(filtered_df['content'])
            query_vec = tfidf.transform([translated])
            lex_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

            query_emb = model.encode(translated, convert_to_tensor=True)
            doc_embs = model.encode(filtered_df['content'].tolist(), convert_to_tensor=True)
            sem_scores = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()

            filtered_df['lex_score'] = lex_scores
            filtered_df['sem_score'] = sem_scores
            filtered_df['ir_score'] = (lex_scores * 0.4) + (sem_scores * 0.6)

            primary = filtered_df[filtered_df['name'].isin(matched_names)].copy()
            if not primary.empty:
                primary['final_rank'] = primary['ir_score'] + 1.0 
            
            fallback = filtered_df[~filtered_df['name'].isin(matched_names)].copy()
            fallback['final_rank'] = fallback['ir_score']

            final_results = pd.concat([primary, fallback]).sort_values('final_rank', ascending=False).head(15)

            st.write(f"### 📋 Recommendations for {num_players} Players")
            
            for i, row in final_results.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        img_url = row.get('thumbnail') if row.get('thumbnail') != 0 else "https://via.placeholder.com/150"
                        st.image(img_url)
                    with col2:
                        st.subheader(row['name'])
                        st.write(f"👥 {int(row['minplayers'])}-{int(row['maxplayers'])} Players | ⏳ {int(row['playingtime'])} Minutes")
                        with st.expander("Read description"):
                            st.write(row['description'])
                        
                        match_percent = int(row['ir_score'] * 100)
                        st.write(f"📊 **Match Score:** {match_percent}%")
                st.divider()
    else:
        st.error(f"❌ No games found for {num_players} players.")