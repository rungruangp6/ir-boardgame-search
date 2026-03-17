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

# --- CONFIG ---
st.set_page_config(page_title="Boardgame IR System", page_icon="🎲", layout="wide")

# --- MODEL & DATA LOADING ---
@st.cache_resource
def load_models():
    # ใช้โมเดลขนาดเล็กที่เหมาะกับ Resource บน Streamlit Cloud
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_master_data():
    try:
        # พยายามโหลดไฟล์ Excel จาก Root Directory
        df = pd.read_excel('bgg_top500_all.xlsx').fillna("")
        df.columns = df.columns.str.strip().str.lower()
        rename_dict = {
            'min_players': 'minplayers', 
            'max_players': 'maxplayers', 
            'playing_time': 'playingtime'
        }
        df = df.rename(columns=rename_dict)
        return df
    except Exception as e:
        st.error(f"❌ Error Loading Excel: {e}")
        return pd.DataFrame()

model = load_models()
master_df = load_master_data()

# --- LIVE CRAWLER FUNCTION ---
def get_live_bgg_data():
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9,th;q=0.8',
        'Referer': 'https://www.google.com/'
    }
    
    live_list = []
    status = {"success": False, "count": 0, "error": ""}
    
    try:
        url = "https://boardgamegeek.com/browse/boardgame"
        resp = scraper.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            rows = soup.find_all("tr", id=lambda x: x and x.startswith('row_'))
            
            for row in rows[:50]:
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
        else:
            status["error"] = f"BGG Status Code: {resp.status_code}"
            
    except Exception as e:
        status["error"] = str(e)
        
    return live_list, status

# --- SESSION STATE ---
if 'live_data' not in st.session_state:
    with st.spinner('🔄 Connecting to BoardGameGeek...'):
        data, stat = get_live_bgg_data()
        st.session_state.live_data = data
        st.session_state.sync_status = stat

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("📊 Control Panel")
    
    # การแจ้งสถานะแบบ Fallback (ไม่ใช้ st.error เพื่อความสวยงาม)
    if st.session_state.sync_status["success"]:
        st.success(f"✅ Live Sync Active ({st.session_state.sync_status['count']} games)")
    else:
        st.info("🌐 Mode: Local Database (Offline)")
        st.caption("Note: BoardGameGeek is currently limiting requests. Using verified dataset.")
        if st.button("🔄 Retry Sync"):
            del st.session_state.live_data
            st.rerun()
            
    st.divider()
    num_players = st.number_input("👤 Number of Players", min_value=1, max_value=12, value=4)
    max_time = st.slider("⏳ Max Time (Minutes)", 15, 480, 120, 15)
    
    st.divider()
    st.markdown("### 🔗 Project Source")
    st.link_button("View on GitHub", "https://github.com/rungruangp6/ir-boardgame-search")

# --- MAIN UI ---
st.title("🎲 Boardgame IR Engine")
st.markdown("Intelligence Search powered by **TF-IDF** & **SBERT Semantic Analysis**")

query = st.text_input(
    "🔍 Search", 
    placeholder="เช่น 'Dungeon crawler miniatures' หรือ 'เกมที่เล่นกับเพื่อน แข่งกันทำแต้ม'..."
)

if query:
    is_thai = bool(re.search('[ก-๙]', query))
    
    # 1. NLP & Translation
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

    # 2. Boolean Filtering
    if not master_df.empty:
        mask = (master_df['minplayers'] <= num_players) & \
               (master_df['maxplayers'] >= num_players) & \
               (master_df['playingtime'] <= max_time)
        filtered_df = master_df[mask].copy()

        # 3. Hybrid Scoring
        if not filtered_df.empty:
            with st.spinner('🚀 Analyzing results...'):
                filtered_df['content'] = filtered_df['name'] + " " + filtered_df['description']
                
                # TF-IDF Score (Lexical)
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(filtered_df['content'])
                lex_scores = cosine_similarity(tfidf.transform([translated]), tfidf_matrix).flatten()

                # SBERT Score (Semantic)
                query_emb = model.encode(translated, convert_to_tensor=True)
                doc_embs = model.encode(filtered_df['content'].tolist(), convert_to_tensor=True)
                sem_scores = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()

                # Final Combined Score
                filtered_df['ir_score'] = (lex_scores * 0.4) + (sem_scores * 0.6)
                
                # 4. Sorting & Display
                final_results = filtered_df.sort_values('ir_score', ascending=False).head(10)

                st.write(f"### 📋 Top 10 Recommendations")
                
                for i, row in final_results.iterrows():
                    with st.container():
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            thumb = row.get('thumbnail')
                            img = thumb if thumb and thumb != 0 else "https://via.placeholder.com/150?text=No+Image"
                            st.image(img, use_container_width=True)
                        with col2:
                            st.subheader(row['name'])
                            st.write(f"👤 {int(row['minplayers'])}-{int(row['maxplayers'])} Players | ⏳ {int(row['playingtime'])} Min")
                            st.progress(float(row['ir_score'])) # แสดงแถบความแม่นยำ
                            st.write(f"🎯 **Match Score:** {int(row['ir_score']*100)}%")
                            with st.expander("Show Description"):
                                st.write(row['description'])
                    st.divider()
        else:
            st.error(f"❌ No games found for {num_players} players within {max_time} min.")
    else:
        st.error("❌ Database not found. Please ensure 'bgg_top500_all.xlsx' is in your GitHub.")