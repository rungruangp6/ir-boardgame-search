import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Boardgame Search", page_icon="🎲", layout="wide")

# --- 1. Load Data & Models ---
@st.cache_resource
def load_models():
    # Model AI ที่เข้าใจความหมายไทย-อังกฤษ (Semantic Search)
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data():
    # โหลดไฟล์ Excel และแทนที่ค่าว่างด้วย 0
    df = pd.read_excel('bgg_top500_all.xlsx').fillna(0)
    
    # ล้างชื่อคอลัมน์: ตัดช่องว่าง และทำให้เป็นตัวพิมพ์เล็กทั้งหมด
    df.columns = df.columns.str.strip().str.lower()
    
    # แก้ปัญหาชื่อคอลัมน์ไม่ตรง (จาก min_players เป็น minplayers)
    rename_dict = {
        'min_players': 'minplayers',
        'max_players': 'maxplayers',
        'playing_time': 'playingtime'
    }
    df = df.rename(columns=rename_dict)
    
    # มัดรวมข้อมูลสำหรับทำ Search Index
    df['content'] = (
        df['name'].astype(str) + " " + 
        df['description'].astype(str) + " " + 
        df['categories'].astype(str)
    )
    return df

model = load_models()
df = load_data()

# คำนวณ Vector ความหมาย (Embeddings) เก็บไว้ใน Cache
@st.cache_data
def get_embeddings(_df_content):
    return model.encode(_df_content.tolist(), convert_to_tensor=True)

embeddings = get_embeddings(df['content'])

# --- 2. Sidebar Filters (ตัวกรองด้านข้าง) ---
st.sidebar.title("🎯 ตัวกรอง (Filters)")

# รับค่าจากผู้ใช้เพื่อนำไปกรองข้อมูล
num_players = st.sidebar.number_input("จำนวนผู้เล่น", min_value=1, max_value=10, value=2)
max_time = st.sidebar.slider("ระยะเวลาเล่นสูงสุด (นาที)", min_value=15, max_value=240, value=90, step=15)

# --- 3. Main UI ---
st.title("🎲 Boardgame Search")

query = st.text_input("🔍 ค้นหาเกมที่ต้องการ (เช่น 'เกมสัตว์', 'วางแผนเศรษฐกิจ', 'zombie'):")

if query:
    with st.spinner('กำลังค้นหา...'):
        # --- กองหน้า 1: TF-IDF (เน้น Keyword ตรงตัว) ---
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['content'])
        query_vec = tfidf.transform([query])
        lexical_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # --- กองหน้า 2: Semantic (เน้น ความหมาย/บริบท) ---
        query_embedding = model.encode(query, convert_to_tensor=True)
        semantic_scores = util.cos_sim(query_embedding, embeddings).cpu().numpy().flatten()

        # คำนวณคะแนนรวม (Hybrid Score)
        df['final_score'] = (lexical_scores * 0.3) + (semantic_scores * 0.7)
        
        # --- การกรองข้อมูล (Filtering) ---
        # กรองเฉพาะเกมที่ตรงกับจำนวนผู้เล่นและเวลาที่เลือก
        mask = (df['minplayers'] <= num_players) & \
               (df['maxplayers'] >= num_players) & \
               (df['playingtime'] <= max_time)
        
        filtered_df = df[mask].copy()
        
        # จัดลำดับตามคะแนนความเหมือน
        results = filtered_df.sort_values(by='final_score', ascending=False).head(10)

    # --- 4. แสดงผลลัพธ์ ---
    if not results.empty:
        for i, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    # แสดงรูปภาพ (ถ้ามี)
                    img_url = row.get('thumbnail', "https://via.placeholder.com/150")
                    st.image(img_url)
                    st.metric("Match", f"{int(row['final_score']*100)}%")
                with col2:
                    st.subheader(row['name'])
                    # แสดงข้อมูลผู้เล่นและเวลา
                    st.write(f"👥 ผู้เล่น: {int(row['minplayers'])}-{int(row['maxplayers'])} คน | ⏳ เวลา: {int(row['playingtime'])} นาที")
                    st.write(f"🎭 หมวดหมู่: {row['categories']}")
                    with st.expander("รายละเอียดเพิ่มเติม"):
                        st.write(row['description'])
                st.markdown("---")
    else:
        st.warning("❌ ไม่พบเกมที่ตรงกับเงื่อนไขของคุณ ลองปรับจำนวนผู้เล่นหรือระยะเวลาเล่นดูนะครับ")