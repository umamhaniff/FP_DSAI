# app_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import time
import re

# Import modul lokal dari folder src
# Pastikan folder src dan file-file ini ada di direktori yang sama
from src.data_loader import load_local_data
from src.preprocessing import clean_and_handle_missing_values
from src.feature_engineering import create_features
from src.modelling import build_hybrid_model, calculate_evaluation_metrics
from src.integratedRecommender import IntegratedRecommender
from src.evaluasiLlm import LLMTools, HybridEvaluation
from src.rekom import CollaborativeFilteringRecommender

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfigurasi Global ---
DATA_FILE_PATH = 'data/product_data.csv'

# --- Fungsi Bantu Kategorisasi (Regex Optimized) ---
def filter_by_category(df, category_name):
    """Filter dataframe menggunakan regex boundary untuk akurasi lebih tinggi."""
    df_filtered = pd.DataFrame()
    cat_lower = category_name.lower()
    
    keywords = {
        "skincare": r'\b(skin|facial|cleanser|moisturizer|serum|acne|eczema|face|lip care|vaseline|cetaphil)\b',
        "haircare": r'\b(hair|shampoo|conditioner|dandruff|styling|mousse|pantene|head & shoulders)\b',
        "make up": r'\b(makeup|lipstick|lip balm|foundation|blush|primer|mascara|eyeshadow|concealer|powder)\b',
        "bodycare": r'\b(body|bath|shower|lotion|cream|scrub|soap|deodorant|shaving|razor|body wash)\b'
    }

    if cat_lower in keywords:
        mask = df['Category'].str.contains(keywords[cat_lower], case=False, regex=True, na=False) | \
            df['Name'].str.contains(keywords[cat_lower], case=False, regex=True, na=False)
        df_filtered = df[mask]
    elif cat_lower == "others":
        all_keywords = "|".join(keywords.values())
        mask = ~df['Category'].str.contains(all_keywords, case=False, regex=True, na=False)
        df_filtered = df[mask]
    else:
        df_filtered = df[df['Category'].astype(str).str.contains(category_name, case=False, na=False)]
    
    return df_filtered

# --- Fungsi Bantu Logika ---
def add_to_cart(pid):
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    if pid not in st.session_state.cart:
        st.session_state.cart.append(pid)
        st.toast(f"Produk berhasil ditambahkan ke keranjang! 🛒", icon="✅")
    else:
        st.toast(f"Produk sudah ada di keranjang!", icon="⚠️")

# --- [BARU] FUNGSI HEADER & FOOTER GLOBAL ---

def render_header(show_search_controls=False, custom_title=None):
    """
    Header global yang konsisten untuk semua halaman (kecuali Home yang punya style unik).
    
    Args:
        show_search_controls (bool): Tampilkan input jumlah & tombol evaluate?
        custom_title (str): Judul pengganti search bar (misal nama kategori).
    """
    # Gunakan container dengan ID header-marker agar styling CSS 'putih' tetap jalan
    with st.container(border=True):
        st.markdown('<span id="header-marker"></span>', unsafe_allow_html=True) 
        
        # Layout kolom dinamis tergantung mode
        if show_search_controls:
            c_back, c_logo, c_mid, c_num, c_sbtn, c_ebtn = st.columns([0.6, 1.5, 3.5, 1, 0.6, 1.8], gap="small")
        else:
            # Layout simpel untuk halaman kategori
            c_back, c_logo, c_mid, c_sbtn = st.columns([0.6, 1.5, 6, 0.6], gap="small")

        # 1. Tombol Back (Selalu ada jika bukan home)
        with c_back:
            # Gunakan key unik agar tidak conflict antar halaman
            if st.button("⬅️", help="Kembali ke Home", use_container_width=True, key=f"hdr_back_{st.session_state.get('current_page', 'unk')}"):
                st.session_state["current_page"] = "home"
                # Reset pagination jika kembali dari kategori
                if "cat_page_number" in st.session_state:
                    st.session_state["cat_page_number"] = 0
                st.rerun()

        # 2. Logo Teks
        with c_logo:
            st.markdown("""
                <div style='font-size: 24px; font-weight: 900; color: #385F8C; white-space: nowrap; margin-top: 5px;'>
                    clickmart
                </div>
            """, unsafe_allow_html=True)

        # 3. Middle Section (Search Bar atau Custom Title)
        product_query = ""
        with c_mid:
            if custom_title:
                # Tampilkan Judul Kategori
                st.markdown(f"""
                    <div style='font-size: 20px; font-weight: 700; color: #FFFFFF; margin-top: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
                        {custom_title}
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Tampilkan Search Bar
                default_query = st.session_state.get("global_search_query", "")
                product_query = st.text_input(
                    "Cari Produk", 
                    value=default_query,
                    placeholder="Contoh: 'Moisturizer kulit berminyak'",
                    label_visibility="collapsed",
                    key=f"hdr_search_{st.session_state.get('current_page', 'unk')}"
                )

        # 4. Controls & Action Buttons
        top_n = 10
        run_search = False
        run_eval = False

        if show_search_controls:
            with c_num:
                top_n = st.number_input("Jml", min_value=5, max_value=50, value=10, step=5, label_visibility="collapsed", key="hdr_num")
            
            with c_sbtn:
                run_search = st.button("🔍", type="primary", use_container_width=True, help="Cari Produk", key="hdr_sbtn_main")
                
            with c_ebtn:
                has_recs = 'current_rekom' in st.session_state and st.session_state['current_rekom'] is not None
                run_eval = st.button("Evaluate AI", disabled=not has_recs, use_container_width=True, key="hdr_eval")
        
        else:
            # Di halaman kategori, tombol search icon berfungsi reset ke halaman pencarian kosong
            with c_sbtn:
                if st.button("🔍", use_container_width=True, key="hdr_sbtn_redirect"):
                    st.session_state["global_search_query"] = "" # Reset query
                    st.session_state["current_page"] = "recommender"
                    st.rerun()

    return product_query, top_n, run_search, run_eval

def render_footer():
    """Footer global yang konsisten."""
    st.markdown("---")
    st.markdown("""
        <div style='background-color: #385F8C; color: #FFFFFF; padding: 20px; text-align: center; border-radius: 10px; margin-top: 20px;'>
            <p style='margin:0; font-weight:bold;'>clickmart AI Recommender</p>
            <p style='font-size: 0.8rem; margin:0;'>Ditenagai oleh Streamlit & Google Gemini LLM</p>
            <p style='font-size: 0.8rem; margin:0;'>Developed by Kelompok 5 DSAI CAMP3</p>
        </div>
    """, unsafe_allow_html=True)

# --- Fungsi Bantu UI Lainnya ---

def display_evaluation_ui(evaluation: HybridEvaluation):
    """Menampilkan hasil evaluasi LLM."""
    score = evaluation.score
    description = evaluation.description
    reasons = evaluation.reasons
    summary = evaluation.summary

    if score >= 8: score_color = "green"
    elif score >= 5: score_color = "orange"
    else: score_color = "red"

    st.markdown("---")
    st.subheader("📊 Hybrid Recommendation Evaluation")
    st.markdown(f"**⭐ Score: :{score_color}[{score}/10]**")
    st.markdown("**Description:**")
    st.info(description)
    st.markdown("**Reasons:**")
    for r in reasons:
        st.markdown(f"- {r}")
    st.markdown("**Summary:**")
    st.markdown(f"> {summary}")
    st.markdown("---")

@st.dialog("Detail Produk Lengkap", width="large")
def show_product_popup(product_data, score=None):
    img_url = str(product_data.get('ImageURL', '')).split('|')[0]
    
    if img_url and img_url != 'nan' and pd.notna(img_url):
        st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 20px; height: 350px; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <img src="{img_url}" style="max-height: 100%; max-width: 100%; object-fit: contain;">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; border-radius: 10px; height: 200px; background-color: #f9f9f9; color: #888; margin-bottom: 20px;">
                Gambar tidak tersedia
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<h2 style='color: #385F8C; margin-top: 0;'>{product_data.get('Name', 'No Name')}</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Brand:** {product_data.get('Brand', '-')}")
        st.markdown(f"**Kategori:** {product_data.get('Category', '-')}")
    with c2:
        rating_val = product_data.get('Rating', product_data.get('average_rating', 0))
        st.markdown(f"**Rating:** ⭐ {int(rating_val)}")
        st.markdown(f"**Reviews:** {product_data.get('ReviewCount', 0)}")
        if score is not None:
             st.markdown(f"**Relevance Score:** `{score:.4f}`")

    st.markdown("<br>", unsafe_allow_html=True)

    col_cart, col_buy = st.columns(2)
    with col_cart:
        prod_id = product_data.get('ProdID', product_data.get('Name'))
        if st.button("🛒 Add to Cart", use_container_width=True, key=f"popup_cart_{prod_id}"):
            add_to_cart(prod_id)
    with col_buy:
        if st.button("🛍️ Buy Now", type="primary", use_container_width=True, key=f"popup_buy_{product_data.get('Name')}"):
            with st.spinner("Memproses pembayaran..."):
                time.sleep(1.5)
            st.balloons()
            st.success("Pembayaran Berhasil! Terima kasih telah berbelanja di clickmart.")

    st.divider()
    st.markdown("### Deskripsi")
    st.write(product_data.get('Description', 'Tidak ada deskripsi.'))
    st.divider()

    st.markdown("### Spesifikasi & Data Lengkap")
    exclude_cols = ['Name', 'Brand', 'Category', 'Rating', 'ReviewCount', 'Description', 'ImageURL', 'final_score', 'Name_norm', 'similarity', 'rating_norm', 'review_norm', 'average_rating']
    details = {k: v for k, v in product_data.items() if k not in exclude_cols and pd.notna(v) and str(v).strip() != ''}
    
    if details:
        df_details = pd.DataFrame(list(details.items()), columns=['Atribut', 'Nilai'])
        st.dataframe(df_details, hide_index=True, use_container_width=True)
    else:
        st.caption("Tidak ada informasi tambahan.")

def render_product_card(row, full_df=None, prefix=""):
    with st.container(border=True):
        img_url = str(row.get('ImageURL', '')).split('|')[0].strip() if pd.notna(row.get('ImageURL')) else None
        
        if img_url:
            image_html = f"""
                <div style="height: 150px; display: flex; justify-content: center; align-items: center; overflow: hidden; margin-bottom: 10px; background-color: #ffffff; border-radius: 5px;">
                    <img src="{img_url}" style="height: 100%; width: 100%; object-fit: contain;">
                </div>
            """
        else:
            image_html = """
                <div style="height: 150px; background-color: #f0f2f6; display: flex; justify-content: center; align-items: center; margin-bottom: 10px; color: #888; border-radius: 5px;">
                    No Image
                </div>
            """
        st.markdown(image_html, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="product-title" title="{row.get('Name', 'No Name')}">
                {row.get('Name', 'No Name')}
            </div>
        """, unsafe_allow_html=True)

        st.caption(f"{row.get('Brand', '-')}")
        
        rating_val = row.get('Rating', row.get('average_rating', 0))
        rating_int = int(rating_val)
        
        if 'final_score' in row:
            st.write(f"⭐ {rating_int} | Sc: {row['final_score']:.2f}")
        else:
            st.write(f"⭐ {rating_int}")
            
        unique_key = f"{prefix}_btn_{row.get('ProdID', row.name)}" if prefix else f"btn_{row.get('ProdID', row.name)}"
        if st.button("Lihat Selengkapnya", key=unique_key, use_container_width=True):
            if full_df is not None and row.name in full_df.index:
                full_data = full_df.loc[row.name]
            else:
                full_data = row
            score_val = row.get('final_score', None)
            show_product_popup(full_data, score=score_val)

def display_grid(products, title, full_df=None, prefix=""):
    if title:
        st.subheader(title)
    cols = st.columns(5)
    for idx, (_, product) in enumerate(products.iterrows()):
        with cols[idx % 5]:
            render_product_card(product, full_df=full_df, prefix=prefix)

@st.cache_resource
def initialize_system():
    try:
        df = load_local_data(DATA_FILE_PATH)
        if df.empty:
            return None, None, None, None, None

        df = clean_and_handle_missing_values(df)
        df, tfidf_matrix = create_features(df)
        
        hybrid_sim = build_hybrid_model(df, tfidf_matrix)
        metrics = calculate_evaluation_metrics(df, hybrid_sim)
        
        llm_tools = LLMTools()
        recommender = IntegratedRecommender(df, hybrid_sim)
        
        cf_recommender = CollaborativeFilteringRecommender(data_path=DATA_FILE_PATH, num_users=500, random_seed=42)
        
        return df, recommender, llm_tools, metrics, cf_recommender
        
    except Exception as e:
        logger.error(f"Error initalization: {e}")
        return None, None, None, None, None

# --- Halaman: Category View (Updated with Function) ---

def page_category_view(df):
    category_name = st.session_state.get("selected_category", "Others")
    
    # [1] PANGGIL HEADER (Custom Title)
    render_header(show_search_controls=False, custom_title=f"Kategori: {category_name}")

    # --- FILTERING ---
    filtered_df = filter_by_category(df, category_name)
    
    st.markdown(f"Found **{len(filtered_df)}** products in {category_name}")
    st.markdown("---")

    if filtered_df.empty:
        st.info(f"😔 Maaf, produk untuk kategori '{category_name}' belum tersedia.")
        render_footer() # Tetap tampilkan footer
        return

    # --- PAGINASI ---
    ITEMS_PER_PAGE = 10
    if "cat_page_number" not in st.session_state:
        st.session_state.cat_page_number = 0

    total_pages = (len(filtered_df) - 1) // ITEMS_PER_PAGE + 1
    
    if st.session_state.cat_page_number >= total_pages:
        st.session_state.cat_page_number = 0

    start_idx = st.session_state.cat_page_number * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    batch_df = filtered_df.iloc[start_idx:end_idx]

    # --- RENDER GRID ---
    display_grid(batch_df, "", full_df=df, prefix=f"cat_{category_name}")

    # --- NAVIGASI HALAMAN ---
    st.markdown("<br>", unsafe_allow_html=True)
    c_prev, c_info, c_next = st.columns([1, 2, 1])
    
    with c_prev:
        if st.button("Previous Page", disabled=(st.session_state.cat_page_number == 0), use_container_width=True):
            st.session_state.cat_page_number -= 1
            st.rerun()
            
    with c_info:
        st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Page {st.session_state.cat_page_number + 1} of {total_pages}</div>", unsafe_allow_html=True)
        
    with c_next:
        if st.button("Next Page", disabled=(st.session_state.cat_page_number >= total_pages - 1), use_container_width=True):
            st.session_state.cat_page_number += 1
            st.rerun()

    # [2] PANGGIL FOOTER
    render_footer()


# --- Halaman: Product Recommender (Updated with Function) ---

def page_recommender(df, recommender, llm_tools, metrics, cf_recommender):
    st.title("🛍️ Pencarian & Rekomendasi")
    
    # [1] PANGGIL HEADER (Search Controls Enabled)
    product_query, top_n, run_search, run_eval = render_header(show_search_controls=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ System Info")
        st.metric("Total SKU", len(df))
        if st.button("🔄 Reset Cache / Refresh"):
            st.cache_resource.clear()
            st.rerun()
            
        st.header("🛒 Keranjang Belanja")
        if 'cart' in st.session_state and st.session_state.cart:
            st.write(f"Items: {len(st.session_state.cart)}")
        else:
            st.write("Keranjang kosong.")
        if st.button("Kosongkan Keranjang"):
            st.session_state.cart = []
            st.success("Keranjang dikosongkan!")

    # --- LOGIKA PROSES SEARCH ---
    # Trigger dari Home atau tombol Search di Header baru
    trigger_home = st.session_state.get("global_search_query", "") and st.session_state.get("trigger_search", False)
    
    if (run_search or trigger_home) and product_query:
        st.session_state.trigger_search = False 
        st.session_state["global_search_query"] = product_query # Sync query
        
        with st.spinner(f"Mencari produk terbaik untuk '{product_query}'..."):
            interpreted = product_query
            if llm_tools:
                interpreted = llm_tools.interpret_query_with_llm(product_query)
            
            if interpreted != product_query:
                st.info(f"💡 Query diperjelas AI: **{interpreted}**")

            recs = recommender.get_recommendations(interpreted, int(top_n))

            if isinstance(recs, str) or recs.empty:
                st.error("Tidak ada produk ditemukan.")
                st.session_state['current_rekom'] = None
            else:
                st.session_state['current_rekom'] = recs
                if 'last_eval_result' in st.session_state:
                    del st.session_state['last_eval_result']
        
        st.rerun()

    # --- LOGIKA EVALUASI LLM ---
    if run_eval and 'current_rekom' in st.session_state:
        if not llm_tools:
            st.error("LLM Tools error.")
        else:
            with st.spinner("Gemini sedang menganalisis hasil..."):
                eval_res = llm_tools.evaluate_recommendation_with_llm(st.session_state['current_rekom'])
                if eval_res:
                    st.session_state['last_eval_result'] = eval_res

    # --- MAIN DISPLAY ---
    if 'last_eval_result' in st.session_state:
        with st.expander("📝 Lihat Hasil Analisis & Evaluasi AI", expanded=True):
            display_evaluation_ui(st.session_state['last_eval_result'])

    if 'current_rekom' in st.session_state and st.session_state['current_rekom'] is not None:
        recs = st.session_state['current_rekom']
        st.subheader(f"Hasil Pencarian ({len(recs)} Produk)")
        
        cols = st.columns(5)
        for idx, (index, row) in enumerate(recs.iterrows()):
            with cols[idx % 5]:
                render_product_card(row, full_df=df, prefix="search")

    # --- FOOTER SECTION (EDA) ---
    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    with st.expander("📊 Klik untuk membuka Analisis Data (EDA) & Statistik Dataset"):
        st.subheader("Exploratory Data Analysis")
        c_row1_col1, c_row1_col2 = st.columns(2)
        with c_row1_col1:
            st.markdown("<div style='text-align: center; font-weight: bold;'>Distribusi Rating Produk</div>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(df['Rating'], bins=10, kde=True, ax=ax1, color='teal')
            st.pyplot(fig1)

        with c_row1_col2:
            st.markdown("<div style='text-align: center; font-weight: bold;'>Sample Heatmap Kemiripan</div>", unsafe_allow_html=True)
            n_viz = min(10, len(df))
            sample_indices = range(n_viz)
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.heatmap(recommender.hybrid_sim[np.ix_(sample_indices, sample_indices)], cmap='YlGnBu', ax=ax3)
            st.pyplot(fig3)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; font-weight: bold;'>Top 10 Kategori Produk</div>", unsafe_allow_html=True)
        c_left, c_center, c_right = st.columns([1, 6, 1])
        with c_center:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            df['Category'].value_counts().head(10).sort_values().plot(kind='barh', color='salmon', ax=ax2)
            st.pyplot(fig2, use_container_width=True)
            
    # [2] PANGGIL FOOTER GLOBAL
    render_footer()

# --- Halaman: Home Page ---

def page_home(df, cf_recommender):
    # Header Home tetap Custom karena styling logonya beda
    with st.container(border=True):
        st.markdown('<span id="header-marker"></span>', unsafe_allow_html=True) 
        c_logo, c_search, c_btn = st.columns([1.2, 4.3, 0.5], gap="small")

        with c_logo:
            st.markdown("""
                <div style='
                    font-size: 30px; 
                    font-weight: 900; 
                    color: #FFFFFF; 
                    margin-top: -5px;
                    white-space: nowrap;
                    cursor: pointer;
                '>
                    clickmart
                </div>
            """, unsafe_allow_html=True)

        with c_search:
            search_input = st.text_input("Search", placeholder="Cari produk di clickmart...", label_visibility="collapsed")

        with c_btn:
            if st.button("🔍", key="search_home", use_container_width=True):
                if search_input:
                    st.session_state["global_search_query"] = search_input
                    st.session_state["trigger_search"] = True
                    st.session_state["current_page"] = "recommender"
                    st.rerun()

    if search_input:
        st.session_state["global_search_query"] = search_input
        st.session_state["trigger_search"] = True
        st.session_state["current_page"] = "recommender"
        st.rerun()

    # --- Categories ---
    st.markdown("### Shop with Categories")
    categories = [
        {"name": "Skincare", "image": "https://i.pinimg.com/736x/4c/16/7c/4c167c5ac422efd13eba8e07d04274a7.jpg"},
        {"name": "Bodycare", "image": "https://i.pinimg.com/736x/bf/00/df/bf00df3d3cf4271cdb625a387936f90d.jpg"},
        {"name": "Haircare", "image": "https://i.pinimg.com/736x/02/d5/7f/02d57f094a70a0b5c6c1f7279b21a2d3.jpg"},
        {"name": "Make Up", "image": "https://i.pinimg.com/1200x/e4/14/34/e414342a7464892f646fe9baeee41c51.jpg"},
        {"name": "Others", "image": "https://i.pinimg.com/736x/2d/f3/c2/2df3c287f50c35de6d65d16ff225ebda.jpg"}
    ]

    cols = st.columns(5)
    for i, (col, category) in enumerate(zip(cols, categories)):
        with col:
            with st.container():
                if st.button("", key=f"cat_btn_{i}", use_container_width=True):
                    st.session_state["selected_category"] = category["name"]
                    st.session_state["current_page"] = "category_view"
                    st.session_state["cat_page_number"] = 0 
                    st.rerun()
                
                st.markdown(f"""
                <div class="category-card-img">
                    <img src="{category['image']}" alt="{category['name']}" style="width: 100%; height: 100px; object-fit: cover;">
                    <div class="category-label">{category['name']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .category-card-img {
        border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 10px; transition: all 0.3s ease; position: relative;
    }
    .category-card-img:hover { box-shadow: 0 1px 10px rgba(255, 255, 255, 0.8); transform: translateY(-3px); }
    .category-label { background: #385F8C; color: white; padding: 10px; text-align: center; font-weight: 600; font-size: 0.9rem; }
    .stButton > button { position: absolute !important; top: 0 !important; left: 0 !important; width: 100% !important; height: 100% !important; background: transparent !important; border: none !important; color: transparent !important; z-index: 100 !important; cursor: pointer !important; }
    .stButton > button:hover { background: transparent !important; border: none !important; }
    div[data-testid="column"] button { position: absolute !important; z-index: 2 !important; opacity: 0 !important; height: 500px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Best Sellers & Featured ---
    if cf_recommender:
        most_liked = cf_recommender.get_most_liked_products(top_n=5)
        if not most_liked.empty:
            display_grid(most_liked, "🔥 Produk Terlaris", full_df=df, prefix="best")

    st.subheader("✨ Produk Unggulan Kami")
    display_df = df[df['ImageURL'].notna() & (df['ImageURL'] != '')].head(15)
    cols = st.columns(5)
    for idx, (index, row) in enumerate(display_df.iterrows()):
        with cols[idx % 5]:
            render_product_card(row, full_df=df, prefix="feat")

    if cf_recommender:
        recom_prods = cf_recommender.get_most_liked_products(top_n=15)
        if not recom_prods.empty:
            display_grid(recom_prods, "❤️ Rekomendasi Untuk Anda", full_df=df, prefix="recom")

    # [PANGGIL FOOTER GLOBAL]
    render_footer()

# --- Main Controller ---

def main():
    st.set_page_config(page_title="clickmart", layout="wide")

    # --- CSS STYLE INJECTION ---
    st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; }
        h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #385F8C !important; }
        [data-testid="stVerticalBlockBorderWrapper"] { background-color: #E9E9E9; border-color: #DDDDDD; border-radius: 10px; }
        /* Style khusus untuk container yang memiliki header-marker */
        [data-testid="stVerticalBlockBorderWrapper"]:has(#header-marker) { background-color: #385F8C !important; border: 1px solid #385F8C !important; }
        [data-testid="stVerticalBlockBorderWrapper"]:has(#header-marker) div[data-testid="stButton"] button { color: #385F8C !important; background-color: #FFFFFF !important; border: 1px solid #385F8C !important; }
        [data-testid="stVerticalBlockBorderWrapper"]:has(#header-marker) div[data-testid="stButton"] button:hover { background-color: #f0f0f0 !important; color: #2b4a70 !important; }
        /* Style button default selain header */
        [data-testid="stVerticalBlockBorderWrapper"]:not(:has(#header-marker)) div[data-testid="stButton"] button { color: #FFFFFF !important; background-color: #385F8C !important; border: none !important; }
        [data-testid="stVerticalBlockBorderWrapper"]:not(:has(#header-marker)) div[data-testid="stButton"] button:hover { background-color: #2b4a70 !important; }
        .product-title { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis; height: 42px; font-weight: bold; font-size: 0.90rem; line-height: 1.2; margin-bottom: 5px; color: #385F8C !important; }
        div[data-testid="stButton"] button { padding-top: 5px !important; padding-bottom: 5px !important; padding-left: 10px !important; padding-right: 10px !important; height: auto !important; }
        div[data-testid="stButton"] button p { font-size: 14px; }
        </style>
    """, unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "home"

    df, recommender, llm_tools, metrics, cf_recommender = initialize_system()

    if df is None:
        st.error("Data tidak ditemukan. Pastikan file CSV tersedia.")
        return

    # Routing
    if st.session_state["current_page"] == "home":
        page_home(df, cf_recommender)
    elif st.session_state["current_page"] == "recommender":
        page_recommender(df, recommender, llm_tools, metrics, cf_recommender)
    elif st.session_state["current_page"] == "category_view":
        page_category_view(df)

if __name__ == "__main__":
    main()