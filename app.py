import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. إعدادات الصفحة وجعلها مريحة وبسيطة (Minimalist ChatGPT Style)
st.set_page_config(
    page_title="مساعدك الذكي - مشروع التخرج", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. تصميم الواجهة بالـ CSS (تخصيص الألوان والخلفية البيضاء وحقوق أسفل الصفحة)
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        direction: rtl;
        text-align: right;
    }
    h1, h2, h3, h4, p, span {
        font-family: 'Cairo', sans-serif !important;
    }
    .bot-reply {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        color: #111827;
        font-size: 18px;
        line-height: 1.6;
    }
    .fixed-footer {
        position: fixed;
        bottom: 20px;
        left: 40px;
        text-align: left;
        direction: ltr;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 8px;
        z-index: 999;
    }
    .fixed-footer p {
        margin: 2px 0;
        font-size: 14px;
        color: #4b5563;
        font-family: 'Cairo', sans-serif !important;
        text-align: left;
    }
    .fixed-footer .names {
        font-weight: bold;
        color: #1f2937;
    }
    .fixed-footer .supervisor {
        color: #2563eb;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def init_smart_bot():
    data_url = "https://huggingface.co/datasets/Heba26/chatbot/resolve/5bfc2260c029f3e8328616cdd37e4ec94da00de6/QA_final_output.txt"
    
    # قراءة الملف بدون اعتبار السطر الأول كعنوان (header=None)
    # ثم إعطاؤه أسماء أعمدة ثابتة
    df = pd.read_csv(data_url, sep='\t', header=None, names=['final_question', 'final_answer'])
    
    df['final_question'] = df['final_question'].fillna('')
    df['final_answer'] = df['final_answer'].fillna('')
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['final_question'])
    return df, vectorizer, tfidf_matrix

try:
    df, vectorizer, tfidf_matrix = init_smart_bot()
except Exception as e:
    st.error(f"خطأ في تحميل قاعدة البيانات: {e}")

# 4. بناء هيكل الصفحة البيضاء
st.write("<br><br><br>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #111827; font-weight: 700;'>🤖 نظام المحادثة الآلي الذكي</h2>", unsafe_allow_html=True)
st.write("<br>", unsafe_allow_html=True)

# صندوق الإدخال
user_input = st.text_input(label="", placeholder="اسأل البوت شيئاً... (مثال: حكم صلاة الجماعة للمرأة)", label_visibility="collapsed")

# 5. منطق الاسترجاع
if user_input:
    query_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[0, best_match_idx]
    
    st.markdown("<h4 style='color: #374151; margin-top:30px;'>🤖 الرد الشـرعي:</h4>", unsafe_allow_html=True)
    
    if highest_score > 0.15:
        reply = df.iloc[best_match_idx]['answer']
        st.markdown(f"<div class='bot-reply'>{reply}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='bot-reply' style='color:#b45309; background-color:#fffbeb;'>معذرةً، لم أفهم سؤالكِ جيداً بالصيغة الحالية، هل يمكنكِ إعادة صياغة السؤال الفقهي؟</div>", unsafe_allow_html=True)

# 6. الحقوق
st.markdown("""
    <div class="fixed-footer">
        <p class="names">إعداد: هبة & فاطمة</p>
        <p class="supervisor">إشراف: د. صلاح</p>
        <p>معهد التخطيط</p>
    </div>
""", unsafe_allow_html=True)
