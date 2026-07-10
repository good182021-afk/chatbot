import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. إعدادات الصفحة وتفعيل الكتابة من اليمين لليسار
st.set_page_config(
    page_title="إسألني ",
    page_icon="🤖",
    layout="centered"
)

# 2. إضافة CSS للتنسيق (RTL، الخطوط، والحقوق أسفل اليسار)
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
    <style>
    /* جعل الموقع بالكامل من اليمين لليسار */
    .stApp {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
        background-color: #ffffff;
    }
    
    /* تنسيق صندوق الرد */
    .bot-reply {
        background-color: #f8f9fa;
        border-right: 5px solid #2563eb;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
        font-size: 18px;
        color: #1f2937;
        line-height: 1.6;
    }

    /* تنسيق الحقوق أسفل اليسار */
    .fixed-footer {
        position: fixed;
        bottom: 20px;
        left: 20px;
        text-align: left; /* الأسماء باللغة العربية لكن في جهة اليسار */
        direction: rtl;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
        z-index: 999;
        font-size: 14px;
        line-height: 1.4;
        border: 1px solid #e5e7eb;
    }
    .fixed-footer p { margin: 0; padding: 0; }
    .names { font-weight: bold; color: #111827; }
    .supervisor { color: #2563eb; font-weight: bold; }
    
    /* تعديل مدخل النص ليدعم العربية */
    input {
        direction: rtl !important;
        text-align: right !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. دالة تحميل البيانات (باستخدام رابطك الشغال)
@st.cache_resource
def init_smart_bot():
    data_url = "https://huggingface.co/datasets/Heba26/chatbot/resolve/main/QA_final_output.txt"
    # قراءة الملف مع تسمية الأعمدة
    df = pd.read_csv(data_url, sep='\t', header=None, names=['final_question', 'final_answer'])
    df['final_question'] = df['final_question'].fillna('')
    df['final_answer'] = df['final_answer'].fillna('')
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['final_question'])
    return df, vectorizer, tfidf_matrix

# تحميل البيانات
try:
    df, vectorizer, tfidf_matrix = init_smart_bot()
except Exception as e:
    st.error("نعتذر، حدث خطأ في تحميل البيانات.")
    st.stop()

# 4. واجهة المستخدم
st.write("<br><br>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #111827;'>🤖 إسألني</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b7280;'>مساعدك الذكي للإجابة على التساؤلات الفقهية</p>", unsafe_allow_html=True)

# صندوق الإدخال
user_input = st.text_input(label="", placeholder="اكتبي سؤالك هنا...", label_visibility="collapsed")

if user_input:
    query_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[0, best_match_idx]
    
    if highest_score > 0.15:
        st.markdown("### ✨ الرد الشرعي:")
        reply = df.iloc[best_match_idx]['final_answer']
        st.markdown(f"<div class='bot-reply'>{reply}</div>", unsafe_allow_html=True)
    else:
        st.info("نعتذر، لم نجد إجابة دقيقة في قاعدة البيانات، يرجى إعادة صياغة السؤال.")

# 5. إضافة الحقوق أسفل اليسار
st.markdown(f"""
    <div class="fixed-footer">
        <p class="names">  م إعداد: هبة محمود ابوسريويل &  فاطمة خالد </p>
        <p class="supervisor">إشراف: د. صلاح</p>
        <p>معهد التخطيط</p>
    </div>
""", unsafe_allow_html=True)
