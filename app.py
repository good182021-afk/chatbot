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
    /* جعل الخلفية بيضاء تماماً والاتجاه من اليمين لليسار */
    .stApp {
        background-color: #ffffff;
        direction: rtl;
        text-align: right;
    }
    
    /* إلغاء الحواف والزخارف الزائدة لتشبه ChatGPT */
    h1, h2, h3, h4, p, span {
        font-family: 'Cairo', sans-serif !important;
    }
    
    /* تصميم صندوق الإجابة */
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
    
    /* تصميم الحقوق والإشراف أسفل الصفحة على اليسار ثابته */
    .fixed-footer {
        position: fixed;
        bottom: 20px;
        left: 40px;
        text-align: left;
        direction: ltr; /* لجعل التنسيق محاذي لليسار لغوياً */
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

# 3. دالة كاش لتحميل البيانات والموديل من السيرفر السحابي مباشرة لتفادي مشكلة الحجم
@st.cache_resource
def init_smart_bot():
    # قراءة الداتا من الرابط السحابي المباشر المخصص لمشروعكِ
    data_url = "https://pub-c22f03f3fb6742588f98fb66810a4e7e.r2.dev/nlp_final_processed.csv"
    df = pd.read_csv(data_url)
    df['final_question'] = df['final_question'].fillna('')
    df['final_answer'] = df['final_answer'].fillna('')
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['final_question'])
    return df, vectorizer, tfidf_matrix

try:
    df, vectorizer, tfidf_matrix = init_smart_bot()
except Exception as e:
    st.error(f"خطأ في تحميل قاعدة البيانات السحابية: {e}")

# 4. بناء هيكل الصفحة البيضاء (مكان السؤال في المنتصف)
st.write("<br><br><br>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #111827; font-weight: 700;'>🤖 نظام المحادثة الآلي الذكي</h2>", unsafe_allow_html=True)
st.write("<br>", unsafe_allow_html=True)

# صندوق الإدخال المركزي للمستخدم (مثل شات جي بي تي)
user_input = st.text_input(
    label="", 
    placeholder="اسأل البوت شيئاً... (مثال: حكم صلاة الجماعة للمرأة)",
    label_visibility="collapsed"
)

# 5. منطق المعالجة والاسترجاع عند كتابة السؤال
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

# 6. إضافة الحقوق أسفل الصفحة على اليسار بشكل احترافي وثابت
st.markdown("""
    <div class="fixed-footer">
        <p class="names">إعداد: هبة & فاطمة</p>
        <p class="supervisor">إشراف: د. صلاح</p>
        <p>معهد التخطيط</p>
    </div>
""", unsafe_allow_html=True)