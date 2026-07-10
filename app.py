import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# إعدادات الواجهة
st.set_page_config(page_title="مساعدك الذكي", page_icon="🤖", layout="centered")

@st.cache_resource
def init_smart_bot():
    # الرابط المباشر الخاص بكِ
    data_url = "https://huggingface.co/datasets/Heba26/chatbot/resolve/main/QA_final_output.txt"
    
    # قراءة الملف (استخدام التاب \t كفاصل، وإذا لم ينجح جربي تغييرها لـ sep=',')
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
    st.error(f"خطأ في تحميل البيانات: {e}")
    st.stop()

# تصميم الواجهة
st.markdown("<h2 style='text-align: center;'>🤖 نظام المحادثة الآلي الذكي</h2>", unsafe_allow_html=True)
user_input = st.text_input(label="", placeholder="اسأل البوت شيئاً...")

if user_input:
    query_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    best_match_idx = similarity_scores.argmax()
    
    # التحقق من نسبة التطابق
    if similarity_scores[0, best_match_idx] > 0.1:
        st.markdown("### 🤖 الرد:")
        reply = df.iloc[best_match_idx]['final_answer']
        st.write(reply)
    else:
        st.write("عذراً، لم أجد إجابة واضحة في قاعدة البيانات.")
