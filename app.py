import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="إسألني", page_icon="🤖")

@st.cache_resource
def init_smart_bot():
    # الرابط المباشر
    data_url = "https://huggingface.co/datasets/Heba26/chatbot/resolve/main/QA_final_output.txt"
    # قراءة الملف
    df = pd.read_csv(data_url, sep='\t', header=None, names=['final_question', 'final_answer'])
    df = df.fillna('')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['final_question'])
    return df, vectorizer, tfidf_matrix

try:
    df, vectorizer, tfidf_matrix = init_smart_bot()
    
    st.title("🤖 إسألني")
    
    user_input = st.text_input("اكتبي سؤالك هنا:")
    
    if user_input:
        query_vector = vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
        best_match_idx = similarity_scores.argmax()
        
        if similarity_scores[0, best_match_idx] > 0.1:
            st.subheader("الرد الشرعي:")
            st.write(df.iloc[best_match_idx]['final_answer'])
        else:
            st.write("عذراً، ليس لدي إجابه.")
            
    st.markdown("---")
    st.write("إعداد: هبة & فاطمة | إشراف: د. صلاح | معهد التخطيط")

except Exception as e:
    st.error(f"حدث خطأ في تحميل التطبيق: {e}")
