import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 إسألني")

# تحميل البيانات
@st.cache_resource
def load_data():
    data_url = "https://huggingface.co/datasets/Heba26/chatbot/resolve/main/QA_final_output.txt"
    # قراءة البيانات مع التأكد من الفاصل
    df = pd.read_csv(data_url, sep='\t', header=None, names=['final_question', 'final_answer'])
    df = df.fillna('')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['final_question'])
    return df, vectorizer, tfidf_matrix

try:
    df, vectorizer, tfidf_matrix = load_data()
    
    user_input = st.text_input("اكتبي سؤالك هنا:")
    
    if user_input:
        query_vector = vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
        best_match_idx = similarity_scores.argmax()
        
        if similarity_scores[0, best_match_idx] > 0.1:
            st.write("### الرد :")
            st.write(df.iloc[best_match_idx]['final_answer'])
        else:
            st.write("عذراا،،ليس لدي إجابة")
            
except Exception as e:
    st.write(f"خطأ في التحميل: {e}")

st.markdown("---")
st.write("إعداد: هبة & فاطمة | إشراف: د. صلاح")
