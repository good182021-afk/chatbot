import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="مساعدك الذكي", page_icon="🤖")

@st.cache_resource
def init_smart_bot():
    data_url = "https://huggingface.co/datasets/Heba26/chatbot/resolve/main/nlp_final_processed.csv"
    df = pd.read_csv(data_url)
    
    # 1. طباعة الأعمدة لنعرف الحقيقة (تظهر في الموقع)
    st.write(f"الأعمدة المكتشفة في ملفك هي: {list(df.columns)}")
    
    # 2. تعريف الـ vectorizer هنا لكي لا يظهر خطأ NameError
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    # 3. لنفترض أن الأسماء كذا (إذا كانت مختلفة ستظهر في القائمة أعلاه ونعدلها)
    # ملاحظة: الكود سيستخدم أول عمود كسؤال وثاني عمود كجواب
    col_names = list(df.columns)
    df = df.rename(columns={col_names[0]: 'final_question', col_names[1]: 'final_answer'})
    
    df['final_question'] = df['final_question'].fillna('')
    df['final_answer'] = df['final_answer'].fillna('')
    
    tfidf_matrix = vectorizer.fit_transform(df['final_question'])
    return df, vectorizer, tfidf_matrix

# محاولة تحميل البيانات
try:
    df, vectorizer, tfidf_matrix = init_smart_bot()
    
    # واجهة المستخدم
    user_input = st.text_input("اسأل البوت شيئاً...")
    if user_input:
        query_vector = vectorizer.transform([user_input])
        # ... بقية منطق البحث
except Exception as e:
    st.error(f"خطأ: {e}")
