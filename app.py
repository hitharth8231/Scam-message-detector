import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopword set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load pipeline model
try:
    model = joblib.load(open('spam_classifier_pipeline.pkl', 'rb'))
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words and word not in string.punctuation
    ]
    return " ".join(cleaned)

# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")
st.title('📧 Email Spam Classifier')
st.markdown("Enter your email or SMS message below to check whether it's **Spam** or **Not Spam**.")

input_sms = st.text_area('✉️ Enter your message here:', height=150)

if st.button('🔍 Predict'):
    if input_sms.strip():
        try:
            # 1. Preprocess input
            transformed_sms = transform_text(input_sms)

            # 2. Predict using loaded pipeline
            prediction = model.predict([transformed_sms])[0]
            proba = model.predict_proba([transformed_sms])[0]

            # 3. Show result
            if prediction == 1:
                st.error(f"🚫 Spam with {proba[1]*100:.2f}% confidence")
            else:
                st.success(f"✅ Not Spam with {proba[0]*100:.2f}% confidence")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.warning('⚠️ Please enter a message before clicking predict.')
