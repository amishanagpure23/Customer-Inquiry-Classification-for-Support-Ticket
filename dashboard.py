import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
data_path = 'customer_support_tickets.csv'
data = pd.read_csv(data_path)

# Check for column names and assign variables accordingly
text_column = 'Ticket Description'  # Column name for inquiries or complaints
category_column = 'Ticket Type'  # Column name for categories

# Load the model and vectorizer
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

try:
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    vectorizer = None

# Streamlit Title
st.title("Customer Inquiry Classification for Support Ticket Management Dashboard")

# Define tabs for different functionalities
tabs = st.tabs(["Classify", "Ticket Inquiries", "Insights"])

# Classify Inquiry Tab
with tabs[0]:
    st.header("Classify Inquiry Category")
    user_input = st.text_area("Enter your inquiry:")
    if st.button("Classify", key='predict_button'):
        if user_input.strip():
            # Preprocess input (if needed) and make prediction
            processed_input = user_input.lower().strip()

            # Check if the model is a pipeline that includes TfidfVectorizer
            if hasattr(model, 'predict'):
                try:
                    # Try making prediction directly; this works if model is a pipeline
                    prediction = model.predict([processed_input])[0]
                except AttributeError:
                    # If there's an error, use vectorizer separately
                    if vectorizer:
                        vectorized_input = vectorizer.transform([processed_input])
                        prediction = model.predict(vectorized_input)[0]
                    else:
                        st.error("Vectorizer not found.")
            else:
                st.error("Model does not support predictions.")
            
            # Display prediction
            st.write(f"The entered inquiry lies in **{prediction}** category.")
        else:
            st.write("Please enter an inquiry to get a prediction.")

# Ticket Inquiries Overview Tab
with tabs[1]:
    st.header("Ticket Inquiries Overview")
    if category_column in data.columns:
        category_counts = data[category_column].value_counts().reset_index()
        category_counts.columns = ['Ticket Type', 'Count']
        st.table(category_counts)
    else:
        st.error(f"Column '{category_column}' not found in the dataset.")

# Insights Tab
with tabs[2]:
    st.header("Inquiry Category Insights")

    # Word Cloud for common words in inquiries
    st.subheader("Most Common Words in Inquiries")
    if text_column in data.columns:
        text_data = data[text_column].dropna().astype(str)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_data))
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.error(f"Column '{text_column}' not found in the dataset.")

    # Bar chart for most common inquiry categories
    st.subheader("Most Common Types of Inquiry Categories")
    if category_column in data.columns:
        category_counts = data[category_column].value_counts().head(10)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=category_counts.values, y=category_counts.index, palette='coolwarm')
        plt.title("Top 10 Most Common Inquiry Categories")
        plt.xlabel("Number of Inquiries")
        plt.ylabel("Category")
        st.pyplot(plt)
    else:
        st.error(f"Column '{category_column}' not found in the dataset.")
