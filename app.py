import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Medicine Effectiveness Predictor", page_icon="💊", layout="wide")

# Custom CSS to enhance the app's appearance
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background: #000;
        padding: 3rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #1f618d;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #2980b9;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_keras_model():
    return load_model('medicine_review_model.keras')

model = load_keras_model()

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

tokenizer = load_tokenizer()

# Load the datasets
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    drug_df = pd.read_csv('drug4.csv')
    drug_df['Medicine Name'] = drug_df['Medicine Name'].str.lower()
    return df, drug_df

df, drug_df = load_data()

# Define functions
def predict_medicine_effectiveness(substitute):
    seq = tokenizer.texts_to_sequences([substitute])
    padded = pad_sequences(seq, maxlen=model.input_shape[1])
    pred = model.predict(padded)
    return pred[0][0]

def suggest_alternatives(selected_drug):
    selected_drug = selected_drug.lower()
    if selected_drug in drug_df['Medicine Name'].values:
        uses = drug_df.loc[drug_df['Medicine Name'] == selected_drug, 'Uses'].values[0]
        alternatives = drug_df[drug_df['Uses'].str.contains(uses, case=False, na=False)]
        alternatives = alternatives[alternatives['Medicine Name'] != selected_drug]
        return alternatives[['Medicine Name', 'Composition', 'Manufacturer', 'Excellent Review %']]
    else:
        return pd.DataFrame()

# Streamlit UI
st.title('💊 Medicine Effectiveness Predictor')

st.markdown("""
This app predicts the effectiveness of medicines based on a trained LSTM model and provides alternative suggestions.
""")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Predict", "Explore Data", "About"])

if page == "Home":
    st.header("Welcome to the Medicine Effectiveness Predictor")
    st.write("""
    This application helps you:
    - Predict the effectiveness of a medicine
    - Find alternative medicines for similar uses
    - Explore the dataset and visualize medicine statistics
    
    Navigate through the pages using the sidebar to access different features.
    """)
    
    # Display a sample visualization on the home page
    st.subheader("Sample Visualization: Top 10 Manufacturers")
    top_manufacturers = drug_df['Manufacturer'].value_counts().nlargest(10)
    fig = px.bar(x=top_manufacturers.index, y=top_manufacturers.values, 
                 labels={'x': 'Manufacturer', 'y': 'Number of Medicines'},
                 title='Top 10 Manufacturers by Number of Medicines')
    st.plotly_chart(fig)

elif page == "Predict":
    st.header("Predict Medicine Effectiveness")
    
    # Dropdown for medicine selection
    medicine_list = drug_df['Medicine Name'].tolist()
    substitute = st.selectbox('Choose a Medicine', medicine_list)
    
    if st.button('Predict Effectiveness'):
        if substitute:
            prediction = predict_medicine_effectiveness(substitute)
            
            # Create a gauge chart for the prediction
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Effectiveness Score"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.75], 'color': "gray"},
                        {'range': [0.75, 1], 'color': "darkgray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5}}))
            
            st.plotly_chart(fig)
            
            if prediction > 0.5:
                st.success(f'The medicine "{substitute}" is predicted to be effective.')
            else:
                st.warning(f'The medicine "{substitute}" is predicted to be less effective.')
            
            # Display medicine details
            medicine_details = drug_df[drug_df['Medicine Name'] == substitute.lower()].iloc[0]
            st.subheader("Medicine Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Composition:** {medicine_details['Composition']}")
                st.write(f"**Uses:** {medicine_details['Uses']}")
                st.write(f"**Manufacturer:** {medicine_details['Manufacturer']}")
            with col2:
                st.write(f"**Excellent Review %:** {medicine_details['Excellent Review %']:.2f}%")
                st.write(f"**Average Review %:** {medicine_details['Average Review %']:.2f}%")
                st.write(f"**Poor Review %:** {medicine_details['Poor Review %']:.2f}%")
            
            # Display image if available
            if pd.notna(medicine_details['Image URL']):
                st.image(medicine_details['Image URL'], caption=substitute, use_column_width=True)
            
            # Suggest alternatives
            alternatives = suggest_alternatives(substitute)
            if not alternatives.empty:
                st.subheader("Alternative Medicines")
                # Create an interactive table for alternatives
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(alternatives.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[alternatives[col] for col in alternatives.columns],
                               fill_color='lavender',
                               align='left'))
                ])
                fig.update_layout(width=700, height=400)
                st.plotly_chart(fig)
            else:
                st.info("No alternative medicines found for similar uses.")
        else:
            st.error('Please select a medicine.')

elif page == "Explore Data":
    st.header("Explore Medicine Data")
    
    # Review percentages distribution
    st.subheader("Review Percentages Distribution")
    fig = px.histogram(drug_df, x="Excellent Review %", nbins=30, 
                       title="Distribution of Excellent Review Percentages")
    st.plotly_chart(fig)
    
    # Manufacturer distribution
    st.subheader("Top Manufacturers")
    top_n = st.slider("Select number of top manufacturers to display", 5, 20, 10)
    top_manufacturers = drug_df['Manufacturer'].value_counts().nlargest(top_n)
    fig = px.bar(x=top_manufacturers.index, y=top_manufacturers.values, 
                 labels={'x': 'Manufacturer', 'y': 'Number of Medicines'},
                 title=f'Top {top_n} Manufacturers by Number of Medicines')
    st.plotly_chart(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = drug_df.select_dtypes(include=[np.number]).columns
    corr_matrix = drug_df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', aspect="auto")
    fig.update_layout(title="Correlation Heatmap of Numeric Features")
    st.plotly_chart(fig)
    
    # Raw data display
    if st.checkbox('Show raw data'):
        st.write(drug_df)

else:  # About page
    st.header("About This App")
    st.write("""
    The Medicine Effectiveness Predictor is a machine learning-powered application designed to assist in predicting the effectiveness of medicines and suggesting alternatives.
    
    **Key Features:**
    - Predict the effectiveness of a selected medicine using a trained LSTM model
    - View detailed information about each medicine, including composition, uses, and review percentages
    - Explore alternative medicines for similar uses
    - Visualize and analyze the medicine dataset
    
    **Disclaimer:** This application is for informational purposes only and should not be considered as professional medical advice. Always consult with a qualified healthcare provider before making any decisions about your medication.
    
    **Data Source:** The data used in this application is sourced from www.kaggle.com. The LSTM model was trained on [briefly describe your training data and process].
    
    **About the Developers:** This application was developed by [your name/organization]. For more information or to report issues, please contact [your contact information].
    
    **Version:** 1.0
    **Last Updated:** 5/08/24
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed Annmarie and Alvin")