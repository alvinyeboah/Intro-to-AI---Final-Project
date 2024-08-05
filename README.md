# Medicine Effectiveness Predictor

## Description

The Medicine Effectiveness Predictor is a Streamlit-based web application that uses machine learning to predict the effectiveness of medicines and suggest alternatives. This app is designed to assist healthcare professionals and researchers in exploring medicine effectiveness based on a trained LSTM model.

## Features

- Predict the effectiveness of a selected medicine using a trained LSTM model
- View detailed information about each medicine, including composition, uses, and review percentages
- Explore alternative medicines for similar uses
- Visualize and analyze the medicine dataset through interactive charts and graphs
- Multi-page layout for easy navigation (Home, Predict, Explore Data, About)

## Installation

1. Clone this repository:https://github.com/alvinyeboah/Intro-to-AI---Final-Project.git
cd medicine-effectiveness-predictor
 ## Usage

2. Ensure all required files are in the project directory:
- `app.py`: Main application file
- `medicine_review_model.keras`: Trained LSTM model
- `tokenizer.pkl`: Tokenizer for text processing
- `cleaned_data.csv`: Cleaned dataset
- `drug4.csv`: Drug information dataset

## Run the Streamlit app:
Open your web browser and go to `http://localhost:8501` to use the application.

## File Structure

- `app.py`: Main application file containing the Streamlit interface and prediction logic
- `medicine_review_model.keras`: Trained LSTM model for predicting medicine effectiveness
- `tokenizer.pkl`: Tokenizer used for processing text input
- `cleaned_data.csv`: Cleaned dataset used for training and analysis
- `drug4.csv`: Dataset containing detailed information about drugs
- `requirements.txt`: List of Python packages required for the project

## Dependencies

- Python 3.7+
- Streamlit
- Keras
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly

## Model Information

The medicine effectiveness prediction is based on an LSTM (Long Short-Term Memory) model trained on historical medicine data. The model takes into account various factors such as composition, uses, and review percentages to make predictions.

## Data Sources

The data used in this application is sourced from www.kaggle.com. 

## Disclaimer

This application is for informational purposes only and should not be considered as professional medical advice. Always consult with a qualified healthcare provider before making any decisions about medication.

## Contributing

Contributions to improve the Medicine Effectiveness Predictor are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT - Exisiting in repository.

## Contact


Project Link:https://github.com/alvinyeboah/Intro-to-AI---Final-Project.git

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Keras](https://keras.io/)
- [Plotly](https://plotly.com/)
