import streamlit as st
import joblib
import os

# Function to load the model
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at path: {model_path}")
            return None
        model = joblib.load(model_path)
        st.success("Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found. Please ensure the file '{model_path}' is in the correct path.")
        return None
    except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
        st.error(f"Joblib TerminatedWorkerError while loading the model: {e}")
        return None
    except EOFError:
        st.error(f"EOFError: The model file might be corrupted or incomplete.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Function to preprocess and predict
def predict_resume(content, model):
    try:
        processed_content = preprocess(content)
        prediction = model.predict([processed_content])
        return prediction
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Function to preprocess the resume content
def preprocess(content):
    try:
        # Implement your preprocessing steps here
        # For example, text cleaning, feature extraction, etc.
        return content
    except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        return None

# Get the current working directory
cwd = os.getcwd()
st.write(f"Current working directory: {cwd}")

# Model path
model_filename = 'resume_screening_model.pkl'
model_path = os.path.join(cwd, model_filename)

# Load the model
model = load_model(model_path)

# Streamlit UI
st.title('Resume Screening Prediction System')
st.write('Upload a resume text file')

uploaded_file = st.file_uploader('Choose a text file...', type=['txt'])

if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode('utf-8')
        st.write('Uploaded Resume:')
        st.write(content)

        if model:
            st.write('Predicting...')
            prediction = predict_resume(content, model)
            if prediction is not None:
                st.write(f'Prediction: {prediction[0]}')
            else:
                st.error("Failed to get a prediction.")
        else:
            st.error("Model is not loaded. Unable to make predictions.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.write("Please upload a text file.")



