import streamlit as st
import joblib
import pandas as pd

# Description: Streamlit application for predicting heart disease based on user input features.

# Streamlit application
def main():
    st.title("Heart Disease Prediction")

    # Sidebar for user input
    st.sidebar.header("Input Features")

    # Function to take user input
    def user_input_features():
        age = st.sidebar.slider("Age", 20, 80, 50)
        sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
        cp = st.sidebar.selectbox("Chest Pain Type", (0, 1, 2, 3))
        trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
        chol = st.sidebar.slider("Serum Cholesterol", 100, 400, 200)
        fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
        restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", (0, 1, 2))
        thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 70, 210, 150)
        exang = st.sidebar.selectbox("Exercise Induced Angina", (0, 1))
        oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
        slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", (0, 1, 2))
        ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
        thal = st.sidebar.selectbox("Thalassemia", (1, 2, 3))

        # Map user inputs to DataFrame
        data = {
            'age': age,
            'sex': 1 if sex == "Male" else 0,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        features = pd.DataFrame(data, index=[0])
        return features

    # Load model and scaler
    best_model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Get user input
    input_df = user_input_features()

    # Scale the user input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = best_model.predict(input_scaled)
    prediction_proba = best_model.predict_proba(input_scaled)[0][1]

    # Display results
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write(f"There is a **{prediction_proba*100:.2f}%** chance that the person has heart disease.")
    else:
        st.write("There is a low chance that the person has heart disease.")

    st.subheader("Input Features")
    st.write(input_df)

if __name__ == "__main__":
    main()
    

# The Streamlit application allows users to input their data and get predictions on whether they have heart disease.
# The application loads the trained model and scaler from the saved files and uses them to make predictions on the user input.
# If the model or scaler files are not found, an error message is displayed.
# The user input features are displayed along with the prediction probability.
# The application is run using the `streamlit run` command in the terminal.
# The user can interact with the application through the Streamlit interface.
# The application provides a user-friendly way to predict heart disease based on input features.
