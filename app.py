import streamlit as st
import pickle as pkl
import numpy as np

# Load the pickled model
@st.cache_data()
def load_model():
    with open('rf_model.pkl', 'rb') as file:
        model = pkl.load(file)
    return model

model = load_model()

# Function to calculate confidence score
def calculate_confidence_score(model, input_features):
    # Get predictions from each tree
    all_tree_predictions = np.array([tree.predict(input_features) for tree in model.estimators_])
    # Calculate mean and standard deviation of predictions
    mean_prediction = np.mean(all_tree_predictions, axis=0)
    std_dev_prediction = np.std(all_tree_predictions, axis=0)
    return mean_prediction, std_dev_prediction

def main():
    st.title('Player Rating Prediction')
    
    # Form for user input
    st.header('Enter Player Profile')
    
    with st.form(key='profile_form'):
        potential = st.number_input('Potential', min_value=0, max_value=100, value=50)
        value_eur = st.number_input('Value (EUR)', min_value=0, value=50000)
        wage_eur = st.number_input('Wage (EUR)', min_value=0, value=5000)
        age = st.number_input('Age', min_value=15, max_value=45, value=25)
        international_reputation = st.number_input('International Reputation', min_value=0, max_value=5, value=1)
        shooting = st.number_input('Shooting', min_value=0, max_value=100, value=50)
        passing = st.number_input('Passing', min_value=0, max_value=100, value=50)
        dribbling = st.number_input('Dribbling', min_value=0, max_value=100, value=50)
        physic = st.number_input('Physic', min_value=0, max_value=100, value=50)
        attacking_short_passing = st.number_input('Attacking Short Passing', min_value=0, max_value=100, value=50)
        skill_curve = st.number_input('Skill Curve', min_value=0, max_value=100, value=50)
        skill_long_passing = st.number_input('Skill Long Passing', min_value=0, max_value=100, value=50)
        skill_ball_control = st.number_input('Skill Ball Control', min_value=0, max_value=100, value=50)
        movement_reactions = st.number_input('Movement Reactions', min_value=0, max_value=100, value=50)
        power_shot_power = st.number_input('Power Shot Power', min_value=0, max_value=100, value=50)
        power_long_shots = st.number_input('Power Long Shots', min_value=0, max_value=100, value=50)
        mentality_vision = st.number_input('Mentality Vision', min_value=0, max_value=100, value=50)
        mentality_composure = st.number_input('Mentality Composure', min_value=0, max_value=100, value=50)
        real_face = st.selectbox('Real Face', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

        submit_button = st.form_submit_button(label='Predict')
    
    if submit_button:
        input_features = np.array([[potential, value_eur, wage_eur, age, international_reputation, shooting, passing, dribbling, 
                                    physic, attacking_short_passing, skill_curve, skill_long_passing, skill_ball_control, 
                                    movement_reactions, power_shot_power, power_long_shots, mentality_vision, mentality_composure, 
                                    real_face]])
        prediction, confidence = calculate_confidence_score(model, input_features)
        
        st.subheader('Prediction Results')
        st.write(f'Player Rating: {prediction[0]}')
        st.write(f'Confidence Score (Standard Deviation): {confidence[0]:.2f}')
        
if __name__ == '__main__':
    main()
