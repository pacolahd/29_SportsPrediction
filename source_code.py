'''
Before running:
Extract the midfielders_model.zip to get the .pkl file
Make sure the sklearn version you have installed is version 1.2.2
Make sure xgboost is installed

'''

import streamlit as st
import pickle
import numpy as np

# Load the machine learning models
forward_model = pickle.load(open("forwards_model.pkl", "rb"))
defenders_model = pickle.load(open("defenders_model.pkl", "rb"))
midfield_model = pickle.load(open("midfielders_model.pkl", "rb"))
goalkeeper_model = pickle.load(open("goalkeepers_model.pkl", "rb"))

# Load separate scalers for each model
forward_scaler = pickle.load(open("forwards_scaler.pkl", "rb"))
defenders_scaler = pickle.load(open("defenders_scaler.pkl", "rb"))
midfield_scaler = pickle.load(open("midfielders_scaler.pkl", "rb"))
goalkeeper_scaler = pickle.load(open("goalkeepers_scaler.pkl", "rb"))

# Set Streamlit app title and favicon
st.set_page_config(
    page_title="FIFA Player Rating Prediction",
    page_icon="⚽️",
    layout="wide"
)

# App title and description
st.title('FIFA Player Rating Prediction App')
st.markdown('Welcome to the FIFA Player Rating Prediction App! This app allows you to predict player ratings for different positions.')

# Authors' names and course description
st.markdown("Authors: Sombang Patience and Ryan Mbun Tangwe")
st.markdown("Course: Introduction to AI at Ashesi University")

# Player positions and descriptions
positions_description = {
    'Forward': 'Forwards include center forward (CF), left winger (LW), right winger (RW), and striker (ST). They are known for their goal-scoring abilities.',
    'Defender': 'Defenders play crucial roles in preventing the opponent from scoring. They include center-back (CB), left-back (LB), right-back (RB), left wing-back (LWB), and right wing-back (RWB).',
    'Midfield': 'Midfielders are key players who control the game. They cover positions such as central attacking midfielder (CAM), central defensive midfielder (CDM), central midfielder (CM), left midfielder (LM), and right midfielder (RM).',
    'Goalkeeper': 'Goalkeepers (GK) are responsible for protecting the goal. They require skills like movement reactions, goalkeeping diving, handling, kicking, positioning, and reflexes.'
}


st.sidebar.header('Input Features')

# Create a select box to choose the model
model_choice = st.sidebar.selectbox('Choose a player position', ['Forward', 'Defender', 'Midfield', 'Goalkeeper'])

# Display position description when a model is selected
if model_choice in positions_description:
    st.sidebar.markdown(positions_description[model_choice])

# Initialize an empty dictionary to store the input fields
input_features = {}

if model_choice == 'Forward':
    st.sidebar.markdown("### Forward Player Features")
    st.sidebar.text("Enter statistics for the forward player")

    # List of feature names
    forward_feature_names = ['shooting', 'dribbling', 'skill_ball_control', 'movement_reactions', 'mentality_positioning', 'st', 'cf', 'rw', 'cam', 'cm', 'rm']
    forward_feature_meanings = ['Shooting Rating', 'Dribbling Rating', 'Skill Ball Control Rating', 'Movement Reactions Rating', 'Mentality Positioning Rating', 'Striker (ST) Rating', 'Center Forward (CF) Rating', 'Right Wing (RW) Rating', 'Central Attacking Midfielder (CAM) Rating', 'Central Midfielder (CM) Rating', 'Right Midfielder (RM) Rating']
    forward_feature_dict = {key: value for key, value in zip(forward_feature_names, forward_feature_meanings)}

    
  # Loop through the forward feature names and create input fields
    for feature_name in forward_feature_names:
        meaning = forward_feature_dict[feature_name]      
        input_features[feature_name] = st.sidebar.text_input(meaning , placeholder ='0', key = 's' + feature_name)

    # Determine the selected model and scaler
    selected_model = forward_model
    selected_scaler = forward_scaler

elif model_choice == 'Defender':
    st.sidebar.markdown("### Defenders Features")
    st.sidebar.text("Enter player statistics for the defender")
     # List of feature names
    defenders_feature_names = ['defending', 'movement_reactions', 'mentality_interceptions', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle', 'lwb', 'rdm', 'rwb', 'cb', 'rb']
    defenders_feature_meanings = ['Defending Rating', 'Movement Reactions Rating', 'Mentality Interceptions Rating', 'Defending Marking Awareness Rating', 'Defending Standing Tackle Rating', 'Defending Sliding Tackle Rating', 'Left Wing Back (LWB) Rating', 'Right Defensive Midfielder (RDM) Rating', 'Right Wing Back (RWB) Rating', 'Center Back (CB) Rating', 'Right Back (RB) Rating']
    defenders_feature_dict = {key: value for key, value in zip(defenders_feature_names, defenders_feature_meanings)}

    # Create input features using a loop
    for feature_name in defenders_feature_names:
        meaning = defenders_feature_dict[feature_name]      
        input_features[feature_name] = st.sidebar.text_input(meaning , placeholder ='0', key = 's'+ feature_name)

    # Determine the selected model and scaler
    selected_model = defenders_model
    selected_scaler = defenders_scaler

elif model_choice == 'Midfield':
    st.sidebar.markdown("### Midfield Player Features")
    st.sidebar.text("Enter player statistics for the midfielder.")
     # List of feature names
    midfielders_feature_names = ['passing', 'dribbling', 'attacking_short_passing', 'skill_dribbling', 'skill_ball_control', 'movement_reactions', 'mentality_composure', 'st', 'lw', 'cf', 'cam', 'cm', 'rm', 'cdm', 'rwb']
    midfielders_feature_meanings = ['Passing Rating', 'Dribbling Rating', 'Attacking Short Passing Rating', 'Skill Dribbling Rating', 'Skill Ball Control Rating', 'Movement Reactions Rating', 'Mentality Composure Rating', 'Striker (ST) Rating', 'Left Wing (LW) Rating', 'Center Forward (CF) Rating', 'Central Attacking Midfielder (CAM) Rating', 'Central Midfielder (CM) Rating', 'Right Midfielder (RM) Rating', 'Central Defensive Midfielder (CDM) Rating', 'Right Wing Back (RWB) Rating']
    midfielders_feature_dict = {key: value for key, value in zip(midfielders_feature_names, midfielders_feature_meanings)}

    # Create input features using a loop
    for feature_name in midfielders_feature_names:
        meaning = midfielders_feature_dict[feature_name]      
        input_features[feature_name] = st.sidebar.text_input(meaning , placeholder ='0', key = 's'+ feature_name)


    # Determine the selected model and scaler
    selected_model = midfield_model
    selected_scaler = midfield_scaler

else:
    st.sidebar.markdown("### Goalkeeper Features")
    st.sidebar.text("Enter player statistics for the goalkeeper.")
     # List of feature names
    goalkeepers_feature_names = ['movement_reactions', 'power_shot_power', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'gk']
    goalkeepers_feature_meanings = ['Movement Reactions Rating', 'Power Shot Power Rating', 'Goalkeeping Diving Rating', 'Goalkeeping Handling Rating', 'Goalkeeping Kicking Rating', 'Goalkeeping Positioning Rating', 'Goalkeeping Reflexes Rating', 'Goalkeeper (GK) Rating']
    goalkeepers_feature_dict = {key: value for key, value in zip(goalkeepers_feature_names, goalkeepers_feature_meanings)}

    # Create input features using a loop
    for feature_name in goalkeepers_feature_names:
        meaning = goalkeepers_feature_dict[feature_name]      
        input_features[feature_name] = st.sidebar.text_input(meaning , placeholder ='0', key = 's'+ feature_name)


    # Determine the selected model and scaler
    selected_model = goalkeeper_model
    selected_scaler = goalkeeper_scaler

if st.sidebar.button('Reset'):
    st.experimental_rerun()
    for feature_name in input_features:
        st.session_state['s'+ feature_name] = ""


# Function to extract numeric values from feature values and return them as floats
def extract_numeric_value(value):
    if '+' in value:
        numbers = value.split('+')
        return float(numbers[0]) + float(numbers[1])
    elif '-' in value:
        numbers = value.split('-')
        return float(numbers[0]) - float(numbers[1])
    # If there's neither '+' nor '-', assume it's a plain number and return it as a float
    return float(value)

# Prediction button
if st.button('Predict'):
    try:
        # Extract the input feature values from the dictionary and apply the extraction function
        input_feature_values = [extract_numeric_value(value) for value in input_features.values()]

       
        # Convert the values to a numpy array of floats
        input_features_array = np.array(input_feature_values, dtype=float)

        print(input_features_array)
       
        if np.all(input_features_array == 0):
            prediction = 0
            st.success(f'Predicted Rating: {prediction}')
        else:       
            # Scale the input features using the selected scaler
            input_features_scaled = selected_scaler.transform(input_features_array.reshape(1, -1))

            # Make predictions with the selected model
            prediction = selected_model.predict(input_features_scaled)

            st.success(f'Predicted Rating: {prediction[0]:.0f}')

        
    except Exception as e:
        if str(e) == "argument of type 'NoneType' is not iterable":
            st.error(f'Prediction error: You have to provide values for all features before attempting the prediction. Please fill in all the spaces.')
        elif str(e) == "could not convert string to float: ''":
            st.error(f'Prediction error: Not all features have been filled correctly. Please check the values you entered.')


        else:
            st.error(f'Prediction error: {str(e)}')
            

