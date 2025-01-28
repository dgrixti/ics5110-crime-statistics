import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import os
import random
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import joblib  

# Load datasets
# Function to merge split CSV files
def merge_csv_files(parts, output_file=None):
    """
    Merge multiple CSV parts into a single DataFrame.

    Parameters:
        parts (list): List of file paths to the CSV parts.
        output_file (str): Optional output file path to save the merged DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    dataframes = [pd.read_csv(part) for part in parts]
    merged_df = pd.concat(dataframes, ignore_index=True)
    if output_file:
        merged_df.to_csv(output_file, index=False)
    return merged_df

# Merge the split CSV files
encoded_parts = ['output_chunks/clean_df_3_single_weapon_part1.csv', 'output_chunks/clean_df_3_single_weapon_part2.csv']
dummy_parts = ['output_chunks/crime_dumscalab_part1.csv', 'output_chunks/crime_dumscalab_part2.csv']

df_encoded = merge_csv_files(encoded_parts)
df_dummy = merge_csv_files(dummy_parts)

# Initialize encoders for encoded dataset
categorical_columns = ['Agency Type', 'Victim Sex', 'Victim Age', 'Victim Ethnicity',
                      'Perpetrator Sex', 'Perpetrator Ethnicity',
                      'Relationship Category', 'Region', 'Season']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

#target_encoder = LabelEncoder()
#df_encoded['Weapon Category'] = target_encoder.fit_transform(df_encoded['Weapon Category'])

weapon_mapping = {'Non-Firearm': 0, 'Firearm': 1}  ## Force the mapping manually instead of using LabelEncoder
df_encoded['Weapon Category'] = df_encoded['Weapon Category'].map(weapon_mapping)

print("Weapon Category Mapping:")
for category, value in weapon_mapping.items():
    print(f"'{category}' -> {value}")

# Prepare features and target for encoded dataset
X_encoded = df_encoded.drop(columns=['Weapon Category'])
y_encoded = df_encoded['Weapon Category']

# Scale features for encoded dataset
scaler_encoded = StandardScaler()
X_scaled_encoded = scaler_encoded.fit_transform(X_encoded)

# Train-test split for encoded dataset
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(
    X_scaled_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Apply SMOTE for encoded dataset
smote_encoded = SMOTE(random_state=42)
X_train_resampled_encoded, y_train_resampled_encoded = smote_encoded.fit_resample(X_train_encoded, y_train_encoded)

# One-hot encoding for XGBoost
#onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_onehot_encoded = onehot_encoder.fit_transform(df_encoded[categorical_columns])
X_onehot_encoded = np.hstack((X_onehot_encoded, df_encoded.drop(columns=categorical_columns + ['Weapon Category']).values))

# Train-test split for XGBoost dataset
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_onehot_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Apply SMOTE for XGBoost dataset
smote_xgb = SMOTE(random_state=42)
X_train_resampled_xgb, y_train_resampled_xgb = smote_xgb.fit_resample(X_train_xgb, y_train_xgb)

# Prepare data for dummy variables (logistic regression)
scaler_dummy = StandardScaler()
df_dummy['Victim Age Scaled'] = scaler_dummy.fit_transform(df_dummy[['Victim Age']])
X_dummy = df_dummy.drop(columns=['Unnamed: 0', 'Victim Age', 'Weapon Category', 'Weapon Category.1'])
y_dummy = df_dummy['Weapon Category.1']

# Train-test split for dummy dataset
X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
    X_dummy, y_dummy, test_size=0.3, random_state=42, stratify=y_dummy
)

#class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
#class_weights = dict(enumerate(class_weights))

# Apply SMOTE for dummy dataset
smote_dummy = SMOTE(random_state=42)
X_train_resampled_dummy, y_train_resampled_dummy = smote_dummy.fit_resample(X_train_dummy, y_train_dummy)

# File path for saving/loading the model
neural_network_model_path = "neural_network_model.h5"
random_forest_model_path = "random_forest_model.pkl"

# Model training functions
def train_or_load_neural_network():
    if os.path.exists(neural_network_model_path):
        print("Loading saved neural network model...")
        model = load_model(neural_network_model_path)
    else:
        print("Training a new neural network model...")
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_encoded),
            y=y_train_encoded
        )
        class_weights = dict(enumerate(class_weights))
        print("Class weights:", class_weights)

        # Define the model
        model = Sequential([
            Dense(16, input_dim=X_train_resampled_encoded.shape[1], activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Add early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        model.fit(
            X_train_resampled_encoded, 
            y_train_resampled_encoded, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.2, 
            class_weight=class_weights, 
            callbacks=[early_stopping], 
            verbose=1
        )

        # Save the model
        model.save(neural_network_model_path)
    
    return model


def train_or_load_random_forest():
    if os.path.exists(random_forest_model_path):
        print("Loading saved Random Forest model...")
        rf_model = joblib.load(random_forest_model_path)
    else:
        print("Training a new Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features=5,  # specific number of features
            random_state=42,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train_resampled_encoded, y_train_resampled_encoded)  # Using SMOTE resampled data
        joblib.dump(rf_model, random_forest_model_path)  # Save the model to file
        print("Random Forest model saved to file.")
    
    return rf_model

def train_logistic_regression():
    log_model = LogisticRegression(
        C=0.15808394808776696, class_weight=None,
        max_iter=3000, penalty='l1', solver='liblinear'
    )
    log_model.fit(X_train_resampled_dummy, y_train_resampled_dummy)
    return log_model

def train_xgboost():
    # scale_pos_weight = len(y_train_resampled_xgb[y_train_resampled_xgb == 1]) / \
    #                    len(y_train_resampled_xgb[y_train_resampled_xgb == 0])
                       
    #pd.set_option('display.max_rows', 150) 
    print(y_train_resampled_xgb.sample(150))
    #pd.reset_option('display.max_rows')
    
    # Best Parameters found when using RandomizedSearchCV
                       
    xgb_model = XGBClassifier(
        eval_metric='aucpr',
        colsample_bytree = 0.923330571119153, 
        learning_rate = 0.26689728756342773, 
        max_depth = 8, 
        n_estimators = 169, 
        scale_pos_weight = 0.9877700987609598, 
        subsample = 0.9254642243837563,
        random_state=42
    )
    xgb_model.fit(X_train_resampled_xgb, y_train_resampled_xgb)
    return xgb_model

# Helper functions
def format_confusion_matrix(matrix):
    return (
        f"Confusion Matrix:\n"
        f"                    Predicted: Non-Firearm    Predicted: Firearm\n"
        f"Actual: Non-Firearm {matrix[0][0]:<6}                    {matrix[0][1]:<6}\n"
        f"Actual: Firearm     {matrix[1][0]:<6}                    {matrix[1][1]:<6}"
    )

def get_random_value(choices):
    return random.choice(choices)

# Mapping dictionaries for dummy variables
region_mapping = {"Midwest": [0, 0, 0], "Northeast": [1, 0, 0], "South": [0, 1, 0], "West": [0, 0, 1]}
season_mapping = {"Autumn": [0, 0, 0], "Spring": [1, 0, 0], "Summer": [0, 1, 0], "Winter": [0, 0, 1]}
relationship_mapping = {"Acquaintance": [0, 0, 0], "Family": [1, 0, 0], "Lover": [0, 1, 0], "Stranger": [0, 0, 1]}
agency_mapping = {"Municipal Police": [0, 0], "Other Police": [1, 0], "Sheriff": [0, 1]}
sex_mapping = {"Male": 1, "Female": 0}
ethnicity_mapping = {"Hispanic": 0, "Not Hispanic": 1}

# Gradio interface function
def gradio_interface(
    model_choice,
    region, season, relationship, agency,
    victim_sex, perpetrator_sex, victim_ethnicity, perpetrator_ethnicity,
    victim_age
):
    if model_choice == "Logistic Regression":
        # Process input for logistic regression (dummy variables)
        region_dummies = region_mapping[region]
        season_dummies = season_mapping[season]
        relationship_dummies = relationship_mapping[relationship]
        agency_dummies = agency_mapping[agency]
        
        victim_sex_numeric = sex_mapping[victim_sex]
        perpetrator_sex_numeric = sex_mapping[perpetrator_sex]
        victim_ethnicity_numeric = ethnicity_mapping[victim_ethnicity]
        perpetrator_ethnicity_numeric = ethnicity_mapping[perpetrator_ethnicity]
        
        scaled_age = scaler_dummy.transform([[victim_age]])[0][0]
        
        input_features = (
            region_dummies +
            season_dummies +
            relationship_dummies +
            agency_dummies +
            [victim_sex_numeric, perpetrator_sex_numeric,
             victim_ethnicity_numeric, perpetrator_ethnicity_numeric] +
            [scaled_age]
        )
        
        model = train_logistic_regression()
        X_test = X_test_dummy
        y_test = y_test_dummy
        
    elif model_choice == "XGBoost":
        # Process input for XGBoost (one-hot encoding)
        input_features = [
            agency, victim_sex, victim_age, victim_ethnicity,
            perpetrator_sex, perpetrator_ethnicity, relationship,
            region, season
        ]

        input_features_onehot = onehot_encoder.transform(pd.DataFrame([input_features], columns=categorical_columns))
        model = train_xgboost()
        input_pred_proba = model.predict_proba(input_features_onehot)[0]
        input_pred = np.argmax(input_pred_proba)
        X_test = X_test_xgb
        y_test = y_test_xgb

    else:
        # Process input for other models (encoded variables)
        input_features = [
            label_encoders['Agency Type'].transform([agency])[0],
            label_encoders['Victim Sex'].transform([victim_sex])[0],
            label_encoders['Victim Age'].transform([str(victim_age)])[0],
            label_encoders['Victim Ethnicity'].transform([victim_ethnicity])[0],
            label_encoders['Perpetrator Sex'].transform([perpetrator_sex])[0],
            label_encoders['Perpetrator Ethnicity'].transform([perpetrator_ethnicity])[0],
            label_encoders['Relationship Category'].transform([relationship])[0],
            label_encoders['Region'].transform([region])[0],
            label_encoders['Season'].transform([season])[0]
        ]

        input_features_scaled = scaler_encoded.transform([input_features])

        if model_choice == "Neural Network":
            model = train_or_load_neural_network()
            input_pred_proba = model.predict(input_features_scaled)
            input_pred = (input_pred_proba > 0.5).astype(int).flatten()
            input_pred_proba = np.array([1 - input_pred_proba[0][0], input_pred_proba[0][0]])
        elif model_choice == "Random Forest Classifier":
            model = train_or_load_random_forest()
        
        X_test = X_test_encoded
        y_test = y_test_encoded

    # Make predictions
    if model_choice != "Neural Network" and model_choice != "XGBoost":
        input_pred = model.predict([input_features])[0]
        input_pred_proba = model.predict_proba([input_features])[0]

    # Get model performance metrics
    if model_choice == "Neural Network":
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred)
    # average='weighted' Handle multi-class or imbalanced binary classification if applicable

    print("Weighted Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    
    print("Classification Report:")
    print(class_report)

    # Format prediction label
    if model_choice in ["Logistic Regression", "Random Forest Classifier", "Neural Network"]:
        prediction_label = "Firearm" if input_pred_proba[1] > input_pred_proba[0] else "Non-Firearm"
    elif model_choice == "XGBoost":
        prediction_label = "Firearm" if input_pred_proba[1] > input_pred_proba[0] else "Non-Firearm"
    else:
        prediction_label = target_encoder.inverse_transform([input_pred])[0]

    return (
        f"Prediction: {prediction_label}",
        f"Probability of Firearm Being Used: {float(input_pred_proba[1]):.2f}",
        f"Probability of Firearm Not Being Used: {float(input_pred_proba[0]):.2f}",
        f"{format_confusion_matrix(conf_matrix)}",
        f"Accuracy: {accuracy:.2f}",
        f"Recall: {recall:.2f}",
        f"Precision: {precision:.2f}",
        f"F1 Score: {f1:.2f}",
        f"Classification Report:\n{class_report}"
    )

# Custom CSS
custom_css = """
body {
    background: linear-gradient(to bottom right, #6a11cb, #2575fc);
    color: white;
    font-family: Arial, sans-serif;
}
.error-text {
    color: red;
    font-weight: bold;
}
"""

# Create Gradio interface
demo = gr.Interface(
    title="Weapon Use Prediction (Combined Models)",
    description="**Instructions:** Select the model and input the required information.",
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(
            ["Neural Network", "Random Forest Classifier", "Logistic Regression", "XGBoost"],
            value="Neural Network",
            label="Model Choice"
        ),
        gr.Dropdown(["Midwest", "Northeast", "South", "West"], value="Midwest", label="Region"),
        gr.Dropdown(["Autumn", "Spring", "Summer", "Winter"], value="Autumn", label="Season"),
        gr.Dropdown(["Acquaintance", "Family", "Lover", "Stranger"], value="Acquaintance", label="Relationship"),
        gr.Dropdown(["Municipal Police", "Other Police", "Sheriff"], value="Municipal Police", label="Agency"),
        gr.Dropdown(["Male", "Female"], value="Male", label="Victim Sex"),
        gr.Dropdown(["Male", "Female"], value="Male", label="Perpetrator Sex"),
        gr.Dropdown(["Hispanic", "Not Hispanic"], value="Hispanic", label="Victim Ethnicity"),
        gr.Dropdown(["Hispanic", "Not Hispanic"], value="Hispanic", label="Perpetrator Ethnicity"),
        gr.Slider(0, 100, step=1, value=30, label="Victim Age")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability of Firearm Being Used"),
        gr.Textbox(label="Probability of Firearm Not Being Used"),
        gr.Textbox(label="Confusion Matrix"),
        gr.Textbox(label="Accuracy"),
        gr.Textbox(label="Recall"),
        gr.Textbox(label="Precision"),
        gr.Textbox(label="F1 Score"),
        gr.Textbox(label="Classification Report")
    ],
    css=custom_css
)

# Launch the interface
demo.launch()
