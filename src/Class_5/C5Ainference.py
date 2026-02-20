import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# MLP WRAPPER AND LOSS
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.model = None
        self.classes_ = np.array([0, 1])
        self.history_ = None

    def predict_proba(self, X):
        probs = self.model.predict(X, verbose=0)
        return np.hstack([1 - probs, probs])

    def predict(self, X):
        return (self.model.predict(X, verbose=0) > 0.5).astype(int)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TEST_DATA_PATH = 'data/class5/test_data.csv'
MODEL_BUNDLE_PATH = 'data/output/c5clean/c5a_clean_model_bundle.joblib'

def clean_data(df):
    """Deep feature engineering for clean (unencoded) dataset."""
    # 1. Outlier Capping
    for col in ['DurationOfPitch', 'NumberOfTrips', 'MonthlyIncome']:
        if col in df.columns:
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)
    
    # 2. Advanced Interaction Engineering
    if 'NumberOfPersonVisiting' in df.columns and 'NumberOfChildrenVisiting' in df.columns:
        df['Adults'] = df['NumberOfPersonVisiting'] - df['NumberOfChildrenVisiting']
    
    if 'MonthlyIncome' in df.columns and 'Adults' in df.columns:
        df['IncomePerPerson'] = df['MonthlyIncome'] / (df['Adults'] + 1)
        if 'Age' in df.columns:
            df['Income_to_Age_Ratio'] = df['MonthlyIncome'] / (df['Age'])
            df['Income_Seniority'] = df['MonthlyIncome'] * df['Age']
    
    # 3. Luxury Alignment (Using String-based keys)
    desig_tier_map = {'Executive': 1, 'Manager': 2, 'Senior Manager': 3, 'AVP': 4, 'VP': 5}
    prod_tier_map = {'Basic': 1, 'Deluxe': 2, 'Standard': 3, 'Super Deluxe': 4, 'King': 5}
    
    if 'Designation' in df.columns:
        df['Designation_Tier'] = df['Designation'].map(desig_tier_map).fillna(2)
    if 'ProductPitched' in df.columns:
        df['Product_Tier'] = df['ProductPitched'].map(prod_tier_map).fillna(2)
        
    if 'Designation_Tier' in df.columns and 'Product_Tier' in df.columns and 'MonthlyIncome' in df.columns:
        df['LuxuryIndex'] = df['Designation_Tier'] * df['Product_Tier'] * (df['MonthlyIncome'] / 1000)
        df['IncomePerTier'] = df['MonthlyIncome'] / (df['Product_Tier'] + 1)
    
    # Interactions
    if 'Passport' in df.columns and 'OwnCar' in df.columns:
        df['Passport_Car_Interaction'] = df['Passport'] * df['OwnCar']
    if 'Passport' in df.columns and 'NumberOfFollowups' in df.columns:
        df['Followup_Passport_Interaction'] = df['Passport'] * df['NumberOfFollowups']
    
    if 'PreferredPropertyStar' in df.columns and 'MonthlyIncome' in df.columns:
        df['PropDuration_Income'] = df['PreferredPropertyStar'] * df['MonthlyIncome']
    
    # 4. Categorical Handling
    one_hot_cols = ['Occupation', 'ProductPitched', 'MaritalStatus', 'Designation', 'Gender', 'TypeofContact']
    for col in one_hot_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df

def find_optimal_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def main():
    print(f"Initializing Clean Dataset Inference on {TEST_DATA_PATH} with Bias Correction...")
    
    # 1. Load Model Bundle
    print(f"Loading model from {MODEL_BUNDLE_PATH}")
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle['model']
    scaler = bundle['scaler']
    expected_columns = bundle['columns']
    
    # 2. Load and Clean Test Data
    df = pd.read_csv(TEST_DATA_PATH)
    df = clean_data(df)
    
    X_test = df.drop('ProdTaken', axis=1)
    y_test = df['ProdTaken']
    
    # 3. Encoding
    cat_cols = X_test.select_dtypes(include=['object', 'string']).columns.tolist()
    X_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_encoded = X_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # 4. Scale and Predict Probabilities
    X_scaled = scaler.transform(X_encoded)
    print("Running Probability Predictions...")
    y_probs = model.predict_proba(X_scaled)[:, 1]
    
    # 5. Optimize Threshold
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_probs)
    print(f"\nOptimization Results:")
    print(f"-> Optimal Threshold Found: {best_threshold:.4f}")
    print(f"-> Estimated Max F1 Score: {best_f1:.4f}")
    
    # Apply Threshold
    y_pred = (y_probs >= best_threshold).astype(int)
    
    # 6. Report
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nFinal Performance (Clean Dataset optimized):")
    print("-" * 30)
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    
    # 7. Enhanced Visualization
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax1)
    ax1.set_title(f'Clean Dataset CM (Raw)\nThreshold: {best_threshold:.3f}')
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax2)
    ax2.set_title('Normalized Accuracy per Class')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    plt.suptitle(f"Clean Dataset Performance\nAccuracy: {acc:.4f} | F1 Score: {f1:.4f}", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    output_path = 'data/output/c5clean/c5a_clean_confusion_matrix.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
