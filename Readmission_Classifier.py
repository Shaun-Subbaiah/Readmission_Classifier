import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve
import warnings
warnings.filterwarnings('ignore')


class ReadmissionClassifier:
    """Class to handle all the diabetes readmission classification tasks"""
    
    def __init__(self, csv_file, random_state=42):
        self.csv_file = csv_file
        self.data = None
        self.random_state = random_state
        
    def load_data(self):
        """Load the dataset and set up the target variable"""
        try:
            self.data = pd.read_csv(self.csv_file)
            self.data = self.data.replace('?', np.nan)
            
            if 'readmitted' not in self.data.columns:
                raise ValueError("'readmitted' column not found")
            
            # Create binary target for ML, 1 if readmitted within 30 days, else 0
            self.data['target'] = (self.data['readmitted'] == '<30').astype(int)
            
            print(f"Loading data... (check) ({len(self.data):,} patients)")
            
        except FileNotFoundError:
            print(f"Error: {self.csv_file} not found.")
            raise
    
    def prepare_features(self):
        """Get our features ready - encoding and cleaning"""
        print("Preparing features...")
        
        # Core features for prediction
        features = ['age', 'gender', 'time_in_hospital', 'num_medications', 
                   'number_inpatient', 'number_emergency', 'num_lab_procedures',
                   'change', 'diabetesMed']
        
        # Only use features that exist
        features = [f for f in features if f in self.data.columns]
        X = self.data[features].copy()
        
        # Age made to numbers for ordering 
        if 'age' in X.columns:
            age_map = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
                      '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
                      '[80-90)': 8, '[90-100)': 9}
            X['age'] = X['age'].map(age_map)
        
        # One-hot encoding - usefull
        cat_features = ['gender', 'change', 'diabetesMed']
        cat_features = [f for f in cat_features if f in X.columns]
        if cat_features:
            X = pd.get_dummies(X, columns=cat_features, drop_first=True)
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        return X, self.data['target']
    
    def get_user_input(self):
        """Ask which model to use"""
        while True:
            print("\nModel: (lr/rf/both): ", end='')
            choice = input().lower().strip()
            if choice in ['lr', 'rf', 'both']:
                return choice
            print("Invalid input. Please enter 'lr', 'rf', or 'both'")
    
    def difficulty_validation(self, X, y):
        """
        Validate on difficult cases
        
        Easy cases: Very short stays (<=2 days) or very long stays (>=10 days)
        Hard cases: Medium stays (4-7 days) where readmission risk is ambiguous
        
        Train on easy cases and test on hard cases. The main goal is to see if the model can help with the tough clinical decisions
        """
        los = self.data['time_in_hospital'].values
        
        # Difficulty based on length of stay
        easy_case = (los <= 2) | (los >= 10)
        hard_case = (los >= 4) & (los <= 7)
        
        print("\nDifficulty-Based Validation")   # Basically testing the model on ambiguous cases where it matters most
        print(f"  Clear cases: {easy_case.sum():,} patients")
        print(f"  Ambiguous cases: {hard_case.sum():,} patients")    #These are basically patients that are NOT easy or hard cases (<=2 or >=10 days), instead they are the in between cases which are harder to diagnose (>=4 and <=7 days)
        
        # Train on clear cases, test on ambiguous ones
        X_train = X[easy_case]
        y_train = y[easy_case]
        X_test = X[hard_case]
        y_test = y[hard_case]
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"  AUC on ambiguous cases: {auc:.3f}")
        
        return auc
    
    def train_model(self, X, y, model_type):
        """Train and check a model"""
        
        # Split data while maintaining class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Model setup -> balanced weights
        if model_type == 'lr':
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )
            name = 'Logistic Regression'
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            name = 'Random Forest'
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # The metrics
        return {
            'model': model,
            'name': name,
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'features': X_train.columns.tolist()
        }
    
    def show_results(self, results):
        """Print results of the info + plotting the relavant graphs"""
        
        print("\n ~ RESULTS ~")
        
        # Single model results
        if len(results) == 1:
            r = list(results.values())[0]
            print(f"Model: {r['name']}")
            print(f"AUC: {r['auc']:.3f}")
            print(f"Precision: {r['precision']:.0%} | Recall: {r['recall']:.0%} | F1: {r['f1']:.0%}")
            
            self.plot_single(r)
        
        # Both models comparison
        else:
            print(f"{'Model':<20} {'AUC':<8} {'Precision':<12} {'Recall':<8}")
            print("-" * 50)
            
            best_auc = 0
            best_name = None
            
            for r in results.values():
                print(f"{r['name']:<20} {r['auc']:<8.3f} {r['precision']:<12.0%} {r['recall']:<8.0%}")
                if r['auc'] > best_auc:
                    best_auc = r['auc']
                    best_name = r['name']
            
            print(f"\nBest: {best_name} (AUC: {best_auc:.3f})")
            
            self.plot_comparison(results)
    
    def plot_single(self, result):
        """Create plots for single model"""
        
        if result['name'] == 'Random Forest':
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Feature importance
            importances = result['model'].feature_importances_
            top_idx = np.argsort(importances)[-10:]
            
            axes[0].barh(range(10), importances[top_idx], color='steelblue')
            axes[0].set_yticks(range(10))
            axes[0].set_yticklabels([result['features'][i] for i in top_idx])
            axes[0].set_xlabel('Importance')
            axes[0].set_title('Top 10 Features')
            
            # Confusion matrix
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            axes[1].imshow(cm, cmap='Blues')
            axes[1].set_title('Confusion Matrix')
            
            for i in range(2):
                for j in range(2):
                    axes[1].text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
            
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            axes[1].set_xticks([0, 1])
            axes[1].set_yticks([0, 1])
            axes[1].set_xticklabels(['No Readmit', 'Readmit'])
            axes[1].set_yticklabels(['No Readmit', 'Readmit'])
        
        else:
            # Just confusion matrix for Logistic Regression
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            ax.imshow(cm, cmap='Blues')
            ax.set_title('Confusion Matrix')
            
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['No Readmit', 'Readmit'])
            ax.set_yticklabels(['No Readmit', 'Readmit'])
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, results):
        """Create comparison plots for both models"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # ROC curves
        for r in results.values():
            fpr, tpr, _ = roc_curve(r['y_test'], r['y_pred_proba'])
            axes[0].plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})")
        
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Feature importance info from rf
        if 'rf' in results:
            rf = results['rf']
            importances = rf['model'].feature_importances_
            top_idx = np.argsort(importances)[-10:]
            
            axes[1].barh(range(10), importances[top_idx], color='steelblue')
            axes[1].set_yticks(range(10))
            axes[1].set_yticklabels([rf['features'][i] for i in top_idx])
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Top 10 Features (RF)')
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Run the complete program to analyze data"""
        
        print("-- DIABETES READMISSION CLASSIFIER --")
       
        # Load data
        self.load_data()
        
        # model choice
        model_choice = self.get_user_input()
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Run the difficulty validation but only for Random Forest
        if model_choice in ['rf', 'both']:
            self.difficulty_validation(X, y)
        
        # Train model/s
        results = {}
        
        if model_choice in ['lr', 'both']:
            print("Training Logistic Regression...")
            results['lr'] = self.train_model(X, y, 'lr')
        
        if model_choice in ['rf', 'both']:
            print("Training Random Forest...")
            results['rf'] = self.train_model(X, y, 'rf')
        
        # Show results
        self.show_results(results)
        
        print("\n - Complete -")


# Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python readmission_classifier.py <csv_file>")
        print("Example: python readmission_classifier.py diabetic_data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    classifier = ReadmissionClassifier(csv_file)
    classifier.run()