from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

class SpamModelTrainer:
    def __init__(self):
        # We select algorithms well-suited for high-dimensional sparse text data.
        # class_weight='balanced' ensures the model pays extra attention to the minority class (Spam)
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            "Linear SVC": LinearSVC(class_weight='balanced', random_state=42, dual="auto"),
            "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1)
        }
        self.results = {}

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate them on the test set."""
        print("Starting model training and evaluation...\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            train_time = time.time() - start_time
            
            # Evaluate (Focusing on Precision and F1-Score)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            self.results[name] = {
                "Model_Object": model,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "Train_Time_sec": train_time
            }
            
            print(f"[{name}] Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Time: {train_time:.2f}s")
            
        return self.results

    def select_best_model(self):
        """Intelligently select the best model based on F1-Score and Precision."""
        best_name = None
        best_score = -1
        best_precision = -1
        
        for name, metrics in self.results.items():
            f1 = metrics['F1_Score']
            prec = metrics['Precision']
            
            # Primary metric: F1-Score. Secondary metric (tie-breaker): Precision
            if f1 > best_score:
                best_score = f1
                best_precision = prec
                best_name = name
            elif f1 == best_score:
                if prec > best_precision:
                    best_precision = prec
                    best_name = name
                    
        print(f"\n🏆 Best Model Selected: {best_name} (F1: {best_score:.4f}, Precision: {best_precision:.4f})")
        return best_name, self.results[best_name]['Model_Object']
