import pandas as pd
from sklearn.metrics import classification_report
from src.utils.classify_stress import classify_stress

def run_evaluation(csv_path):
    df = pd.read_csv(csv_path)
    
    y_true = []
    y_pred = []

    for index, row in df.iterrows():
        # Splitting sentences into individual words
        target_words = str(row['target']).split()
        pred_words = str(row['predicted']).split()

        for t, p in zip(target_words, pred_words):
            y_true.append(classify_stress(t))
            y_pred.append(classify_stress(p))

    print("Tagalog Accent Classification Performance:")
    print("-" * 50)
    print(classification_report(
        y_true, 
        y_pred, 
        labels=["Malumay", "Malumi", "Mabilis", "Maragsa"]
    ))
