#PCA
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Zaman ölçümünü başlat
start_time = time.perf_counter()

# Veri kümesini yükle
df = pd.read_csv("processed_dataset_30_bin_dengeli.csv")

numeric_cols = [
    'votes_helpful', 'votes_funny', 'weighted_vote_score', 'comment_count',
    'author.num_games_owned', 'author.num_reviews', 
    'author.playtime_forever', 'author.playtime_last_two_weeks', 'author.playtime_at_review',
    'steam_purchase', 'received_for_free', 'written_during_early_access'
]

# === Korelasyon Matrisi Görselleştirme ===
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Sayısal Özellikler Arası Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# === Vektörleştirme ve Özellik Hazırlama ===
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_text = vectorizer.fit_transform(df['review'].fillna("")).toarray()  # Dense matris

X_numeric = df[numeric_cols].fillna(0).values
X_numeric_scaled = MinMaxScaler().fit_transform(X_numeric)

# TF-IDF + Sayısal Özellikleri birleştir
X_combined = np.hstack([X_text, X_numeric_scaled])

# === PCA Uygula ===
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_combined)

# === PCA sonrası negatif değerleri 0-1 aralığına getir ===
X_final = MinMaxScaler().fit_transform(X_pca)

# Etiket
y = df['recommended']

# Modeller
models = {
    'Naive Bayes': MultinomialNB(),
    'kNN': KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine', algorithm='brute', leaf_size=50),
    'Decision Tree': DecisionTreeClassifier(),
    'Linear SVM': LinearSVC(),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Random Forest': RandomForestClassifier(),
    'MLP': MLPClassifier(hidden_layer_sizes=(50), max_iter=500, learning_rate='adaptive', early_stopping=True, solver='sgd'),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# Sonuçlar
performance_metrics = []
conf_matrices = {}

# 10-Fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Model: {name}")
    
    y_pred = cross_val_predict(model, X_final, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    conf_matrices[name] = cm
    
    TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    specificity = recall_score(y, y_pred, pos_label=0) 
    f1 = f1_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)

    performance_metrics.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'MCC': mcc
    })

# Performans Matrisi Görseli
performance_df = pd.DataFrame(performance_metrics)
performance_df = performance_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 4))
sns.set(font_scale=1)
sns.heatmap(performance_df.drop(columns='Model').set_index(performance_df['Model']).round(3),
            annot=True, cmap="YlGnBu", fmt=".3f", cbar=True)
plt.title(" PCA Sonrası Sınıflandırma Modelleri Performans Matrisi", fontsize=14)
plt.xlabel("Metrikler")
plt.ylabel("Modeller")
plt.tight_layout()
plt.show()

# Karmaşıklık Matrisleri
for name, cm in conf_matrices.items():
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Tahmin 0', 'Tahmin 1'],
                yticklabels=['Gerçek 0', 'Gerçek 1'])
    plt.title(f"Karmaşıklık Matrisi - {name}")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.show()

# Çalışma süresi
end_time = time.perf_counter()
print(f"Kodun toplam çalışma süresi: {end_time - start_time:.2f} saniye")