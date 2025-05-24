import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from scipy.sparse import hstack

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from deap import base, creator, tools, algorithms
import random

# ============ 1. Başlangıç ============

start_time = time.perf_counter()
df = pd.read_csv("processed_dataset_30_bin_dengeli.csv")

numeric_cols = [
    'votes_helpful', 'votes_funny', 'weighted_vote_score', 'comment_count',
    'author.num_games_owned', 'author.num_reviews',
    'author.playtime_forever', 'author.playtime_last_two_weeks', 'author.playtime_at_review',
    'steam_purchase', 'received_for_free', 'written_during_early_access'
]

# Sabit seed → TEKRARLANABİLİR SONUÇLAR için
random.seed(42)
np.random.seed(42)

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_text = vectorizer.fit_transform(df['review'].fillna(""))

X_numeric = df[numeric_cols].fillna(0).values
X_numeric_scaled = MinMaxScaler().fit_transform(X_numeric)

X_all = hstack([X_text, X_numeric_scaled])
X_all = X_all.tocsr()  # slicing için gerekli

y = df['recommended'].values
feature_names = vectorizer.get_feature_names_out().tolist() + numeric_cols

# ============ 2. Genetik Algoritma ============

num_features = X_all.shape[1]

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", lambda: random.randint(0, 1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_individual(individual):
    if sum(individual) == 0:
        return (0.0,)
    X_selected = X_all[:, individual == 1]
    model = RandomForestClassifier(n_jobs=-1)
    scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X_selected, y):
        model.fit(X_selected[train_idx], y[train_idx])
        preds = model.predict(X_selected[test_idx])
        scores.append(accuracy_score(y[test_idx], preds))
    return (np.mean(scores),)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=40)
hof = tools.HallOfFame(1)  # En iyi birey burada saklanır
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats, halloffame=hof, verbose=True)

# ============ 3. Seçilen / Elenen Özellikler ============

best_individual = hof[0]
selected_indices = [i for i, bit in enumerate(best_individual) if bit == 1]
removed_indices = [i for i, bit in enumerate(best_individual) if bit == 0]

selected_features = [feature_names[i] for i in selected_indices]
removed_features = [feature_names[i] for i in removed_indices]

print(f"\nSeçilen {len(selected_features)} özellik:", selected_features)
print(f"Çıkarılan {len(removed_features)} özellik:", removed_features)

# ============ 4. Korelasyon Matrisi ============

selected_numeric = [f for f in numeric_cols if f in selected_features]
if selected_numeric:
    plt.figure(figsize=(10, 8))
    corr_matrix = df[selected_numeric].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("GA Sonrası Özellikler Arası Korelasyon Matrisi")
    plt.tight_layout()
    plt.show()

# ============ 5. Tüm Modellerin Eğitimi ============

X_selected_final = X_all[:, selected_indices]
cv_final = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    'Naive Bayes': MultinomialNB(),
    'kNN': KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine', algorithm='brute', leaf_size=50),
    'Decision Tree': DecisionTreeClassifier(),
    'Linear SVM': LinearSVC(),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Random Forest': RandomForestClassifier(),
    'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, learning_rate='adaptive', early_stopping=True, solver='sgd'),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

performance_metrics = []
conf_matrices = {}

for name, model in models.items():
    print(f"Model: {name}")
    y_pred = cross_val_predict(model, X_selected_final, y, cv=cv_final)
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

performance_df = pd.DataFrame(performance_metrics)
performance_df = performance_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

# ============ 6. Performans Matrisi ============

plt.figure(figsize=(12, 4))
sns.set(font_scale=1)
sns.heatmap(performance_df.drop(columns='Model').set_index(performance_df['Model']).round(3),
            annot=True, cmap="YlGnBu", fmt=".3f", cbar=True)
plt.title("GA Sonrası Sınıflandırma Modelleri Performans Matrisi", fontsize=14)
plt.xlabel("Metrikler")
plt.ylabel("Modeller")
plt.tight_layout()
plt.show()

# ============ 7. Karmaşıklık Matrisleri ============

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

# ============ 8. Süreyi Yazdır ============

end_time = time.perf_counter()
print(f"\nToplam çalışma süresi: {end_time - start_time:.2f} saniye")