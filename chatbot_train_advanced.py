import json
import pickle
import os
import csv
from datetime import datetime

# bibliotecas de machine learning ML

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#função de pré-processamento
from preprocess import preprocess

# =========================
#  Carregar intents
# =========================
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        processed = preprocess(pattern)
        if processed.strip():
            texts.append(processed)
            labels.append(intent["tag"])

print(f"📚 Total de frases: {len(texts)}")
print(f"🏷️ Total de intenções: {len(set(labels))}")

# =========================
#  SALVAR DATASET ORIGINAL
# =========================
with open("train_data.pkl", "wb") as f:
    pickle.dump((texts, labels), f)

print("📦 Dataset de treino salvo (train_data.pkl)")

# =========================
#  Vetorização TF-IDF (nucleo do modelo NLP)
# =========================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000,
    min_df=2,
    sublinear_tf=True
)


X = vectorizer.fit_transform(texts)
y = labels

# =========================
#  Treino / Teste
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Implementação do modelo Naive Bayes

model = MultinomialNB(alpha=0.3)
model.fit(X_train, y_train)

# =========================
#  Avaliação do modelo
# =========================
y_pred = model.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred), 3)
loss = round(1 - accuracy, 3)

print("\n📊 Relatório de classificação:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
#  Salvar modelo e vetorizador
# =========================
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print
("🤖 Modelo e vetorizador salvos (chatbot_model.pkl, vectorizer.pkl)")