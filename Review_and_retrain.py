import json
import pickle
import os
import csv
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from preprocess import preprocess

# ===============================
# FICHEIROS
# ===============================
REVIEW_FILE = "to_review.jsonl"
INTENTS_FILE = "intents.json"
METRICS_FILE = "training_metrics.csv"

MIN_CHARS = 5

# ===============================
# Verificar revisão
# ===============================
if not os.path.exists(REVIEW_FILE):
    print("ℹ️ Nenhum dado para revisão.")
    exit()

with open(REVIEW_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

if not lines:
    print("ℹ️ Nenhum dado novo.")
    exit()

# ===============================
# Carregar intents
# ===============================
with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    intents = json.load(f)

valid_tags = {i["tag"] for i in intents["intents"]}
added = 0

# ===============================
# Revisão manual
# ===============================
for line in lines:
    item = json.loads(line)
    frase = item["original_text"]

    print("\n FRASE:", frase)
    print("🤖 Previsão:", item["predicted_intent"])
    print("📉 Confiança:", item["confidence"])

    if len(frase) < MIN_CHARS:
        print("⚠️ Frase curta.")
        continue

    correct = input("👉 Intenção correta (ENTER ignora): ").strip()

    if not correct or correct not in valid_tags:
        continue

    for intent in intents["intents"]:
        if intent["tag"] == correct:
            if frase not in intent["patterns"]:
                intent["patterns"].append(frase)
                added += 1
            break

# ===============================
# Atualizar intents
# ===============================
if added > 0:
    with open(INTENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(intents, f, ensure_ascii=False, indent=2)

    print(f"✅ {added} frases adicionadas ao dataset.")
else:
    print("ℹ️ Nenhuma frase adicionada.")

open(REVIEW_FILE, "w", encoding="utf-8").close()

# ===============================
# Re-treino completo
# ===============================
texts = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        processed = preprocess(pattern)
        if processed.strip():
            texts.append(processed)
            labels.append(intent["tag"])

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9
)

X = vectorizer.fit_transform(texts)

model = MultinomialNB(alpha=0.3)
model.fit(X, labels)

preds = model.predict(X)
accuracy = round(accuracy_score(labels, preds), 3)
loss = round(1 - accuracy, 3)

# Salvar modelo
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("🔁 Modelo re-treinado com sucesso!")

# ===============================
# Registar métricas
# ===============================
fase = "re_train"   # NUNCA usar hífen
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
existe = os.path.exists(METRICS_FILE)

with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    if not existe:
        writer.writerow(["timestamp", "phase", "loss", "accuracy"])

    writer.writerow([
        timestamp,
        fase,
        round(loss, 4),
        round(accuracy, 4)
    ])
print(f"📊 Métricas registadas em {METRICS_FILE}")
