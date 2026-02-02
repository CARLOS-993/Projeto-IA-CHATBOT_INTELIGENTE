import tkinter as tk
import json
import pickle
import random
import numpy as np
import datetime
from preprocess import preprocess



# configuração do modelo de classificação de texto e intents, limiar  de confiança
# =========================
CONFIDENCE_THRESHOLD = 0.30  # CORRETO PARA NAIVE BAYES

# DEFINIR LIMIAR DE APRENDIZAGEM

LEARNING_THRESHOLD = 0.45


#=========================
#função para registar interações para aprendizagem futura
#=========================
def log_interaction(original, processed, intent, confidence):
    log = {
        "original_text": original,
        "processed_text": processed,
        "predicted_intent": intent,
        "confidence": round(confidence, 2),
        "timestamp": datetime.datetime.now().isoformat()
    }

    with open("learning_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
        
        
#=========================
#função para marcar interações para revisão humana
#=========================
def flag_for_review(original, processed, intent, confidence):
    if confidence >= LEARNING_THRESHOLD:
        return

    review_item = {
        "original_text": original,
        "processed_text": processed,
        "predicted_intent": intent,
        "confidence": round(confidence, 2),
        "needs_review": True
    }

    with open("to_review.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(review_item, ensure_ascii=False) + "\n")





# =========================
# Carregar modelo
# =========================
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# =========================
# Função Principal do BOT
# =========================
def get_response(text):
    raw = text.lower().strip()

    # ===============================
    # CORREÇÃO 3 — REGRAS FORTES
    # ===============================

    if raw in ["oi", "ola", "olá", "ei", "hey"]:
        return "Olá!Bem-vindo à Universidade Internacional do Cuanza. Em que posso ajudar? 😊"

    if raw in ["tchau", "adeus", "até breve", "até mais", "ate logo"]:
        return "Até logo!, volte semepre 👋"

    if raw in ["obrigado", "obrigada", "valeu","viva"]:
        return "De nada! 😊"

    # ===============================
    # PRÉ-PROCESSAMENTO
    # ===============================
    processed = preprocess(text)

    if not processed.strip():
        return "❓ Podes reformular a pergunta?"

    # FRASES CURTAS NÃO VÃO AO MODELO
    if len(processed.split()) < 2:
        return "Podes escrever a pergunta com mais detalhes? 🙂"

    # ===============================
    # MODELO ML
    # ===============================
    X = vectorizer.transform([processed])
    probs = model.predict_proba(X)[0]

    max_prob = probs.max()
    intent = model.classes_[probs.argmax()]

    print(f"[DEBUG] '{processed}' → {intent} ({max_prob:.2f})")
    
    log_interaction(text, processed, intent, max_prob)
    
    flag_for_review(text, processed, intent, max_prob)


    

    if max_prob < CONFIDENCE_THRESHOLD:
        return "🤔 Ainda não entendi bem. Podes reformular com mais detalhes?"

    for item in intents["intents"]:
        if item["tag"] == intent:
            return random.choice(item["responses"])

    return "🤔 Não encontrei resposta para isso."


# =========================
# GUI SIMPLES (teste)
# =========================


BG_COLOR = "#ece5dd"
HEADER_COLOR = "#1A582C"
USER_COLOR = "#dcf8c6"
BOT_COLOR = "#ffffff"
FONT = ("Segoe UI", 10)

root = tk.Tk()
root.title("Assistente Virtual - UNIC")
root.geometry("520x650")
root.minsize(420, 500)
root.configure(bg=BG_COLOR)

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# HEADER
header = tk.Frame(root, bg=HEADER_COLOR, height=55)
header.grid(row=0, column=0, sticky="nsew")
header.grid_propagate(False)

tk.Label(
    header,
    text="Assistente Virtual - UNIC",
    bg=HEADER_COLOR,
    fg="white",
    font=("Segoe UI", 13, "bold")
).pack(pady=14)

# CHAT
chat_container = tk.Frame(root, bg=BG_COLOR)
chat_container.grid(row=1, column=0, sticky="nsew")

canvas = tk.Canvas(chat_container, bg=BG_COLOR, highlightthickness=0)
scrollbar = tk.Scrollbar(chat_container, command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

messages_frame = tk.Frame(canvas, bg=BG_COLOR)
canvas.create_window((0, 0), window=messages_frame, anchor="nw")

def update_scroll(event=None):
    canvas.configure(scrollregion=canvas.bbox("all"))

messages_frame.bind("<Configure>", update_scroll)

def add_message(text, sender="bot"):
    container = tk.Frame(messages_frame, bg=BG_COLOR)
    container.pack(fill=tk.X, padx=10, pady=6)

    if sender == "user":
        bg, anchor, padx = USER_COLOR, "e", (100, 5)
    else:
        bg, anchor, padx = BOT_COLOR, "w", (5, 100)

    tk.Label(
        container,
        text=text,
        bg=bg,
        wraplength=300,
        justify="left",
        padx=10,
        pady=6,
        font=FONT
    ).pack(anchor=anchor, padx=padx)

    canvas.update_idletasks()
    canvas.yview_moveto(1)


def send_message(event=None):
    text = entry.get().strip()
    if not text:
        return

    add_message(f"👤 {text}", "user")
    entry.delete(0, tk.END)
    response = get_response(text)
    root.after(200, lambda: add_message(f"🤖 {response}", "bot"))

# INPUT
input_frame = tk.Frame(root, bg="#f0f0f0", height=60)
input_frame.grid(row=2, column=0, sticky="nsew")
input_frame.grid_columnconfigure(0, weight=1)

entry = tk.Entry(input_frame, font=("Segoe UI", 11), relief=tk.FLAT)
entry.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
entry.bind("<Return>", send_message)

tk.Button(
    input_frame,
    text="➤",
    command=send_message,
    bg=HEADER_COLOR,
    fg="white",
    font=("Segoe UI", 12, "bold"),
    width=4,
    relief=tk.FLAT
).grid(row=0, column=1, padx=10)

# Mensagem inicial
add_message(
    "Olá! Sou o assistente virtual da Equipa 5.\n"
    "📚 Posso ajudar com cursos, propinas, inscrições e contactos.",
    "bot"
)

root.mainloop()
