import re
import unicodedata

# Stopwords em português (customizada para ambiente universitário)
STOPWORDS = set([
    "a", "o", "os", "as", "um", "uma",
    "de", "do", "da", "dos", "das",
    "e", "ou", "para", "por", "com",
    "em", "no", "na", "nos", "nas",
    "que", "como", "quando", "onde",
    "qual", "quais", "é", "são",
    "me", "te", "se", "minha", "meu"
])

def remover_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def preprocess(text):
    if not text:
        return ""

    # 1. Minúsculas
    text = text.lower()

    # 2. Remover acentos
    text = remover_acentos(text)

    # 3. Remover pontuação e caracteres especiais
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 4. Tokenização simples
    tokens = text.split()

    # 5. Remover stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    # 6. Reconstruir frase
    return " ".join(tokens)
