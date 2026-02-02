import csv

# =============================
# CONFIGURAÇÃO DE DADOS
# =============================
# Relatório do console que você mostrou
# Aqui colocamos as métricas que você já tem

# Você pode adicionar mais linhas se tiver mais épocas ou fases
metrics_data = [
    # fase, epoch, loss (estimado), accuracy
    ['treino_total', 1, 0.8, 0.74],
    ['treino_total', 2, 0.6, 0.76],
    ['re-train', 1, 0.5, 0.77],
    ['teste', 1, 0.0, 0.76]  # teste não altera o modelo
]

# Nome do CSV que será gerado
csv_file = 'metrics_chatbot_cycle.csv'

# =============================
# GERAR CSV
# =============================
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fase', 'epoch', 'loss', 'accuracy'])
    for row in metrics_data:
        writer.writerow(row)

print(f"✅ CSV gerado com sucesso: {csv_file}")
