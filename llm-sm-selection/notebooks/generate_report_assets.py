import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib_venn import venn2, venn3
import os

# Configurações de plotagem
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 7]

# Caminhos dos arquivos
HUMAN_PATH = "../data/articles_2.json"
GPT_PATH = "../results/gpt-5.4-prod-3-opcoes/result.json"
GEMINI_PATH = "../results/gemini-3-flash-base-2/result.json"
CONSENSUS_PATH = "../results/consensu-gpt-5.4-gemini-flash-3/result.json"

def load_data():
    def extract_human_cis(inclusao_val):
        result = {'CI1': False, 'CI2': False, 'CI3': False}
        if not inclusao_val or pd.isna(inclusao_val): return result
        inclusao_val = str(inclusao_val).strip().upper()
        for ci in ['CI1', 'CI2', 'CI3']:
            if ci in inclusao_val: result[ci] = True
        return result

    # Humano
    with open(HUMAN_PATH, 'r', encoding='utf-8') as f:
        human_data = json.load(f)
    rows_human = []
    for art in human_data:
        cis = extract_human_cis(art.get('Inclusão'))
        rows_human.append({
            'Título': art.get('Título'),
            'CI1_Human': cis['CI1'], 'CI2_Human': cis['CI2'], 'CI3_Human': cis['CI3']
        })
    df_merged = pd.DataFrame(rows_human)

    # LLMs
    for path, suffix in [(GPT_PATH, 'GPT'), (GEMINI_PATH, 'Gemini'), (CONSENSUS_PATH, 'Consensus')]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rows = []
        for res in data:
            meta = res.get('article_metadata', {})
            inc_res = res.get('inclusion_results', [])
            row = {'Título': meta.get('title')}
            for criterion in ['CI1', 'CI2', 'CI3']:
                decision = "NO"
                for item in inc_res:
                    if item.get('criterion', '').upper() == criterion:
                        decision = item.get('decision', 'NO')
                        break
                row[f'{criterion}_{suffix}'] = (decision == 'YES')
            rows.append(row)
        df_llm = pd.DataFrame(rows)
        df_merged = df_merged.merge(df_llm, on='Título', how='inner')
    
    return df_merged

df = load_data()
df['Included_Human'] = df[['CI1_Human', 'CI2_Human', 'CI3_Human']].any(axis=1)
df['Included_GPT'] = df[['CI1_GPT', 'CI2_GPT', 'CI3_GPT']].any(axis=1)
df['Included_Gemini'] = df[['CI1_Gemini', 'CI2_Gemini', 'CI3_Gemini']].any(axis=1)
df['Included_Consensus'] = df[['CI1_Consensus', 'CI2_Consensus', 'CI3_Consensus']].any(axis=1)

# Imagem 1: Acceptance Comparison
plt.figure(figsize=(10, 6))
counts = [df['Included_Human'].sum(), df['Included_GPT'].sum(), df['Included_Gemini'].sum(), df['Included_Consensus'].sum()]
labels = ['Humano', 'GPT 5.4', 'Gemini 3.1 Flash', 'Consenso']
sns.barplot(x=labels, y=counts, palette='magma')
plt.title("Total de Artigos Incluídos por Método")
plt.ylabel("Quantidade")
plt.savefig("acceptance_comparison.png")
plt.close()

# Imagem 2: Venn Diagram IA Intersection
plt.figure(figsize=(10, 8))
venn3([set(df[df['Included_GPT']].index), 
       set(df[df['Included_Gemini']].index), 
       set(df[df['Included_Consensus']].index)], 
      ('GPT 5.4', 'Gemini 3.1 Flash', 'Consenso'))
plt.title("Interseção de Inclusão entre as IAs")
plt.savefig("venn_intersection.png")
plt.close()

print("Imagens geradas com sucesso!")
