import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import os

# Configurações de plotagem
sns.set_theme(style="whitegrid")

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

# Imagem: Venn Diagrams (IA vs Humano)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
set_h = set(df[df['Included_Human']].index)

venn2([set_h, set(df[df['Included_GPT']].index)], ('Humano', 'GPT 5.4'), ax=axes[0])
axes[0].set_title("Humano vs GPT 5.4")

venn2([set_h, set(df[df['Included_Gemini']].index)], ('Humano', 'Gemini 3.1 Flash'), ax=axes[1])
axes[1].set_title("Humano vs Gemini 3.1 Flash")

venn2([set_h, set(df[df['Included_Consensus']].index)], ('Humano', 'Consenso'), ax=axes[2])
axes[2].set_title("Humano vs Consenso")

plt.savefig("venn_human_comparison.png")
plt.close()

# Extração de Títulos para a Tabela de Divergências (Consenso vs Humano)
ia_only = df[(df['Included_Consensus'] == True) & (df['Included_Human'] == False)]['Título'].tolist()
human_only = df[(df['Included_Consensus'] == False) & (df['Included_Human'] == True)]['Título'].tolist()

print("\n--- ARTIGOS ACEITOS APENAS PELA IA (CONSENSO) ---")
for t in ia_only: print(f"- {t}")

print("\n--- ARTIGOS ACEITOS APENAS PELO HUMANO ---")
for t in human_only: print(f"- {t}")

print("\nImagens geradas: venn_human_comparison.png")
