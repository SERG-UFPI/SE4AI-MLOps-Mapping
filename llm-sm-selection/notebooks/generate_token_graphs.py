import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# File paths
gpt_path = "llm-sm-selection/results/gpt-5.4-prod-3-opcoes/result.json"
gemini_path = "llm-sm-selection/results/gemini-3-flash-base-2/result.json"
consensus_path = "llm-sm-selection/results/consensu-gpt-5.4-gemini-flash-3/result.json"

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

gpt_data = load_data(gpt_path)
gemini_data = load_data(gemini_path)
consensus_data = load_data(consensus_path)

def extract_metrics(data, is_consensus=False):
    total_tokens = 0
    total_latency = 0
    prompt_tokens = 0
    completion_tokens = 0
    article_tokens = []
    article_latencies = []
    
    # For consensus breakdown
    gpt_part_tokens = 0
    gemini_part_tokens = 0

    for article in data:
        art_tokens = 0
        art_latency = 0
        
        if is_consensus:
            for res in article.get('inclusion_results', []):
                telemetry = res.get('telemetry', {})
                gpt_t = telemetry.get('gpt_tokens', 0)
                gem_t = telemetry.get('gemini_tokens', 0)
                gpt_part_tokens += gpt_t
                gemini_part_tokens += gem_t
                art_tokens += (gpt_t + gem_t)
                art_latency += (telemetry.get('gpt_latency', 0) + telemetry.get('gemini_latency', 0))
        else:
            telemetry_summary = article.get('total_article_telemetry', {})
            art_tokens = telemetry_summary.get('total_tokens', 0)
            art_latency = telemetry_summary.get('total_latency', 0)
            
            for res in article.get('inclusion_results', []):
                t = res.get('telemetry', {})
                prompt_tokens += t.get('tokens_prompt', 0)
                completion_tokens += t.get('tokens_completion', 0)
        
        total_tokens += art_tokens
        total_latency += art_latency
        article_tokens.append(art_tokens)
        article_latencies.append(art_latency)
            
    return {
        'total_tokens': total_tokens,
        'total_latency': total_latency,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'article_tokens': article_tokens,
        'article_latencies': article_latencies,
        'gpt_part_tokens': gpt_part_tokens,
        'gemini_part_tokens': gemini_part_tokens
    }

gpt_metrics = extract_metrics(gpt_data)
gemini_metrics = extract_metrics(gemini_data)
consensus_metrics = extract_metrics(consensus_data, is_consensus=True)

labels = ['GPT 5.4', 'GEMINI 3.1 Flash', 'GPT 5.4 + GEMINI 3']

# --- Graph 1: Total Token Usage (Stacked for Consensus) ---
plt.figure(figsize=(10, 7))
x = np.arange(len(labels))
width = 0.6

# Base bars
plt.bar(labels[0], gpt_metrics['total_tokens'], width, color='skyblue', label='GPT 5.4')
plt.bar(labels[1], gemini_metrics['total_tokens'], width, color='lightgreen', label='GEMINI 3.1 Flash')

# Stacked bar for consensus
plt.bar(labels[2], consensus_metrics['gpt_part_tokens'], width, color='skyblue', label='GPT Part' if 'GPT Part' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.bar(labels[2], consensus_metrics['gemini_part_tokens'], width, bottom=consensus_metrics['gpt_part_tokens'], color='lightgreen', label='GEMINI Part' if 'GEMINI Part' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title('Total Token Usage Comparison', fontsize=14)
plt.ylabel('Total Tokens', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Values on top
totals = [gpt_metrics['total_tokens'], gemini_metrics['total_tokens'], consensus_metrics['total_tokens']]
for i, v in enumerate(totals):
    plt.text(i, v + 2000, f'{int(v):,}', ha='center', va='bottom', fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig('llm-sm-selection/notebooks/total_tokens_comparison.png')
print("Updated total_tokens_comparison.png")

# --- Graph 2: Total Latency ---
plt.figure(figsize=(10, 7))
latencies = [gpt_metrics['total_latency'], gemini_metrics['total_latency'], consensus_metrics['total_latency']]
plt.bar(labels, latencies, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Total Execution Latency Comparison', fontsize=14)
plt.ylabel('Seconds', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(latencies):
    plt.text(i, v + 10, f'{int(v):,}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('llm-sm-selection/notebooks/total_latency_comparison.png')
print("Updated total_latency_comparison.png")

# --- Graph 3: Prompt vs Completion (Individual Models) ---
plt.figure(figsize=(10, 7))
models = ['GPT 5.4', 'GEMINI 3.1 Flash']
prompts = [gpt_metrics['prompt_tokens'], gemini_metrics['prompt_tokens']]
completions = [gpt_metrics['completion_tokens'], gemini_metrics['completion_tokens']]

plt.bar(models, prompts, label='Prompt Tokens', color='tab:blue', alpha=0.7)
plt.bar(models, completions, bottom=prompts, label='Completion Tokens', color='tab:orange', alpha=0.7)

plt.title('Prompt vs Completion Token Distribution', fontsize=14)
plt.ylabel('Tokens', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('llm-sm-selection/notebooks/prompt_completion_distribution.png')
print("Saved prompt_completion_distribution.png")

# --- Graph 4: Cumulative Token Usage ---
plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(gpt_metrics['article_tokens']), label='GPT 5.4', linewidth=2)
plt.plot(np.cumsum(gemini_metrics['article_tokens']), label='GEMINI 3.1 Flash', linewidth=2)
plt.plot(np.cumsum(consensus_metrics['article_tokens']), label='GPT 5.4 + GEMINI 3', linewidth=2, linestyle='--')

plt.title('Cumulative Token Consumption', fontsize=14)
plt.xlabel('Number of Articles', fontsize=12)
plt.ylabel('Cumulative Tokens', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('llm-sm-selection/notebooks/cumulative_tokens.png')
print("Updated cumulative_tokens.png")
