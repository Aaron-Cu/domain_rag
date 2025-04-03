import os
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datasets import load_metric
from openai import OpenAI
import numpy as np

def load_and_prepare_data(filepath, output_file, sample_size=200):
    """Load the dataset, filter invalid rows, sample, and load previous results if present."""
    test_data = pd.read_csv(filepath)
    test_data = test_data[~test_data['Improved Answer'].str.startswith("Error during improvement", na=False)]
    sample_size = min(sample_size, len(test_data))
    test_data_sample = test_data.sample(n=sample_size, random_state=42)

    if os.path.exists(output_file):
        processed_data = pd.read_csv(output_file)
        columns_to_merge = ["question"] + [col for col in processed_data.columns if "Answer" in col]
        processed_data = processed_data[columns_to_merge]
        test_data_sample = pd.merge(
            test_data_sample, processed_data, how="left", on="question", suffixes=("", "_existing")
        )

    return test_data_sample

def generate_model_responses(test_data_sample, models, output_file, save_interval=5):
    """Generate model responses and incrementally save them to disk."""
    print("Generating responses from all configurations...")
    for model_name, model_id, client in models:
        col_name = f"{model_name} Answer"
        if col_name not in test_data_sample.columns:
            test_data_sample[col_name] = None

        for index, row in tqdm(test_data_sample.iterrows(), total=len(test_data_sample), desc=f"Processing {model_name}"):
            if pd.notna(test_data_sample.at[index, col_name]):
                continue

            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": row['question']}],
                    temperature=0.7,
                    max_tokens=-1,
                )
                response = completion.choices[0].message.content.strip()
            except Exception as e:
                response = f"Error during generation: {e}"

            test_data_sample.at[index, col_name] = response

            if index % save_interval == 0:
                test_data_sample.to_csv(output_file, index=False)

    test_data_sample.to_csv(output_file, index=False)
    print(f"Generated responses saved to {output_file}")
    return test_data_sample

def evaluate_scores(test_data_sample, models):
    """Compute BLEU and ROUGE scores for all model outputs."""
    bleu_metric = load_metric("bleu", trust_remote_code=True)
    rouge_metric = load_metric("rouge", trust_remote_code=True)

    results = {
        "model_names": [],
        "bleu": [],
        "rouge1": [],
        "rouge2": [],
        "rougel": [],
        "rougel_sum": [],
    }

    for model_name, _, _ in models:
        col_name = f"{model_name} Answer"
        if col_name not in test_data_sample:
            continue

        predictions = test_data_sample[col_name].tolist()
        references = [[ref.split()] for ref in test_data_sample["Improved Answer"].tolist()]

        bleu_score = bleu_metric.compute(predictions=[p.split() for p in predictions], references=references)["bleu"]
        rouge_scores = rouge_metric.compute(predictions=predictions, references=test_data_sample["Improved Answer"].tolist())

        results["model_names"].append(model_name)
        results["bleu"].append(bleu_score)
        results["rouge1"].append(rouge_scores["rouge1"].mid.fmeasure)
        results["rouge2"].append(rouge_scores["rouge2"].mid.fmeasure)
        results["rougel"].append(rouge_scores["rougeL"].mid.fmeasure)
        results["rougel_sum"].append(rouge_scores["rougeLsum"].mid.fmeasure)

    return results

def visualize_scores(results):
    """Create bar plots to compare BLEU and ROUGE scores."""
    model_names = results["model_names"]
    x = np.arange(len(model_names))
    width = 0.2

    # BLEU
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, results["bleu"], color='skyblue')
    plt.title("BLEU Scores Across Models")
    plt.ylabel("BLEU Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # ROUGE
    plt.figure(figsize=(12, 8))
    plt.bar(x - width*1.5, results["rouge1"], width, label="ROUGE-1", color='lightcoral')
    plt.bar(x - width*0.5, results["rouge2"], width, label="ROUGE-2", color='gold')
    plt.bar(x + width*0.5, results["rougel"], width, label="ROUGE-L", color='limegreen')
    plt.bar(x + width*1.5, results["rougel_sum"], width, label="ROUGE-LSUM", color='deepskyblue')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.title("ROUGE Scores Across Models")
    plt.ylabel("ROUGE F-Measure")
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_improvements(results, baseline_index=0):
    """Calculate % improvements over a baseline model."""
    def percent_change(baseline, values):
        return [(v - baseline) / baseline * 100 if baseline != 0 else 0 for v in values]

    baseline = {
        metric: results[metric][baseline_index] for metric in ["bleu", "rouge1", "rouge2", "rougel", "rougel_sum"]
    }

    improvement_table = pd.DataFrame({
        "Model": results["model_names"],
        "BLEU Improvement (%)": percent_change(baseline["bleu"], results["bleu"]),
        "ROUGE-1 F (%)": percent_change(baseline["rouge1"], results["rouge1"]),
        "ROUGE-2 F (%)": percent_change(baseline["rouge2"], results["rouge2"]),
        "ROUGE-L F (%)": percent_change(baseline["rougel"], results["rougel"]),
        "ROUGE-LSUM F (%)": percent_change(baseline["rougel_sum"], results["rougel_sum"]),
    })

    return improvement_table

def generate_wordclouds(test_data_sample, models):
    """Generate and display word clouds for each model's answers."""
    for model_name, _, _ in models:
        col_name = f"{model_name} Answer"
        if col_name not in test_data_sample:
            continue

        text = " ".join(test_data_sample[col_name].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {model_name}")
        plt.show()
