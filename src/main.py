import json
from article_classification.classifier import SLRClassifier


def main():
    # Defina sua API key e modelo
    reviewer = SLRClassifier(
        api_key="AIzaSyB82pChUJLujYzNKXZw-iqCgU6erK2__SI", 
        model_name="gemini-2.5-flash"
    )

    criteria = {
        "IC1": "Studies that address software engineering in the development, evolution, or maintenance of Artificial Intelligence-based systems.",
        "IC2": "Studies that present best practices, recommendations, or patterns for the development of AI systems."
    }

    with open("data/articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:6]
    df = reviewer.run_and_export(data, criteria)
    print(df.head())


if __name__ == "__main__":    
    main()