import json
from article_classification.classifier import SLRClassifier


def main():
    # Defina sua API key e modelo
    reviewer = SLRClassifier(
        api_key="AIzaSyB82pChUJLujYzNKXZw-iqCgU6erK2__SI", 
        model_name="gemini-2.5-flash"
    )

    criteria = {
        "IC1": "Estudos que abordem engenharia de software ao desenvolvimento, evolução ou manutenção de sistemas baseados em Inteligência Artificial.",
        "IC2": "Estudos que apresentem boas práticas, recomendações ou padrões para o desenvolvimento de sistemas de IA.",
    }

    with open("data/articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:6]
    df = reviewer.run_and_export(data, criteria)
    print(df.head())


if __name__ == "__main__":    
    main()