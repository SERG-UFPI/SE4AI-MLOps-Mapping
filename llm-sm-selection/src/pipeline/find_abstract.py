import re
import time
import pandas as pd
import requests
from habanero import Crossref

RAW_CSV = "data/ROS/raw/initial_selection.csv"
OUTPUT_CSV = "data/ROS/processed/initial_selection_with_abstracts.csv"
SAVE_EVERY = 10  # salva a cada N registros processados

MISSING = {"", "Abstract não encontrado", "Erro na busca"}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AbstractFetcher/1.0 (jelsonmatheus200f@gmail.com)"})


def _strip_tags(text: str) -> str:
    """Remove tags XML/HTML (ex: <jats:p>) e normaliza espaços."""
    if not text:
        return text
    clean = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", clean).strip()


def _semantic_scholar_by_doi(doi: str) -> str | None:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract"
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code == 429:
            time.sleep(60)
            r = SESSION.get(url, timeout=15)
        if r.status_code == 200:
            abstract = r.json().get("abstract")
            if abstract:
                return abstract.strip()
    except requests.RequestException:
        pass
    return None


def _semantic_scholar_by_title(title: str) -> str | None:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 3, "fields": "abstract,title"}
    try:
        r = SESSION.get(url, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(60)
            r = SESSION.get(url, params=params, timeout=15)
        if r.status_code == 200:
            for paper in r.json().get("data", []):
                # Aceita se o título bate de forma aproximada (ignora pontuação e case)
                candidate = re.sub(r"[^\w\s]", "", (paper.get("title") or "").lower())
                target = re.sub(r"[^\w\s]", "", title.lower())
                if candidate == target or candidate.startswith(target[:40]):
                    abstract = paper.get("abstract")
                    if abstract:
                        return abstract.strip()
    except requests.RequestException:
        pass
    return None


def _crossref_by_doi(doi: str, cr: Crossref) -> str | None:
    try:
        res = cr.works(ids=doi)
        abstract = res["message"].get("abstract")
        if abstract:
            return _strip_tags(abstract)
    except Exception:
        pass
    return None


def fetch_abstracts(csv_file: str = RAW_CSV) -> None:
    cr = Crossref()
    df = pd.read_csv(csv_file)

    if "Abstract" not in df.columns:
        df["Abstract"] = ""

    pending = df[
        df["Abstract"].isna() | df["Abstract"].isin(MISSING)
    ].index.tolist()

    print(f"Total de registros: {len(df)} | Pendentes: {len(pending)}")

    saved_count = 0
    for i, index in enumerate(pending, 1):
        row = df.loc[index]
        doi = str(row.get("DOI", "")).replace("https://doi.org/", "").strip()
        title = str(row.get("Title", "")).strip()
        abstract = None

        # 1) Semantic Scholar via DOI
        if doi and doi != "nan":
            abstract = _semantic_scholar_by_doi(doi)
            time.sleep(1.2)

        # 2) Semantic Scholar via título (fallback)
        if not abstract and title:
            abstract = _semantic_scholar_by_title(title)
            time.sleep(1.2)

        # 3) Crossref via DOI (último recurso)
        if not abstract and doi and doi != "nan":
            abstract = _crossref_by_doi(doi, cr)
            time.sleep(1)

        if abstract:
            df.at[index, "Abstract"] = abstract
            status = "OK"
        else:
            df.at[index, "Abstract"] = "Abstract não encontrado"
            status = "NAO ENCONTRADO"

        print(f"[{i}/{len(pending)}] {status} — {title[:70]}")

        saved_count += 1
        if saved_count % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_CSV, index=False)

    df.to_csv(OUTPUT_CSV, index=False)
    found = df[~df["Abstract"].isin(MISSING) & df["Abstract"].notna()].shape[0]
    print(f'\nConcluído. {found}/{len(df)} abstracts encontrados.')
    print(f'Arquivo salvo em "{OUTPUT_CSV}".')


if __name__ == "__main__":
    fetch_abstracts()
