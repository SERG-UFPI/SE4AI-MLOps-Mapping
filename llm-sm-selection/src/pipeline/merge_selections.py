import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "ROS" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "ROS"

INITIAL_CSV = RAW_DIR / "initial_selection.csv"
SNOWBALLING_CSV = RAW_DIR / "snowballing_selection.csv"

MARK_VALUES = {"x", "X", "?"}


def is_marked(val: str) -> bool:
    return str(val).strip() in MARK_VALUES


def build_inclusion_initial(row: pd.Series) -> str | None:
    met = [f"I{i}" for i in range(1, 5) if str(row.get(f"I{i}", "")).upper() == "TRUE"]
    return ", ".join(met) if met else None


def build_exclusion_initial(row: pd.Series) -> str | None:
    triggered = [f"E{i}" for i in range(1, 4) if str(row.get(f"E{i}", "")).upper() == "TRUE"]
    return ", ".join(triggered) if triggered else None


def build_inclusion_snowballing(row: pd.Series) -> str | None:
    cols = {
        "I1": "I1 (!ROS)",
        "I2": "I2 (!SE)",
        "I3": "I3 (!English)",
        "I4": "I4 (<2007)",
        "I5": "I5 (!Peer reviewed)",
    }
    met = [label for label, col in cols.items() if not is_marked(row.get(col, ""))]
    return ", ".join(met) if met else None


def build_exclusion_snowballing(row: pd.Series) -> str | None:
    triggered = []
    if is_marked(row.get("E1 (only-implementation)", "")):
        triggered.append("E1")
    if is_marked(row.get("E2 (dup previous)", "")) or is_marked(row.get("E2 (duplicate)", "")):
        triggered.append("E2")
    if is_marked(row.get("E3 (no full-text)", "")):
        triggered.append("E3")
    return ", ".join(triggered) if triggered else None


def should_exclude_initial(row: pd.Series) -> bool:
    if str(row.get("I2", "")).upper() == "FALSE":
        return True
    if str(row.get("I3", "")).upper() == "FALSE":
        return True
    if str(row.get("E2", "")).upper() == "TRUE":
        return True
    if str(row.get("E3", "")).upper() == "TRUE":
        return True
    return False


def should_exclude_snowballing(row: pd.Series) -> bool:
    if is_marked(row.get("I2 (!SE)", "")):
        return True
    if is_marked(row.get("I3 (!English)", "")):
        return True
    if is_marked(row.get("E2 (dup previous)", "")) or is_marked(row.get("E2 (duplicate)", "")):
        return True
    if is_marked(row.get("E3 (no full-text)", "")):
        return True
    return False


def clean(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s if s else None


def process_initial(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        if should_exclude_initial(row):
            continue
        doi_raw = clean(row.get("DOI"))
        records.append({
            "Ano": clean(row.get("Year")),
            "Autor": clean(row.get("Authors")),
            "Título": clean(row.get("Title")),
            "Abstract": None,
            "Link do pdf": doi_raw,
            "ISBNs": None,
            "ISSN": None,
            "DOI": doi_raw,
            "Inclusão": build_inclusion_initial(row),
            "Exclusão": build_exclusion_initial(row),
            "Fonte": "initial",
        })
    return records


def process_snowballing(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        if should_exclude_snowballing(row):
            continue
        doi_raw = clean(row.get("DOI"))
        url = clean(row.get("Url"))
        records.append({
            "Ano": clean(row.get("Year")),
            "Autor": clean(row.get("Author")),
            "Título": clean(row.get("Title")),
            "Abstract": clean(row.get("Abstract")),
            "Link do pdf": url or doi_raw,
            "ISBNs": None,
            "ISSN": clean(row.get("Publication Title")),
            "DOI": doi_raw,
            "Inclusão": build_inclusion_snowballing(row),
            "Exclusão": build_exclusion_snowballing(row),
            "Fonte": "snowballing",
        })
    return records


def dedup_by_doi(records: list[dict]) -> list[dict]:
    seen_doi = set()
    result = []
    for rec in records:
        doi = rec.get("DOI")
        if doi and doi in seen_doi:
            continue
        if doi:
            seen_doi.add(doi)
        result.append(rec)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Lendo {INITIAL_CSV}...")
    df_initial = pd.read_csv(INITIAL_CSV, dtype=str)
    print(f"  {len(df_initial)} artigos carregados")

    print(f"Lendo {SNOWBALLING_CSV}...")
    df_snowballing = pd.read_csv(SNOWBALLING_CSV, dtype=str)
    print(f"  {len(df_snowballing)} artigos carregados")

    initial_records = process_initial(df_initial)
    print(f"  {len(initial_records)} artigos após filtros (initial)")

    snowballing_records = process_snowballing(df_snowballing)
    print(f"  {len(snowballing_records)} artigos após filtros (snowballing)")

    merged = initial_records + snowballing_records
    merged = dedup_by_doi(merged)
    print(f"Total após merge e deduplicação por DOI: {len(merged)}")

    output_fields = ["Ano", "Autor", "Título", "Abstract", "Link do pdf", "ISBNs", "ISSN", "DOI", "Inclusão", "Exclusão", "Fonte"]

    json_path = OUTPUT_DIR / "merged_selection.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)
    print(f"JSON salvo em: {json_path}")

    df_out = pd.DataFrame(merged, columns=output_fields)
    csv_path = OUTPUT_DIR / "merged_selection.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"CSV salvo em: {csv_path}")


if __name__ == "__main__":
    main()
