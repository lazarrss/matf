from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PODESAVANJA
# =========================================================
N_ROWS = 10
SAVE_OUTPUTS = True

# Ako pokreces kao .py fajl iz PyCharm-a / terminala
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parent
else:
    # fallback za notebook / interaktivno okruzenje
    ROOT_DIR = Path.cwd()

OUT_DIR = ROOT_DIR / "preview_outputs"
if SAVE_OUTPUTS:
    OUT_DIR.mkdir(exist_ok=True)


# =========================================================
# KOLONE KOJE SU ZAISTA DATUMI
# =========================================================
DATE_COLUMNS = [
    "contractor_birthdate",
    "second_driver_birthdate",
    "vehicle_first_registration_date",
    "vehicle_country_first_registration_date",
    "vehicle_last_registration_date",
    "vehicle_inspection_report_date",
    "vehicle_inspection_expiry_date",
]


# =========================================================
# POMOCNE FUNKCIJE
# =========================================================
def prettify_column_name(col: str) -> str:
    """
    vehicle_first_registration_date -> Vehicle first registration date
    """
    col = str(col).strip().replace("_", " ")
    col = " ".join(col.split())
    return col[:1].upper() + col[1:]


def try_parse_number(value):
    """
    Ako string izgleda kao broj, pretvori ga u broj.
    U suprotnom vrati originalnu vrednost.
    """
    if pd.isna(value):
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        return value

    text = str(value).strip()
    if text == "":
        return value

    # pokusaj konverzije string -> broj
    try:
        # zamena zareza tackom ako postoji
        normalized = text.replace(",", ".")
        num = float(normalized)
        return num
    except Exception:
        return value


def format_value(value):
    """
    Pretvara vrednost u citljiv i razuman tekst.
    """
    if pd.isna(value):
        return "nedostaje"

    # bool
    if isinstance(value, (bool, np.bool_)):
        return "da" if value else "ne"

    # timestamp
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    # pokusaj da string broj pretvorimo u broj
    numeric_value = try_parse_number(value)

    if isinstance(numeric_value, (int, np.integer)):
        return str(int(numeric_value))

    if isinstance(numeric_value, (float, np.floating)):
        if float(numeric_value).is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:.2f}"

    # tekstualne bool-like vrednosti
    text = str(value).strip()
    lowered = text.lower()

    if lowered in {"true", "yes", "y", "ja", "da"}:
        return "da"
    if lowered in {"false", "no", "n", "ne"}:
        return "ne"

    return text


def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parsira samo poznate datum kolone, uz dayfirst=True
    da se izbegnu warning-i i pogresna interpretacija.
    """
    df = df.copy()

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    return df


def summarize_schema(df: pd.DataFrame, file_name: str):
    print("\n" + "-" * 120)
    print(f"STRUKTURA: {file_name}")
    print("-" * 120)
    print(f"Broj redova prikazanih u preview: {len(df)}")
    print(f"Broj kolona: {len(df.columns)}")
    print("Kolone i tipovi:")
    for col in df.columns:
        print(f" - {col} ({df[col].dtype})")


def print_raw_preview(df: pd.DataFrame, file_name: str):
    print("\n" + "-" * 120)
    print(f"RAW PREVIEW: {file_name}")
    print("-" * 120)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 50
    ):
        print(df)


def dataframe_to_human_text(df: pd.DataFrame, file_name: str) -> str:
    """
    Vraca tekstualni, citljiv prikaz prvih redova.
    """
    lines = []
    lines.append("=" * 120)
    lines.append(f"FAJL: {file_name}")
    lines.append("=" * 120)

    pretty_cols = {c: prettify_column_name(c) for c in df.columns}

    for idx, row in df.iterrows():
        lines.append(f"\nRed {idx}:")
        for col in df.columns:
            readable_col = pretty_cols[col]
            readable_val = format_value(row[col])
            lines.append(f" - {readable_col}: {readable_val}")

    return "\n".join(lines)


def save_outputs(df: pd.DataFrame, file_path: Path, human_text: str):
    """
    Snima raw preview i human-readable preview.
    """
    if not SAVE_OUTPUTS:
        return

    raw_csv_path = OUT_DIR / f"{file_path.stem}_top{N_ROWS}_raw.csv"
    txt_path = OUT_DIR / f"{file_path.stem}_top{N_ROWS}_human_readable.txt"

    df.to_csv(raw_csv_path, index=False, sep=";")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(human_text)

    print("\nSacuvano:")
    print(f" - RAW CSV: {raw_csv_path}")
    print(f" - HUMAN TXT: {txt_path}")


# =========================================================
# GLAVNI DEO
# =========================================================
def main():
    parquet_files = sorted(ROOT_DIR.rglob("*.parquet"))

    print(f"Radni folder: {ROOT_DIR}")
    print(f"Pronadjeno parquet fajlova: {len(parquet_files)}")

    if not parquet_files:
        print("Nijedan .parquet fajl nije pronadjen.")
        print("Proveri da li se parquet fajlovi nalaze u istom folderu kao skript,")
        print("ili u njegovim podfolderima.")
        return

    print("\nLista pronadjenih parquet fajlova:")
    for pf in parquet_files:
        print(f" - {pf.name}")

    for pf in parquet_files:
        try:
            print("\n" + "=" * 120)
            print(f"OBRADA FAJLA: {pf.name}")
            print("=" * 120)

            # Ucitavanje prvih N redova
            df = pd.read_parquet(pf).head(N_ROWS)

            # Parsiranje samo poznatih datumskih kolona
            df = parse_date_columns(df)

            # Ispis strukture
            summarize_schema(df, pf.name)

            # Raw pregled
            print_raw_preview(df, pf.name)

            # Human-readable pregled
            human_text = dataframe_to_human_text(df, pf.name)

            print("\n" + human_text)

            # Snimanje izlaza
            save_outputs(df, pf, human_text)

        except Exception as e:
            print("\n" + "!" * 120)
            print(f"Greska pri obradi fajla: {pf.name}")
            print(f"Detalji: {e}")
            print("!" * 120)


if __name__ == "__main__":
    main()