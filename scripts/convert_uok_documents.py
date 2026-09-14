import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UOK_DIR = PROJECT_ROOT / "documents" / "UoK 16.12.2025 3"
DATA_DIR = PROJECT_ROOT / "ai-service" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_exporter_sheet(filepath: Path, sector_name: str) -> list:
    if not filepath.exists():
        return []
    df = pd.read_excel(filepath, header=None)
    lines = []
    current_company = []
    
    for i in range(len(df)):
        val = str(df.iloc[i, 0]).strip() if pd.notna(df.iloc[i, 0]) else ""
        if not val or val.lower() == "nan":
            continue
        if any(skip in val for skip in [
            "Sri Lanka Export Statistics", "Period Selected", "Trader Profiles", "Company Details", "Product :"
        ]):
            continue
        
        if val in ["Tel", "Fax", "eMail", "Web"]:
            detail = ""
            if df.shape[1] > 1 and pd.notna(df.iloc[i, 1]):
                detail = str(df.iloc[i, 1]).strip()
            current_company.append(f"{val}: {detail}" if detail else val)
        else:
            if any(marker in val.upper() for marker in ["LTD", "PLC", "PVT", "ENTERPRISES", "EXPORTS", "TRADING", "CO "]) and len(current_company) > 2:
                lines.append("\n".join(current_company))
                current_company = [f"Company: {val} (Sector: {sector_name})"]
            else:
                if not current_company:
                    current_company = [f"Company: {val} (Sector: {sector_name})"]
                else:
                    current_company.append(val)
                    
    if current_company:
        lines.append("\n".join(current_company))
    return lines


def run_conversion():
    exporters_out = []
    exporters_dir = UOK_DIR / "Exporters"
    
    coconut_exp = exporters_dir / "Coconut.Xlsx"
    c_list = parse_exporter_sheet(coconut_exp, "Coconut")
    exporters_out.append(f"=== REGISTERED SRI LANKAN COCONUT EXPORTERS & TRADERS ({len(c_list)} PROFILES) ===\n")
    exporters_out.extend(c_list[:100])
    
    kithul_exp = exporters_dir / "Kithul & Jaggery.Xlsx"
    k_list = parse_exporter_sheet(kithul_exp, "Kithul & Jaggery")
    exporters_out.append(f"\n=== REGISTERED SRI LANKAN KITHUL & JAGGERY EXPORTERS & TRADERS ({len(k_list)} PROFILES) ===\n")
    exporters_out.extend(k_list)
    
    palmyrah_exp = exporters_dir / "Palmyrah.Xlsx"
    p_list = parse_exporter_sheet(palmyrah_exp, "Palmyrah")
    exporters_out.append(f"\n=== REGISTERED SRI LANKAN PALMYRAH EXPORTERS & TRADERS ({len(p_list)} PROFILES) ===\n")
    exporters_out.extend(p_list)
    
    out_file = DATA_DIR / "edb_registered_exporters_directory.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(exporters_out))
    print(f"Saved {out_file} ({len(c_list)} coconut, {len(k_list)} kithul, {len(p_list)} palmyrah profiles)")
    
    stats_out = [
        "=== SRI LANKA EXPORT DEVELOPMENT BOARD (EDB) OFFICIAL PRODUCT CATEGORIES & HS CODES ===",
        "",
        "## Kithul & Traditional Palm Sugars",
        "- HS Code H.17029022: Kithul Treacle (Caryota urens syrup)",
        "- HS Code H.17029030: Kithul Jaggery (Concentrated palm sugar blocks)",
        "- HS Code H.17029021: Palmyrah Treacle / Palmyrah Sugar syrup",
        "- Major Export Markets for Kithul/Jaggery: Australia, UK, USA, Canada, UAE, New Zealand, Italy (targeting Sri Lankan diaspora and organic natural sweetener markets)",
        "",
        "## Palmyrah Based Products",
        "- HS Code H.22089020: Palmyrah-based distilled arrack and spirits",
        "- Palmyrah Jaggery & Pinattu (dried fruit pulp): High demand in northern diaspora markets (UK, Canada, France)",
        "- Palmyrah Handcraft & Fiber: Eco-friendly brush fiber and utility baskets",
        "",
        "## Coconut & Coconut Based Products",
        "- Code S.030205: Desiccated Coconut (major export markets in Europe, Middle East, Americas)",
        "- Code S.030107: Virgin Coconut Oil (Organic cold-pressed VCO)",
        "- Code S.030301: Coconut Shell Activated Carbon & Charcoal (Haycarb, Jacobi Carbons)",
        "- Code S.030206: Coconut Flour & Defatted Coconut residue",
        "- Code S.030201: Coconut Milk, Coconut Cream, and Coconut Water (tetrapak & canned)",
        "- Code S.030401: Coconut Coir Fiber, Coco Peat grow slabs and substrates",
    ]
    stats_file = DATA_DIR / "edb_export_statistics_kithul_palmyrah.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_out))
    print(f"Saved {stats_file}")


if __name__ == "__main__":
    run_conversion()
