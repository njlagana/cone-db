import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser

from utils import calculate_HRR_O2_only, colorize

# Parses Atlas cone calorimeter exports (CONE150/CONE171 "Atlas Cone" software,
# e.g. the FAA example 021210A1.txt) into the same Parsed CSV + JSON layout that
# Parse_Cone-FTT.py writes, so the files can be imported into the Cone Explorer.
#
# Usage: python scripts/Parse_Cone-FAA.py [<input_dir> <output_source>]
#   input_dir      folder of Atlas .txt exports (default: data/raw/FAA)
#   output_source  first part of "Original Source" and the output folder name (default: FAA)

args = sys.argv[1:]
if len(args) not in (0, 2):
    print("""Usage: python Parse_Cone-FAA.py <input_dir> <output_source>
          Leave empty to use defaults.""")
    sys.exit()

# Path handling: relative to this script's location (same as Parse_Cone-FTT.py)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "FAA"
SOURCE = "FAA"
if len(args) == 2:
    INPUT_DIR = Path(args[0])
    SOURCE = args[1]

OUTPUT_DIR = PROJECT_ROOT / "Exp-Data_Parsed" / SOURCE
META_DIR = PROJECT_ROOT / "Metadata" / "Parsed" / SOURCE
LOG_FILE = PROJECT_ROOT / f"parse_{SOURCE}_log.json"

# Same metadata keys, in the same order, as Parse_Cone-FTT.py
EXPECTED_KEYS = [
    "Material ID", "Material Name", "Sample Mass (g)", "Residual Mass (g)", "Replicate",
    "Original Testname", "Testname", "Thickness (mm)", "Sample Description", "Specimen Prep",
    "Instrument", "Test Date", "Test Time", "Operator", "Director", "Sponsor", "Institution",
    "Report Name", "Original Source", "Parsed", "Auto Prepared", "Manually Prepared", "SmURF",
    "Bad Data", "Autoprocessed", "Manually Reviewed Series", "Pass Review", "Published",
    "Heat Flux (kW/m2)", "Orientation", "C Factor", "Surface Area (m2)", "Grid", "Edge Frame",
    "Ignition Source", "Separation (mm)", "Test Start Time (s)", "Test End Time (s)",
    "MLR EOT Mass (g/m2)", "End of test criterion", "Heat of Combustion O2 (MJ/kg)",
    "OD Correction Factor", "Substrate", "Non-scrubbed", "Duct Diameter (m)",
    "O2 Delay Time (s)", "CO2 Delay Time (s)", "CO Delay Time (s)",
    "Ambient Temperature (°C)", "Barometric Pressure (Pa)", "Relative Humidity (%)",
    "X_O2 Initial", "X_CO2 Initial", "X_CO Initial",
    "t_ignition (s)", "t_ignition Outlier",
    "m_ignition (g)", "m_ignition Outlier",
    "Residue Yield (%)", "Residue Yield Outlier",
    "Heat Release Rate Outlier",
    "Average HRRPUA 60s (kW/m2)", "Average HRRPUA 60s Outlier",
    "Average HRRPUA 180s (kW/m2)", "Average HRRPUA 180s Outlier",
    "Average HRRPUA 300s (kW/m2)", "Average HRRPUA 300s Outlier",
    "Steady Burning MLRPUA (g/s-m2)", "Steady Burning MLRPUA Outlier",
    "Peak MLRPUA (g/s-m2)", "Peak MLRPUA Outlier",
    "Steady Burning HRRPUA (kW/m2)", "Steady Burning HRRPUA Outlier",
    "Peak HRRPUA (kW/m2)", "Peak HRRPUA Outlier",
    "Total Heat Release (MJ/m2)", "Total Heat Release Outlier",
    "Effective Heat Of Combustion (kJ/g)", "Effective Heat Of Combustion Outlier",
    "Average Specific Extinction Area (m2/kg)", "Average Specific Extinction Area Outlier",
    "Smoke Production Pre-ignition (m2/m2)", "Smoke Production Pre-ignition Outlier",
    "Smoke Production Post-ignition (m2/m2)", "Smoke Production Post-ignition Outlier",
    "Smoke Production Total (m2/m2)", "Smoke Production Total Outlier",
    "Y_Soot (g/g)", "Y_Soot Outlier",
    "Y_CO2 (g/g)", "Y_CO2 Outlier",
    "Y_CO (g/g)", "Y_CO Outlier",
    "Fire Growth Potential (m2/J)", "Fire Growth Potential Outlier",
    "Ignition Energy (MJ/m2)", "Ignition Energy Outlier",
    "t_flameout (s)", "t_flameout Outlier",
    "Comments", "Data Corrections",
]

# Data columns written to the Parsed CSV (same as Parse_Cone-FTT.py)
OUTPUT_COLUMNS = [
    "Time (s)", "Mass (g)", "HRR (kW)", "MFR (kg/s)", "T Duct (K)",
    "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", "K Smoke (1/m)",
]


#region parse_dir
def parse_dir(input_dir):
    paths = sorted(Path(input_dir).glob("**/*.txt"))
    print(colorize(f"Found {len(paths)} files to parse", "purple"))

    log = {}
    parsed_ok = 0
    for path in paths:
        try:
            parse_file(path)
            parsed_ok += 1
            print(colorize(" - Parsed successfully", "green"))
        except Exception as e:
            log[str(path)] = str(e)
            print(colorize(f" - Error parsing {path}: {e}", "red"))

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4)
    if paths:
        print(colorize(f"COMPLETE: {parsed_ok}/{len(paths)} files parsed successfully", "blue"))


#region parse_file
def parse_file(file_path):
    file_path = Path(file_path)
    print(f"Parsing: {file_path.stem}")

    # The exports contain mangled degree/superscript characters, so replace undecodable bytes
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().replace("\r", "").split("\n")

    # The data table starts at the column-label row ("Scan  HRR  HOC ..."), followed by a units row
    header_idx = next((i for i, line in enumerate(lines) if line.strip().startswith("Scan")), None)
    if header_idx is None:
        raise Exception("No data table found (no line starting with 'Scan')")

    metadata = parse_metadata(lines[:header_idx], file_path)
    data, extra = parse_data(lines[header_idx:], metadata)

    if len(data) < 20:
        raise Exception(f"Only {len(data)} scans of data before Test Length")

    self_check(data, metadata, extra)

    year = metadata["Test Date"][:4]
    (OUTPUT_DIR / year).mkdir(parents=True, exist_ok=True)
    (META_DIR / year).mkdir(parents=True, exist_ok=True)
    data_output_path = OUTPUT_DIR / year / f"{file_path.stem}.csv"
    metadata_output_path = META_DIR / year / f"{file_path.stem}.json"

    # Don't overwrite metadata that has already been SmURFed or marked as Bad Data
    if metadata_output_path.exists():
        with open(metadata_output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("Bad Data"):
            print(colorize(f"{file_path.stem} has been marked as Bad Data. Skipping.", "yellow"))
            return
        if existing.get("SmURF"):
            print(colorize(f"{file_path.stem} has already been SmURFed. Metadata not overwritten.", "yellow"))
            data.to_csv(data_output_path, index=False)
            return

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    data.to_csv(data_output_path, index=False)
    print(f"\tData: {data.shape[0]}x{data.shape[1]} -> {data_output_path.relative_to(PROJECT_ROOT)}")


#region parse_metadata
def parse_metadata(header_lines, file_path):
    lines = [line.rstrip() for line in header_lines]
    text = "\n".join(lines)

    def extract_num(key):
        # allow a label suffix before the colon, e.g. "C-Factor (Methane)   : 0.04454730"
        match = re.search(re.escape(key) + r"[^:\n]*:\s*(-?\d+\.?\d*)", text)
        return float(match.group(1)) if match else None

    def extract_block(key):
        # value on the same line as "key :" plus any indented/continuation lines up to the next blank line
        for i, line in enumerate(lines):
            if re.match(r"\s*" + re.escape(key) + r"\s*:", line):
                parts = [line.split(":", 1)[1].strip()]
                for nxt in lines[i + 1:]:
                    if not nxt.strip() or re.match(r"\s*[A-Za-z][A-Za-z .()/-]*\s*:(\s|$)", nxt):
                        break
                    parts.append(nxt.strip())
                value = " ".join(p for p in parts if p)
                return value or None
        return None

    def extract_section(start, end):
        # free-text comment sections, e.g. "Pre-test Comments:" up to "Post-test Comments:"
        match = re.search(re.escape(start) + r"\s*\n(.*?)\n\s*(?:" + end + ")", text, flags=re.DOTALL)
        if not match:
            return None
        value = " ".join(line.strip() for line in match.group(1).split("\n") if line.strip())
        return value or None

    metadata = {key: None for key in EXPECTED_KEYS}

    # Date and time, e.g. "Test Date : 10-Dec-02  Test Time : 02:14 p.m.  Operator  : SC"
    date = re.search(r"Test Date\s*:\s*(\d{1,2}-\w{3}-\d{2,4})", text).group(1)
    time_match = re.search(r"Test Time\s*:\s*(\d{1,2}:\d\d(\s*[ap]\.?\s*m\.?)?)", text, flags=re.IGNORECASE)
    # dateutil understands "02:14 pm" but not "02:14 p.m.", so drop the periods
    time = time_match.group(1).replace(".", "") if time_match else "00:00"
    test_datetime = parser.parse(f"{date} {time}", dayfirst=True)
    metadata["Test Date"] = test_datetime.date().isoformat()
    metadata["Test Time"] = test_datetime.strftime("%H:%M")

    operator = re.search(r"Operator\s*:\s*(\S.*)", text)
    metadata["Operator"] = operator.group(1).strip() if operator else None

    # The first non-blank line of the file is the lab header (e.g. "FAA", "USDA-FPL")
    lab = next((line.strip() for line in lines if line.strip()), None)
    metadata["Institution"] = lab
    metadata["Instrument"] = "Atlas Cone Calorimeter"

    # Test parameters
    metadata["Test End Time (s)"] = extract_num("Test Length")
    # Surface area is given in m2 (the old version of this script converted to cm2)
    metadata["Surface Area (m2)"] = extract_num("Sample Surface Area")
    heat_flux = extract_num("Radiant Heat Flux")
    # store whole-number fluxes as integers (65.0 -> 65) so the Explorer names files "65kW", not "65p0kW"
    metadata["Heat Flux (kW/m2)"] = int(heat_flux) if heat_flux is not None and heat_flux.is_integer() else heat_flux
    orientation = extract_block("Sample Orientation")
    metadata["Orientation"] = orientation.upper() if orientation else None
    metadata["Sponsor"] = extract_block("Sponsor")
    metadata["Material Name"] = extract_block("Sample Material")

    # Comments: same list format as Parse_Cone-FTT.py
    comments = []
    pre = extract_section("Pre-test Comments:", r"Post-test Comments:")
    post = extract_section("Post-test Comments:", r"[^\n]*Reduction Parameters")
    if pre:
        comments.append(f"Pre-test: {pre}")
    if post:
        comments.append(f"Post-test: {post}")
    metadata["Comments"] = comments

    # Reduction parameters
    metadata["C Factor"] = extract_num("C-Factor")
    metadata["Heat of Combustion O2 (MJ/kg)"] = extract_num("Conversion Factor")
    for key, gas in (("X_O2 Initial", "O2"), ("X_CO2 Initial", "CO2"), ("X_CO Initial", "CO")):
        baseline = extract_num(f"Baseline {gas} Reading")
        metadata[key] = baseline / 100 if baseline is not None else None

    # Test results
    t_ign = extract_num("Time to Sustained Ignition")
    # the Explorer only accepts whole seconds; round rather than truncate
    metadata["t_ignition (s)"] = int(round(t_ign)) if t_ign is not None else None
    metadata["Sample Mass (g)"] = extract_num("Entered Initial Specimen Mass")
    metadata["Residual Mass (g)"] = extract_num("Measured Final Specimen Mass")

    metadata["Original Testname"] = Path(file_path).stem
    metadata["Original Source"] = f"{SOURCE}/{test_datetime.year}"
    metadata["Parsed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata["Data Corrections"] = []

    # Values from the "Test Results" block, kept only for the self-check (not written to the JSON)
    metadata["_reported"] = {
        "peak_hrr": extract_num("Peak Heat Release Rate"),
        "thr": extract_num("Total Heat Released"),
        "t_ign_raw": t_ign,
    }
    return metadata


#region parse_data
def parse_data(table_lines, metadata):
    names = [n.strip() for n in table_lines[0].split("\t")]
    units = [u.strip() for u in table_lines[1].split("\t")]
    # CO2 and CO appear twice (yields in kg/kg and concentrations in %), so key columns by name + unit
    columns = [f"{n} [{u}]" for n, u in zip(names, units)]

    rows = [line.split("\t")[: len(columns)] for line in table_lines[2:] if line.strip()]
    df = pd.DataFrame(rows, columns=columns).apply(pd.to_numeric, errors="coerce")

    def col(name, unit_hint=""):
        for c in df.columns:
            n, u = c[:-1].split(" [", 1)
            if n == name and unit_hint in u:
                return df[c]
        raise Exception(f"Column '{name}' ({unit_hint}) not found")

    area = metadata["Surface Area (m2)"]
    if not area:
        raise Exception("No Sample Surface Area in header")

    data = pd.DataFrame({
        "Time (s)": col("Time"),
        "Mass (g)": col("Mass"),
        # the instrument's HRR is per unit area; cone-db stores kW
        "HRR (kW)": col("HRR", "kW") * area,
        "MFR (kg/s)": col("Duct Mass"),
        # Parse_Cone-FTT.py maps the smoke thermocouple to "T Duct (K)"
        "T Duct (K)": col("Smoke T") + 273.15,
        "O2 (Vol fr)": col("O2", "%") / 100,
        "CO2 (Vol fr)": col("CO2", "%") / 100,
        "CO (Vol fr)": col("CO", "%") / 100,
        # the units row says "l/m" but this is 1/m
        "K Smoke (1/m)": col("Ext Coef"),
    })
    # kept for the HRR recalculation in the self-check, not written out
    extra = pd.DataFrame({"dP": col("Exh Pres"), "Te": col("Stack T") + 273.15})

    # The export keeps logging after the test ends (about 120 scans in the examples). Past that
    # point the analyzers go off line (O2 reads below zero) and the HRR column shows fake values
    # of ~8,000 kW/m2, so cut at the Test Length from the header.
    test_length = metadata.get("Test End Time (s)")
    if test_length is not None:
        keep = data["Time (s)"] <= test_length
        n_dropped = int((~keep).sum())
        data, extra = data[keep], extra[keep]
        if n_dropped:
            print(f"\tDropped {n_dropped} scans after Test Length ({test_length} s)")

    if data["Time (s)"].diff().max() > 1:
        raise Exception("Time increments are not 1 second")

    return data.reset_index(drop=True), extra.reset_index(drop=True)


#region self_check
def self_check(data, metadata, extra):
    """Compare the parsed data against the instrument's own Test Results block and
    against an HRR recalculated from the raw columns. Results go in "Data Corrections"
    (the same list the Explorer appends to), so nothing is silently changed."""
    reported = metadata.pop("_reported")
    area = metadata["Surface Area (m2)"]
    hrrpua = data["HRR (kW)"] / area
    notes = metadata["Data Corrections"]

    peak = hrrpua.max()
    thr = (data["HRR (kW)"].sum() / area) / 1000  # 1 s scans: sum of kW/m2 = kJ/m2
    if reported["peak_hrr"] and abs(peak - reported["peak_hrr"]) > 0.01 * reported["peak_hrr"]:
        notes.append(f"Peak HRRPUA from data ({peak:.1f}) differs from reported ({reported['peak_hrr']}) by more than 1%")
    if reported["thr"] and abs(thr - reported["thr"]) > 0.01 * reported["thr"]:
        notes.append(f"THR from data ({thr:.2f} MJ/m2) differs from reported ({reported['thr']}) by more than 1%")
    print(f"\tCheck: peak {peak:.1f} kW/m2 (reported {reported['peak_hrr']}), THR {thr:.2f} MJ/m2 (reported {reported['thr']})")

    # Recalculate HRR from O2, exhaust pressure and stack temperature with the shared utils
    # function. The Atlas HRR column matches the O2-only equation (within ~2%) with no extra
    # time shift, which suggests the exported gas columns are already delay-corrected.
    c, e, x_o2 = metadata["C Factor"], metadata["Heat of Combustion O2 (MJ/kg)"], metadata["X_O2 Initial"]
    if None not in (c, e, x_o2):
        recalc = pd.Series([
            calculate_HRR_O2_only(o2, x_o2, dp, te, c, e, area) if dp > 0 else np.nan
            for o2, dp, te in zip(data["O2 (Vol fr)"], extra["dP"], extra["Te"])
        ])
        recalc_peak = recalc.max()
        print(f"\tCheck: recalculated (O2 only) peak {recalc_peak:.1f} kW/m2")
        if abs(recalc_peak - peak) > 0.05 * peak:
            notes.append(f"Recalculated O2-only peak HRRPUA ({recalc_peak:.1f}) differs from the instrument's ({peak:.1f}) by more than 5%")

    if reported["t_ign_raw"] is not None and reported["t_ign_raw"] != metadata["t_ignition (s)"]:
        notes.append(f"t_ignition rounded from {reported['t_ign_raw']} s")


if __name__ == "__main__":
    parse_dir(INPUT_DIR)
