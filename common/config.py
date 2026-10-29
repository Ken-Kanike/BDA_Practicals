"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"

BUS_CSV = DATA_DIR / "bus_delays.csv"
TRAIN_CSV = DATA_DIR / "train_delays.csv"
METADATA_JSON = DATA_DIR / "metadata.json"

BOROUGHS = ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"]
TRAIN_LINES = [
    "Harlem",
    "Hudson Line",
    "New Haven",
    "East of Hudson",
    "Pascack Valley",
    "Port Jervis",
    "Bergen County",
]

# Exact reason counts keep the MapReduce keyword report reproducible.
BUS_REASONS = [
    ("Heavy Traffic", 14177),
    ("Mechanical Problem", 7333),
    ("Late Return From Route", 5967),
    ("Road Construction", 5091),
    ("Weather Conditions", 5025),
    ("Accident", 4500),
    ("Driver Shortage", 4000),
    ("Other", 3907),
]
TRAIN_REASONS = [
    ("Signal Problem", 9991),
    ("Mechanical Failure", 8979),
    ("Track Work", 7581),
    ("Congestion", 6017),
    ("Weather Conditions", 5909),
    ("Overhead Line Issue", 4019),
    ("Door Fault", 3800),
    ("Other", 3704),
]

BUS_COLUMNS = [
    "Date",
    "Route_ID",
    "Borough",
    "Scheduled_Time",
    "Reason",
    "Delay_Minutes",
    "Number_Of_Students_On_Bus",
]
TRAIN_COLUMNS = [
    "Date",
    "Line",
    "Scheduled_Time",
    "Reason",
    "Delay_Minutes",
    "Passengers_Onboard",
]


def output_dir(experiment: str) -> Path:
    """Return (and create) the output directory for one experiment."""
    path = OUTPUTS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def portable_text(text: str) -> str:
    """Replace machine-specific absolute project paths with a placeholder."""
    for variant in (ROOT.as_uri(), ROOT.as_posix(), str(ROOT)):
        text = text.replace(variant, "<PROJECT_ROOT>")
    return text


def rel(path: Path) -> str:
    """Render a project path as a forward-slash path relative to the repo root."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dirs() -> None:
    for path in (DATA_DIR, OUTPUTS_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
