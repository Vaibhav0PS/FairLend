"""Shared column definitions so training and serving can't drift apart."""

TARGET = "loan_status"

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

HOME_OWNERSHIP = ("RENT", "MORTGAGE", "OWN", "OTHER")
LOAN_INTENTS = (
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
)
LOAN_GRADES = ("A", "B", "C", "D", "E", "F", "G")
DEFAULT_FLAGS = ("Y", "N")

# Age limits applied when cleaning training data; the API enforces the same range.
MIN_AGE = 18
MAX_AGE = 100
MAX_EMP_LENGTH = 60
