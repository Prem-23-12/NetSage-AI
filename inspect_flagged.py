import pandas as pd

df = pd.read_csv("ai_diagnoses.csv")
flagged = df[df["grounded"] == False][["case_id", "ai_evidence", "unsupported_tokens"]]

for row in flagged.itertuples():
    print(f"{row.case_id}: unsupported={row.unsupported_tokens}")
    print(f"  evidence: {row.ai_evidence}\n")
