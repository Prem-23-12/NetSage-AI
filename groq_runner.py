"""
NetSage AI - AI Diagnosis Runner

For every case in data/cases.csv:
  1. Pulls the matching deterministic rule findings (from data/rule_findings.csv)
  2. Builds the grounded prompt (case evidence + rule findings)
  3. Calls Groq's free API (Llama 3.3 70B) for a structured JSON diagnosis
  4. Runs an automated grounding check: does every IP / VLAN / interface the
     AI cites in its `evidence` field actually appear in the raw show_output?
     If not, it's auto-flagged as a possible unsupported claim BEFORE any
     human review happens.
  5. Saves everything to data/ai_diagnoses.csv

Usage:
    export GROQ_API_KEY="gsk_..."          # your free key from console.groq.com
    pip install requests --break-system-packages
    python3 groq_runner.py                  # real run against Groq
    python3 groq_runner.py --mock           # offline dry-run, no API key needed,
                                             # proves the pipeline logic end-to-end
"""
import os
import re
import json
import time
import argparse
import pandas as pd
import requests

PROMPT_PATH = os.path.join("prompt", "diagnose_prompt.md")
CASES_PATH = os.path.join("data", "cases.csv")
FINDINGS_PATH = os.path.join("data", "rule_findings.csv")
OUT_PATH = os.path.join("data", "ai_diagnoses.csv")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70b-versatile was deprecated by Groq in June 2026. Using the
# official migration target explicitly rather than relying on whatever
# Groq silently falls back to for a retired model name.
MODEL = "openai/gpt-oss-20b"
REQUEST_DELAY_SECONDS = 8  # paces calls to stay under the free-tier TPM cap

SYSTEM_PROMPT = """You are NetSage AI, a Cisco network troubleshooting assistant operating under a strict human-in-the-loop policy: you propose a diagnosis, but a human network engineer always makes the final call. Rules you must follow:

1. Base every claim strictly on the evidence provided (symptom, topology note, show-command output, and any deterministic rule-engine findings). Never invent commands, IP addresses, VLAN numbers, or config lines that are not present in the input.
2. If deterministic rule-engine findings are supplied, treat them as verified ground truth. Your root_cause must be consistent with them.
3. If no rule-engine findings are supplied, reason carefully from the raw show output alone, and lower your confidence accordingly.
4. Respond with ONLY a single valid JSON object. No markdown fences, no prose before or after it. Schema:
{"root_cause": "string", "osi_layer": "string", "evidence": "string", "next_command": "string", "fix_steps": "string", "confidence": "High | Medium | Low"}
"""


def build_user_prompt(row, rule_finding_text):
    return f"""Symptom: {row['symptom']}
Topology note: {row['topology_note']}
Show output:
{row['show_output']}

Rule engine findings: {rule_finding_text}

Respond with the JSON object only."""


def _extract_wait_seconds(error_text):
    """Groq's 429 body says e.g. 'try again in 247.5ms' or 'in 3.2s'.
    Parse that out so we wait exactly as long as needed, not a guess."""
    m = re.search(r"try again in ([\d.]+)(ms|s)\b", error_text)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    return value / 1000 if unit == "ms" else value


def call_groq(user_prompt, api_key, retries=3):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "reasoning_effort": "low",  # gpt-oss is a reasoning model; hidden
                                    # reasoning tokens count against the TPM
                                    # cap, so keep this low for a batch job
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except requests.exceptions.HTTPError as e:
            last_err = e
            body = e.response.text if e.response is not None else str(e)
            if e.response is not None and e.response.status_code == 429:
                wait = _extract_wait_seconds(body) or 5
                time.sleep(wait + 1)  # small buffer past what Groq asked for
            else:
                # e.g. json_validate_failed - the model occasionally fails to
                # close out valid JSON under reasoning constraints. A plain
                # retry with a fresh generation usually succeeds.
                time.sleep(3 * (attempt + 1))
        except Exception as e:
            last_err = e
            if "resp" in locals():
                try:
                    last_err = Exception(f"{e} | Groq response: {resp.text[:1000]}")
                except Exception:
                    pass
            time.sleep(3 * (attempt + 1))

    # Final fallback: drop the strict JSON response_format and ask in plain
    # text, then pull the first {...} block out manually. Catches the rare
    # case where json_object mode itself is what's failing.
    try:
        payload.pop("response_format", None)
        payload["messages"][-1]["content"] += "\n\nReturn ONLY the JSON object, nothing else."
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        last_err = e

    return {"error": str(last_err)}


def call_mock(row):
    """Offline stand-in so the pipeline can be proven correct without a live key."""
    return {
        "root_cause": f"[MOCK] Likely issue related to {row['concept_tag']}",
        "osi_layer": row["osi_layer"],
        "evidence": f"Mock evidence referencing {row['show_output'].splitlines()[1][:40]}",
        "next_command": "show running-config",
        "fix_steps": "[MOCK] placeholder fix steps",
        "confidence": "Medium",
    }


IFACE_PREFIX_MAP = {"Gi": "GigabitEthernet", "Fa": "FastEthernet", "Se": "Serial"}


def normalize_iface(token):
    """Cisco engineers freely mix Gi0/0.10, gi0/0.10 (as typed in a command),
    and GigabitEthernet0/0.10 - same interface, different notation/case.
    Normalize to lowercase long form so none of that trips a false
    hallucination flag."""
    m = re.match(r"([A-Za-z]+)([0-9./]+)", token)
    if not m:
        return token.lower()
    prefix, num = m.groups()
    prefix_lower = prefix.lower()
    for short, long in IFACE_PREFIX_MAP.items():
        if prefix_lower.startswith(short.lower()):
            return (long + num).lower()
    return (prefix + num).lower()


def extract_plain_tokens(text):
    """IPs and VLAN numbers - these have one canonical written form, so a
    plain substring check is reliable."""
    ips = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
    vlans = re.findall(r"VLAN\s*(\d+)", text, re.IGNORECASE)
    return set(ips) | set(vlans)


def extract_interfaces(text):
    raw = re.findall(r"\b(?:Gi|Fa|Se|GigabitEthernet|FastEthernet|Serial)[0-9./]+\b", text, re.IGNORECASE)
    return set(normalize_iface(t) for t in raw)


def grounding_check(evidence_text, context_text):
    """Returns (is_grounded: bool, unsupported_tokens: list).
    Flags any IP/VLAN/interface the AI cited in its evidence that does not
    appear anywhere in the FULL context it was actually given - symptom,
    topology note, rule-engine findings, and show output combined, not just
    the raw show output alone. A citation from the topology note or from the
    rule engine's own finding text is legitimate grounding, not a
    hallucination, since that text was part of the model's input too.
    Interface names are matched case-insensitively and normalized so
    'Gi0/1' / 'gi0/1' / 'GigabitEthernet0/1' are all treated as the same
    interface."""
    cited_plain = extract_plain_tokens(evidence_text)
    cited_iface = extract_interfaces(evidence_text)
    if not cited_plain and not cited_iface:
        return True, []

    unsupported = [t for t in cited_plain if t not in context_text]

    context_ifaces_normalized = extract_interfaces(context_text)
    unsupported += [t for t in cited_iface if t not in context_ifaces_normalized]

    return (len(unsupported) == 0), unsupported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="offline dry run, no API key needed")
    parser.add_argument("--limit", type=int, default=None, help="only process first N cases (useful while testing)")
    parser.add_argument("--retry-failed", action="store_true",
                         help="only re-run cases that errored last time, keep the rest from the existing ai_diagnoses.csv")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not args.mock and not api_key:
        raise SystemExit(
            "No GROQ_API_KEY found in environment.\n"
            "  export GROQ_API_KEY='gsk_...'   (get a free one at console.groq.com)\n"
            "Or run with --mock to dry-run the pipeline without a key."
        )

    cases = pd.read_csv("D:\\Cisco\\Netsage AI\\cases.csv")
    findings = pd.read_csv("D:\\Cisco\\Netsage AI\\rule_findings.csv")
    merged = cases.merge(findings[["case_id", "rule_findings"]], on="case_id", how="left")

    existing_good = None
    if args.retry_failed:
        if not os.path.exists(OUT_PATH):
            raise SystemExit(f"No existing {OUT_PATH} found to retry against - run a normal batch first.")
        prev = pd.read_csv(OUT_PATH)
        failed_ids = set(prev.loc[prev.get("error", pd.Series(dtype=object)).notna(), "case_id"]) if "error" in prev.columns else set()
        # also catch rows that succeeded structurally but never got a root cause
        if "ai_root_cause" in prev.columns:
            failed_ids |= set(prev.loc[prev["ai_root_cause"].isna(), "case_id"])
        existing_good = prev[~prev["case_id"].isin(failed_ids)]
        merged = merged[merged["case_id"].isin(failed_ids)]
        print(f"Retrying {len(merged)} previously failed case(s): {sorted(failed_ids)}")

    if args.limit:
        merged = merged.head(args.limit)

    results = []
    for i, row in merged.iterrows():
        rule_text = row["rule_findings"] if pd.notna(row["rule_findings"]) else "No deterministic findings"

        if args.mock:
            ai_out = call_mock(row)
        else:
            prompt = build_user_prompt(row, rule_text)
            ai_out = call_groq(prompt, api_key)

        if "error" in ai_out:
            results.append({"case_id": row["case_id"], "error": ai_out["error"]})
            print(f"{row['case_id']}: ERROR - {ai_out['error']}")
            if not args.mock:
                time.sleep(REQUEST_DELAY_SECONDS)
            continue

        rule_text_str = rule_text if isinstance(rule_text, str) else ""
        full_context = " ".join([
            str(row.get("show_output", "")),
            str(row.get("topology_note", "")),
            str(row.get("symptom", "")),
            rule_text_str,
        ])
        grounded, unsupported = grounding_check(ai_out.get("evidence", ""), full_context)

        results.append({
            "case_id": row["case_id"],
            "category": row["category"],
            "rule_findings": rule_text,
            "ai_root_cause": ai_out.get("root_cause"),
            "ai_osi_layer": ai_out.get("osi_layer"),
            "ai_evidence": ai_out.get("evidence"),
            "ai_next_command": ai_out.get("next_command"),
            "ai_fix_steps": ai_out.get("fix_steps"),
            "ai_confidence": ai_out.get("confidence"),
            "expected_fault": row["expected_fault"],
            "grounded": grounded,
            "unsupported_tokens": ", ".join(unsupported) if unsupported else "",
        })
        print(f"{row['case_id']}: {ai_out.get('root_cause', '')[:70]}  [grounded={grounded}]")

        if not args.mock:
            time.sleep(REQUEST_DELAY_SECONDS)  # stay under the free-tier TPM cap

    out_df = pd.DataFrame(results)
    if existing_good is not None:
        out_df = pd.concat([existing_good, out_df], ignore_index=True).sort_values("case_id")

    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out_df)} AI diagnoses to {OUT_PATH}")
    if "grounded" in out_df.columns:
        # `grounded` can end up as object dtype (not clean bool) after a
        # round-trip through CSV mixed with NaN error rows - casting to
        # bool explicitly before negating avoids Python's bitwise ~ on ints.
        grounded_bool = out_df["grounded"].fillna(True).astype(bool)
        n_flagged = (~grounded_bool).sum()
        print(f"Auto-flagged as possibly ungrounded: {n_flagged}/{len(out_df)}")
    if "error" in out_df.columns:
        n_errors = out_df["error"].notna().sum()
        if n_errors:
            print(f"Still failing: {n_errors} case(s). Run again with --retry-failed.")


if __name__ == "__main__":
    main()
