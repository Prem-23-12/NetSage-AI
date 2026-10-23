"""
NetSage AI - Deterministic Rule Checker

Runs pattern-based checks over each case's `show_output` BEFORE the AI ever
sees it. This is the "ground truth" layer: no LLM involved, just regex and
simple comparisons against the `rule_hint` metadata (expected VLAN, expected
gateway, expected route, etc.) that describes design intent for each case.

Checks implemented (per the project spec):
  1. Duplicate IP addresses
  2. Wrong subnet mask
  3. Gateway mismatch
  4. Interface administratively/operationally down
  5. Missing VLAN (not in `show vlan brief`, or shown Inactive)
  6. Missing route (destination network absent from `show ip route`)

Usage:
    python3 rule_checker.py                # runs on data/cases.csv, prints summary
    from rule_checker import run_all_checks # import and use programmatically
"""
import re
import pandas as pd
from collections import defaultdict


def parse_hints(hint_str):
    """rule_hint is semicolon-separated key:value pairs, e.g.
    'vlan_expected:30;trunk_check:Gi0/1'"""
    hints = {}
    if not isinstance(hint_str, str):
        return hints
    for part in hint_str.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            hints[k.strip()] = v.strip()
    return hints


def check_interface_down(show_output):
    """Flags interfaces reported down (administratively or operationally)."""
    findings = []
    for line in show_output.splitlines():
        if re.search(r"\bdown\b", line, re.IGNORECASE) and (
            "administratively down" in line.lower()
            or re.search(r"\bdown\s+down\b", line.lower())
        ):
            findings.append(line.strip())
    return findings


def check_missing_vlan(show_output, hints):
    """Compares vlan_expected against VLANs present/active in `show vlan brief`."""
    expected = hints.get("vlan_expected")
    if not expected:
        return None
    vlan_lines = [l for l in show_output.splitlines() if re.match(r"^\s*\d+\s+\S", l)]
    present_ids = {re.match(r"^\s*(\d+)", l).group(1) for l in vlan_lines}
    inactive = any(f"{expected} " in l or l.strip().startswith(expected)
                   for l in show_output.splitlines() if "Inactive" in l)
    if expected not in present_ids:
        return f"VLAN {expected} not found in VLAN database"
    if inactive:
        return f"VLAN {expected} present but marked Inactive"
    return None


def check_trunk_allowed_vlan(show_output, hints):
    trunk_port = hints.get("trunk_check")
    vlan_expected = hints.get("vlan_expected")
    if not trunk_port or not vlan_expected:
        return None
    m = re.search(rf"{re.escape(trunk_port)}\s+([\d,]+)", show_output)
    if m:
        allowed = m.group(1).split(",")
        if vlan_expected not in allowed:
            return f"VLAN {vlan_expected} not in allowed-VLAN list on trunk {trunk_port} (allowed: {m.group(1)})"
    return None


def check_gateway_mismatch(show_output, hints):
    pc_gw = hints.get("gateway_pc")
    expected_gw = hints.get("gateway_expected")
    if pc_gw and expected_gw and pc_gw != expected_gw:
        return f"PC default gateway {pc_gw} does not match router interface {expected_gw}"
    return None


def check_duplicate_ip(show_output, hints):
    """Only fires when the case explicitly flags a duplicate_ip_check target.
    A naive 'any IP seen twice' scan produces false positives on perfectly
    normal output (NAT translation rows and static routes both legitimately
    repeat an address), so this stays targeted rather than generic."""
    target = hints.get("duplicate_ip_check")
    if not target:
        return None
    occurrences = show_output.count(target)
    if occurrences >= 2:
        return f"IP {target} assigned to more than one device"
    return None


def check_wrong_mask(show_output, hints):
    mask_info = hints.get("mask_check")
    expected = hints.get("expected_mask")
    if mask_info and expected:
        device, actual_mask = mask_info.split(":")
        if actual_mask != expected:
            return f"{device} subnet mask {actual_mask} does not match expected {expected}"
    return None


def check_missing_route(show_output, hints):
    expected_net = hints.get("route_expected")
    if not expected_net:
        return None
    if expected_net not in show_output:
        return f"Expected route to {expected_net} not present in routing table"
    return None


def run_all_checks(row):
    """Runs every applicable deterministic check on a single case row.
    Returns a list of finding strings (empty list = nothing flagged)."""
    show_output = row["show_output"]
    hints = parse_hints(row.get("rule_hint", ""))
    findings = []

    down = check_interface_down(show_output)
    if down:
        findings.append(f"[Interface Down] {down[0]}")

    vlan_issue = check_missing_vlan(show_output, hints)
    if vlan_issue:
        findings.append(f"[VLAN] {vlan_issue}")

    trunk_issue = check_trunk_allowed_vlan(show_output, hints)
    if trunk_issue:
        findings.append(f"[Trunk] {trunk_issue}")

    gw_issue = check_gateway_mismatch(show_output, hints)
    if gw_issue:
        findings.append(f"[Gateway] {gw_issue}")

    dup_issue = check_duplicate_ip(show_output, hints)
    if dup_issue:
        findings.append(f"[Duplicate IP] {dup_issue}")

    mask_issue = check_wrong_mask(show_output, hints)
    if mask_issue:
        findings.append(f"[Subnet Mask] {mask_issue}")

    route_issue = check_missing_route(show_output, hints)
    if route_issue:
        findings.append(f"[Routing] {route_issue}")

    return findings


def main():
    df = pd.read_csv("D:\\Cisco\\Netsage AI\\cases.csv")
    results = []
    hit_count = 0
   
    for _, row in df.iterrows():
        findings = run_all_checks(row)
        if findings:
            hit_count += 1
        results.append({
            "case_id": row["case_id"],
            "category": row["category"],
            "rule_findings": " | ".join(findings) if findings else "No deterministic findings",
            "n_findings": len(findings),
        })

    out = pd.DataFrame(results)
    out.to_csv("D:\\Cisco\\Netsage AI\\rule_findings.csv", index=False)

    print(f"Rule checker ran on {len(df)} cases")
    print(f"Cases with at least one deterministic finding: {hit_count}/{len(df)}")
    print()
    print(out.to_string(index=False, max_colwidth=90))


if __name__ == "__main__":
    main()
