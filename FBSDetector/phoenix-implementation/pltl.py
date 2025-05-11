import os
import sys
from packet_utils import get_packet_names_from_pcap_file

def parse_pltl_signatures(pltl_folder):
    """
    Parses the PLTL signatures from the specified folder.
    Returns a list of PLTL formulas.
    """
    pltl_signatures = []
    for filename in os.listdir(pltl_folder):
        filepath = os.path.join(pltl_folder, filename)
        with open(filepath, 'r') as file:
            content = file.read().strip()
            if content.startswith('(!(') and content.endswith('))'):
                pltl_signatures.append(content[2:-2])
            else:
                print(f"Invalid PLTL signature format in {filename}")
    return pltl_signatures

def check_pltl_signatures(pltl_signatures, events):
    """
    Checks the events against the PLTL signatures.
    Returns a list of violations (each is a dict with details).
    """
    violations = []
    for i, signature in enumerate(pltl_signatures):
        # Convert the PLTL signature to a Python expression
        expression = signature.replace('&&', ' and ').replace('||', ' or ').replace('!', ' not ')
        try:
            if not eval(expression):
                violations.append({
                    "index": i,
                    "signature": signature,
                    "message": f"PLTL violation: {signature}"
                })
        except Exception as e:
            violations.append({
                "index": i,
                "signature": signature,
                "message": f"Error evaluating PLTL signature: {e}"
            })
    return violations

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]}  <trace.pcap>")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    pltl_folder = "dataset/signatures/pltl/NAS/"

    events = get_packet_names_from_pcap_file(pcap_file)
    print(f"[INFO] Extracted {len(events)} NAS events from {pcap_file}")

    # Parse PLTL signatures
    pltl_signatures = parse_pltl_signatures(pltl_folder)
    print(f"[INFO] Parsed {len(pltl_signatures)} PLTL signatures from {pltl_folder}")

    # Check PLTL signatures
    violations = check_pltl_signatures(pltl_signatures, events)
    if violations:
        print(f"[ALERT] Found {len(violations)} PLTL violations:")
        for v in violations:
            print(f"  - Index {v['index']}, Signature={v['signature']}, Msg={v['message']}")
    else:
        print("[OK] No PLTL violations detected! The trace satisfies all PLTL signatures.")

if __name__ == "__main__":
    main()