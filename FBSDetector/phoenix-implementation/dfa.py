#!/usr/bin/env python3

import sys
import os
import subprocess
import xml.etree.ElementTree as ET
import pydot
import networkx as nx
from scapy.all import rdpcap
from packet_utils import get_packet_names_from_pcap_file

############################################################
# 1. PARSE THE DOT FILE AND BUILD THE STATE MACHINE
############################################################

def parse_dot_file(dot_file):
    """
    Parses the specified .dot file (Graphviz) and returns:
      - transitions: dict  { (src_state, event): dst_state, ... }
      - start_state: the initial state (derived from __start0)
      - accepting_states: list of states with shape=doublecircle
    """
    # Load the graph(s) from the .dot file
    graphs = pydot.graph_from_dot_file(dot_file)
    if not graphs:
        raise ValueError(f"No valid graphs found in {dot_file}.")
    graph = graphs[0]

    # Convert to networkx for easier traversal
    nx_graph = nx.drawing.nx_pydot.from_pydot(graph)

    # Build transitions dict
    transitions = {}
    for edge in nx_graph.edges(data=True):
        src = edge[0]
        dst = edge[1]
        label = edge[2].get('label')
        if label:
            label = label.strip('"')
            transitions[(src, label)] = dst

    # Find the start state by looking for an edge from __start0
    start_state = None
    for edge in nx_graph.edges(data=True):
        src, dst = edge[0], edge[1]
        if src.startswith("__start"):
            start_state = dst
            break
    if not start_state:
        raise ValueError("Could not find a start state (edge from __start0).")

    # Identify accepting states (i.e., shape=doublecircle)
    accepting_states = []
    for node, attr in nx_graph.nodes(data=True):
        shape = attr.get('shape', '')
        if shape == 'doublecircle':
            accepting_states.append(node)

    return transitions, start_state, accepting_states

############################################################
# 3. USE THE STATE MACHINE TO DETECT ATTACKS / ANOMALIES
############################################################

def detect_anomalies(transitions, start_state, accepting_states, events):
    """
    Replays the list of events over the state machine. If any transition
    isn't defined in the graph, we flag it as an anomaly (possible attack).
    Returns a list of anomalies (each is a dict with details).
    """

    anomalies = []
    current_state = start_state
    for i, event in enumerate(events):
        # Check if we have a valid transition
        if (current_state, event) in transitions:
            # Transition to the next state
            next_state = transitions[(current_state, event)]
            # For debugging/logging:
            # print(f"{i}: {current_state} --({event})--> {next_state}")
            current_state = next_state
        else:
            # This event doesn't match any transition from the current state
            anomaly = {
                "index": i,
                "current_state": current_state,
                "event": event,
                "message": f"Invalid transition: state={current_state}, event={event}",
            }
            anomalies.append(anomaly)
            # Depending on your approach, you might break or attempt to continue
            # We'll continue to see if there are more issues
            # break

    # Optionally, check if final state is in an “accepting” set or not
    if current_state not in accepting_states:
        anomalies.append({
            "index": len(events),
            "current_state": current_state,
            "event": None,
            "message": "Ended in a non-accepting state",
        })

    return anomalies

############################################################
# 4. MAIN LOGIC: Putting It All Together
############################################################

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <state_machine.dot> <trace.pcap>")
        sys.exit(1)
    
    dot_file = sys.argv[1]
    pcap_file = sys.argv[2]

    # 1. Parse the state machine from DOT
    transitions, start_state, accepting_states = parse_dot_file(dot_file)
    print(f"[INFO] Start State: {start_state}")
    print(f"[INFO] Accepting States: {accepting_states}")

    # 2. Parse events from the pcap
    events = get_packet_names_from_pcap_file(pcap_file)
    print(f"[INFO] Extracted {len(events)} NAS events from {pcap_file}")

    # 3. Run detection
    anomalies = detect_anomalies(transitions, start_state, accepting_states, events)
    if anomalies:
        print(f"[ALERT] Found {len(anomalies)} anomalies or attacks:")
        for a in anomalies:
            print(f"  - Index {a['index']}, State={a['current_state']}, Event={a['event']}, Msg={a['message']}")
    else:
        print("[OK] No anomalies detected! The trace follows the automaton.")

if __name__ == "__main__":
    main()
