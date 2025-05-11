import sys
import pydot
import networkx as nx
from packet_utils import get_packet_names_from_pcap_file


class MealyMachine:
    def __init__(self, dot_file):
        self.transitions, self.start_state, self.accepting_states = self.parse_dot_file(dot_file)
        self.current_state = self.start_state

    def parse_dot_file(self, dot_file):
        """
        Parses the specified .dot file (Graphviz) and returns:
          - transitions: dict  { (src_state, event): (dst_state, output), ... }
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
                event, output = label.strip('"').split(" / ")
                transitions[(src, event)] = (dst, output)

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

    def reset(self):
        self.current_state = self.start_state

    def process_event(self, event):
        """
        Processes an event and returns the output.
        """
        if (self.current_state, event) in self.transitions:
            next_state, output = self.transitions[(self.current_state, event)]
            self.current_state = next_state
            return output
        else:
            raise ValueError(f"Invalid transition: state={self.current_state}, event={event}")

    def is_accepting_state(self):
        return self.current_state in self.accepting_states

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace.pcap>")
        sys.exit(1)

    dot_file = "dataset/signatures/mm/nas.dot"
    mealy_machine = MealyMachine(dot_file)
    
    pcap_file = sys.argv[1]

    events = get_packet_names_from_pcap_file(pcap_file)
    print(f"[INFO] Extracted {len(events)} NAS events from {pcap_file}")

    for event in events:
        try:
            output = mealy_machine.process_event(event)
            print(f"Event: {event}, Output: {output}")
        except ValueError as e:
            print(e)
            break

    if mealy_machine.is_accepting_state():
        print("The machine ended in an accepting state.")
    else:
        print("The machine did not end in an accepting state.")