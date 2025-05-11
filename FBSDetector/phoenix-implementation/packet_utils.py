import os
import xml.etree.ElementTree as ET

def get_packet_names(xmlfile):
    packet_names = []
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    nas_msg_emm_fields = root.findall('.//field[@name="nas-eps.nas_msg_emm_type"]')

    for f in nas_msg_emm_fields:
        packet_names.append(f.get('showname').split(":")[-1].split("(")[0].strip().lower().replace(" ", "_"))

    return packet_names

def get_packet_names_from_pcap_file(input_file):
    xml_output_file = input_file.replace("pcap", "xml")
    os.system("tshark -r " + input_file + " -T pdml > " + xml_output_file)
    packet_names = get_packet_names(xml_output_file)
    return packet_names