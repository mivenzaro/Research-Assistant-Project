import os
import csv
from scapy.all import PcapReader
from scapy.layers.inet import IP

def process_pcap_to_csv(pcap_path, output_csv_path):
    rows = []
    serial_number = 1
    forward_src = None

    with PcapReader(pcap_path) as packets:
        for pkt in packets:
            pkt_time = float(pkt.time)
            pkt_size = len(pkt)

            if IP in pkt:
                src_ip = pkt[IP].src
                if forward_src is None:
                    forward_src = src_ip
                direction = 1 if src_ip == forward_src else -1
            else:
                direction = 1

            rows.append([serial_number, pkt_time, pkt_size, direction])
            serial_number += 1

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Serial number", "time", "size", "direction"])
        writer.writerows(rows)

    print(f"{os.path.basename(pcap_path)} -> {len(rows)} packets written")

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith((".pcap", ".pcapng", ".cap")):
            pcap_path = os.path.join(input_folder, file_name)
            csv_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".csv")
            process_pcap_to_csv(pcap_path, csv_path)

if __name__ == "__main__":
    input_folder = "."
    output_folder = "."
    process_folder(input_folder, output_folder)
