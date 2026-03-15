import os
import csv
from scapy.all import PcapReader
from scapy.layers.inet import IP

DEVICE_IP = "10.63.7.79"

def process_pcap_to_csv(pcap_path, output_csv_path):
    rows = []
    serial_number = 1

    with PcapReader(pcap_path) as packets:
        for pkt in packets:
            if IP not in pkt:
                continue

            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst

            if src_ip != DEVICE_IP and dst_ip != DEVICE_IP:
                continue

            pkt_time = float(pkt.time)
            pkt_size = len(pkt)
            direction = 1 if src_ip == DEVICE_IP else -1

            rows.append([serial_number, pkt_time, pkt_size, direction])
            serial_number += 1

    if not rows:
        return

    start_time = rows[0][1]
    for row in rows:
        row[1] = row[1] - start_time

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Serial number", "time", "size", "direction"])
        writer.writerows(rows)

    print(f"{os.path.basename(pcap_path)} -> {len(rows)} packets written")

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in sorted(os.listdir(input_folder)):
        if file_name.lower().endswith(".pcap"):
            pcap_path = os.path.join(input_folder, file_name)
            csv_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".csv")
            process_pcap_to_csv(pcap_path, csv_path)

if __name__ == "__main__":
    process_folder(".", "full_csv")
