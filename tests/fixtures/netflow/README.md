Real NetFlow exports used by `tests/test_netflow.py`.

- `source_traffic.pcap`: synthetic traffic (24 short HTTP sessions 192.168.10.5 -> 192.168.10.50, one
  retransmission every third session, 24 DNS replies from 8.8.8.8), generated with Scapy.
- `softflowd_v5.pcap`, `softflowd_v9.pcap`, `softflowd_v10.pcap`: the UDP export datagrams that
  softflowd 1.x sent when replaying `source_traffic.pcap` (`softflowd -r source_traffic.pcap -n
  127.0.0.1:PORT -v 5|9|10`), wrapped in a capture as a collector would see them.
- `nfdump_v9.csv`: the v9 export collected by nfcapd and printed with `nfdump -R <dir> -o csv`
  (nfdump 1.7.3), including nfdump's trailing summary block.

softflowd in offline (-r) mode stamps flows relative to the export time, not the capture time, so
flow timestamps are October 2026 rather than 2017.
