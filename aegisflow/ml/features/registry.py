"""
Feature registry: documents every feature across canonical flows, host-level
temporal windows, and sequence modeling representations.

Provides machine-readable metadata:
- feature_name
- data_type
- source_field(s)
- calculation
- required
- available
- aggregation_level ("flow", "host-window", "sequence")
- description

Features structurally absent in the source dataset (e.g. TTL, retransmissions in CIC-IDS2017)
are explicitly registered as available=False, never fabricated or silently zero-filled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    data_type: str
    source_fields: str
    calculation: str
    required: bool
    available: bool
    aggregation_level: str  # "flow" | "host-window" | "sequence"
    description: str
    available_in: tuple[str, ...] = ()

    # Backward-compatibility aliases for Phase 1 code/tests
    @property
    def name(self) -> str:
        return self.feature_name

    @property
    def dtype(self) -> str:
        return self.data_type

    @property
    def source(self) -> str:
        return self.source_fields

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "data_type": self.data_type,
            "source_fields": self.source_fields,
            "calculation": self.calculation,
            "required": self.required,
            "available": self.available,
            "aggregation_level": self.aggregation_level,
            "description": self.description,
            "available_in": list(self.available_in),
        }


FEATURE_REGISTRY: list[FeatureSpec] = [
    # =========================================================================
    # FLOW LEVEL - RAW & CANONICAL ADAPTER FIELDS
    # =========================================================================
    FeatureSpec(
        feature_name="timestamp",
        data_type="datetime64[ns]",
        source_fields="CICFlowMeter: Timestamp",
        calculation="Parsed timestamp coerced to UTC datetime64[ns].",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Flow start time recorded by network sensor.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="source_ip",
        data_type="string",
        source_fields="CICFlowMeter: Source IP",
        calculation="Verbatim string representation of source IPv4/IPv6.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Originating host address for the flow.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="destination_ip",
        data_type="string",
        source_fields="CICFlowMeter: Destination IP",
        calculation="Verbatim string representation of destination IPv4/IPv6.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Destination host address for the flow.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="source_port",
        data_type="Int64",
        source_fields="CICFlowMeter: Source Port",
        calculation="Nullable 16-bit integer port.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Transport-layer source port number.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="destination_port",
        data_type="Int64",
        source_fields="CICFlowMeter: Destination Port",
        calculation="Nullable 16-bit integer port.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Transport-layer destination port number (service identifier).",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="protocol",
        data_type="string",
        source_fields="CICFlowMeter: Protocol",
        calculation="IANA protocol number mapped to name (6->TCP, 17->UDP); raw number preserved if unmapped.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Transport protocol identifier.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="flow_duration",
        data_type="float64 (seconds)",
        source_fields="CICFlowMeter: Flow Duration",
        calculation="Raw duration in microseconds divided by 1,000,000.0.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Total duration of the bidirectional flow in seconds.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="packet_count",
        data_type="float64",
        source_fields="fwd_packet_count, bwd_packet_count",
        calculation="fwd_packet_count + bwd_packet_count.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Total number of packets observed in both directions.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="byte_count",
        data_type="float64",
        source_fields="fwd_byte_count, bwd_byte_count",
        calculation="fwd_byte_count + bwd_byte_count.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Total payload/header volume in bytes across both directions.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="dataset_label",
        data_type="string",
        source_fields="CICFlowMeter: Label",
        calculation="Verbatim raw label string emitted by source dataset.",
        required=True,
        available=True,
        aggregation_level="flow",
        description="Ground-truth annotation provided with the raw capture.",
        available_in=("cic_ids2017",),
    ),

    # =========================================================================
    # FLOW LEVEL - PACKET SIZE & TIMING STATISTICS
    # =========================================================================
    FeatureSpec(
        feature_name="packet_size_mean",
        data_type="float64",
        source_fields="CICFlowMeter: Packet Length Mean",
        calculation="Verbatim mean packet size in bytes.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Mean packet length over the flow lifetime.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="packet_size_std",
        data_type="float64",
        source_fields="CICFlowMeter: Packet Length Std",
        calculation="Verbatim standard deviation of packet lengths.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Variability of packet lengths in the flow.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="packet_size_min",
        data_type="float64",
        source_fields="CICFlowMeter: Min Packet Length",
        calculation="Verbatim minimum packet size in bytes.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Smallest packet observed in the flow.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="packet_size_max",
        data_type="float64",
        source_fields="CICFlowMeter: Max Packet Length",
        calculation="Verbatim maximum packet size in bytes.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Largest packet observed in the flow.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="inter_arrival_mean",
        data_type="float64",
        source_fields="CICFlowMeter: Flow IAT Mean",
        calculation="Verbatim mean inter-arrival time in microseconds/seconds.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Mean time between successive packet arrivals.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="inter_arrival_std",
        data_type="float64",
        source_fields="CICFlowMeter: Flow IAT Std",
        calculation="Verbatim standard deviation of inter-arrival times.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Standard deviation of packet arrival spacing.",
        available_in=("cic_ids2017",),
    ),

    # =========================================================================
    # FLOW LEVEL - TCP CONTROL FLAGS & WINDOW STATS
    # =========================================================================
    FeatureSpec(
        feature_name="syn_count",
        data_type="float64",
        source_fields="CICFlowMeter: SYN Flag Count",
        calculation="Verbatim count of SYN flags observed in flow.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Total SYN packets in the flow (connection establishment indicator).",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="ack_count",
        data_type="float64",
        source_fields="CICFlowMeter: ACK Flag Count",
        calculation="Verbatim count of ACK flags observed in flow.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Total ACK packets in the flow.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="rst_count",
        data_type="float64",
        source_fields="CICFlowMeter: RST Flag Count",
        calculation="Verbatim count of RST flags observed in flow.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Total RST packets (abrupt teardown or port-closed indicator).",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="fin_count",
        data_type="float64",
        source_fields="CICFlowMeter: FIN Flag Count",
        calculation="Verbatim count of FIN flags observed in flow.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Total graceful connection termination packets.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="psh_count",
        data_type="float64",
        source_fields="CICFlowMeter: PSH Flag Count",
        calculation="Verbatim count of PSH flags observed in flow.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Total packets requesting immediate application buffer push.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="urg_count",
        data_type="float64",
        source_fields="CICFlowMeter: URG Flag Count",
        calculation="Verbatim count of URG flags observed in flow.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Total packets with urgent pointer set.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="tcp_window_size",
        data_type="float64",
        source_fields="CICFlowMeter: Init_Win_bytes_forward",
        calculation="Initial forward window bytes recorded by sensor.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Coarse TCP receive window size indicator.",
        available_in=("cic_ids2017",),
    ),

    # =========================================================================
    # FLOW LEVEL - UNAVAILABLE IN CIC-IDS2017 (EXPLICITLY MARKED)
    # =========================================================================
    FeatureSpec(
        feature_name="ttl_mean",
        data_type="float64",
        source_fields="not available",
        calculation="Left NA. CICFlowMeter CSV output does not capture IP TTL headers.",
        required=False,
        available=False,
        aggregation_level="flow",
        description="Mean IP Time-To-Live. Unavailable in CIC-IDS2017 flows; never fabricated.",
        available_in=(),
    ),
    FeatureSpec(
        feature_name="ttl_std",
        data_type="float64",
        source_fields="not available",
        calculation="Left NA. CICFlowMeter CSV output does not capture IP TTL headers.",
        required=False,
        available=False,
        aggregation_level="flow",
        description="Standard deviation of IP TTL. Unavailable in CIC-IDS2017 flows; never fabricated.",
        available_in=(),
    ),
    FeatureSpec(
        feature_name="retransmission_count",
        data_type="float64",
        source_fields="not available",
        calculation="Left NA. CICFlowMeter does not compute TCP retransmission metrics.",
        required=False,
        available=False,
        aggregation_level="flow",
        description="Number of retransmitted TCP segments. Unavailable in CIC-IDS2017; never fabricated.",
        available_in=(),
    ),

    # =========================================================================
    # FLOW LEVEL - DERIVED CANONICAL RATE & RATIO FEATURES
    # =========================================================================
    FeatureSpec(
        feature_name="packets_per_second",
        data_type="float64",
        source_fields="packet_count, flow_duration",
        calculation="packet_count / max(flow_duration, 1e-6).",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Rate of packet transmission in packets per second.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="bytes_per_second",
        data_type="float64",
        source_fields="byte_count, flow_duration",
        calculation="byte_count / max(flow_duration, 1e-6).",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Bandwidth utilization rate in bytes per second.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="bytes_per_packet",
        data_type="float64",
        source_fields="byte_count, packet_count",
        calculation="byte_count / max(packet_count, 1.0).",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Average payload size per packet.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="syn_ratio",
        data_type="float64",
        source_fields="syn_count, packet_count",
        calculation="syn_count / max(packet_count, 1.0).",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Proportion of packets carrying the SYN flag (SYN flood / scan indicator).",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="ack_ratio",
        data_type="float64",
        source_fields="ack_count, packet_count",
        calculation="ack_count / max(packet_count, 1.0).",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Proportion of packets carrying the ACK flag.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="rst_ratio",
        data_type="float64",
        source_fields="rst_count, packet_count",
        calculation="rst_count / max(packet_count, 1.0).",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Proportion of packets carrying the RST flag (failed connection / reset indicator).",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="fin_ratio",
        data_type="float64",
        source_fields="fin_count, packet_count",
        calculation="fin_count / max(packet_count, 1.0).",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Proportion of packets carrying the FIN flag.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="psh_ratio",
        data_type="float64",
        source_fields="psh_count, packet_count",
        calculation="psh_count / max(packet_count, 1.0).",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Proportion of packets carrying the PSH flag.",
        available_in=("cic_ids2017",),
    ),

    # =========================================================================
    # FLOW LEVEL - NORMALIZED LABELS & STAGES
    # =========================================================================
    FeatureSpec(
        feature_name="normalized_attack_class",
        data_type="string",
        source_fields="dataset_label, configs/stages.yaml",
        calculation="Mapped via stages.yaml into standard attack families.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Normalized attack classification (Benign, DoS, Brute Force, Web Attack, etc.).",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="attack_stage",
        data_type="string",
        source_fields="dataset_label, configs/stages.yaml",
        calculation="Inferred kill-chain stage proxy. Explicitly marked inferred for attack classes.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Cyber kill-chain stage proxy (Reconnaissance, Initial Access, Impact, etc.).",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="label_confidence",
        data_type="string",
        source_fields="dataset_label",
        calculation="'ground_truth' for Benign, 'inferred' for attack proxy mappings.",
        required=False,
        available=True,
        aggregation_level="flow",
        description="Epistemic confidence tier for attack stage labeling.",
        available_in=("cic_ids2017",),
    ),

    # =========================================================================
    # HOST-WINDOW LEVEL - AGGREGATED TEMPORAL BEHAVIOR
    # =========================================================================
    FeatureSpec(
        feature_name="host_id",
        data_type="string",
        source_fields="group_by column (default: source_ip)",
        calculation="Grouping identifier representing the monitored host.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Network host entity identifier evaluated in this temporal window.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="window_start",
        data_type="datetime64[ns]",
        source_fields="timestamp",
        calculation="Start timestamp of the sliding time window [window_start, window_end).",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Lower temporal bound of the aggregation interval.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="window_end",
        data_type="datetime64[ns]",
        source_fields="timestamp",
        calculation="window_start + window_size_seconds.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Upper temporal bound of the aggregation interval.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="flow_count",
        data_type="float64",
        source_fields="flow records",
        calculation="Total number of flows originating from host in the window interval.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Flows per interval; volume of connection sessions.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="packet_count_sum",
        data_type="float64",
        source_fields="packet_count",
        calculation="Sum of packet counts across all flows in window.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Packets per interval; cumulative packet throughput.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="packet_count_mean",
        data_type="float64",
        source_fields="packet_count",
        calculation="Mean packet count per flow in window.",
        required=False,
        available=True,
        aggregation_level="host-window",
        description="Average flow depth in packet count.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="byte_count_sum",
        data_type="float64",
        source_fields="byte_count",
        calculation="Sum of byte counts across all flows in window.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Bytes per interval; cumulative data transfer volume.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="byte_count_mean",
        data_type="float64",
        source_fields="byte_count",
        calculation="Mean byte count per flow in window.",
        required=False,
        available=True,
        aggregation_level="host-window",
        description="Average session volume in bytes.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="duration_mean",
        data_type="float64",
        source_fields="flow_duration",
        calculation="Mean duration of active flows in window.",
        required=False,
        available=True,
        aggregation_level="host-window",
        description="Average session persistence.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="packets_per_second_mean",
        data_type="float64",
        source_fields="packets_per_second",
        calculation="Mean packet rate across flows in window.",
        required=False,
        available=True,
        aggregation_level="host-window",
        description="Average packet transmission rate across active flows.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="bytes_per_second_mean",
        data_type="float64",
        source_fields="bytes_per_second",
        calculation="Mean byte rate across flows in window.",
        required=False,
        available=True,
        aggregation_level="host-window",
        description="Average bandwidth rate across active flows.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="bytes_per_packet_mean",
        data_type="float64",
        source_fields="bytes_per_packet",
        calculation="Mean bytes per packet across flows in window.",
        required=False,
        available=True,
        aggregation_level="host-window",
        description="Average payload size characteristic.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="unique_destination_ips",
        data_type="float64",
        source_fields="destination_ip",
        calculation="Count of distinct destination IP addresses contacted by host in window.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Host fan-out; horizontal scanning / botnet dissemination indicator.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="unique_destination_ports",
        data_type="float64",
        source_fields="destination_port",
        calculation="Count of distinct destination ports probed by host in window.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Vertical port scanning / service discovery indicator.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="unique_source_ports",
        data_type="float64",
        source_fields="source_port",
        calculation="Count of distinct source ports utilized by host in window.",
        required=False,
        available=True,
        aggregation_level="host-window",
        description="Source port diversity; ephemeral port recycling indicator.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="destination_port_entropy",
        data_type="float64",
        source_fields="destination_port",
        calculation="Shannon entropy H = -sum(p * log2(p)) over destination port distribution.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Dispersion of targeted ports; high entropy indicates sweeping reconnaissance.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="connection_burst_max",
        data_type="float64",
        source_fields="timestamp",
        calculation="Maximum number of flows initiated within any single 1-second sub-interval.",
        required=False,
        available=True,
        aggregation_level="host-window",
        description="Peak short-term burstiness of connection initiations.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="attack_flow_ratio",
        data_type="float64",
        source_fields="normalized_attack_class",
        calculation="Count of non-benign flows divided by total flow_count in window.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Proportion of malicious traffic emitted by host during window.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="benign_flow_ratio",
        data_type="float64",
        source_fields="normalized_attack_class",
        calculation="Count of benign flows divided by total flow_count in window.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Proportion of normal/legitimate traffic during window.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="attack_present",
        data_type="int64",
        source_fields="normalized_attack_class",
        calculation="1 if attack_flow_ratio > 0 else 0.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Binary indicator whether host engaged in any attack during window.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="dominant_class",
        data_type="string",
        source_fields="normalized_attack_class",
        calculation="Attack class with highest flow count (if attack present, among attacks; else 'Benign').",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Dominant attack category active in the window interval.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="dominant_stage",
        data_type="string",
        source_fields="attack_stage",
        calculation="Kill-chain stage with highest flow count (if attack present, among attacks; else 'Benign').",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Dominant attack stage proxy active in window.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="stage_distribution",
        data_type="string (json)",
        source_fields="attack_stage",
        calculation="JSON string serializing exact flow counts for each stage present.",
        required=True,
        available=True,
        aggregation_level="host-window",
        description="Multivariate distribution of attack stages occurring within window.",
        available_in=("cic_ids2017",),
    ),

    # =========================================================================
    # SEQUENCE LEVEL - TEMPORAL SEQUENCES & FUTURE TARGETS
    # =========================================================================
    FeatureSpec(
        feature_name="sequence_id",
        data_type="string",
        source_fields="host_id, window indices",
        calculation="Deterministic identifier {host_id}_seq_{idx}.",
        required=True,
        available=True,
        aggregation_level="sequence",
        description="Unique identifier for ordered window sequence.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="sequence_features",
        data_type="array[sequence_length, feature_count]",
        source_fields="host-window feature vectors",
        calculation="Ordered stack of host-window aggregate feature vectors [W_t ... W_{t+L-1}].",
        required=True,
        available=True,
        aggregation_level="sequence",
        description="Input temporal tensor for world model / neural forecasting.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="target_window_start",
        data_type="datetime64[ns]",
        source_fields="first eligible same-host window_start after final input window_end",
        calculation="For horizon H, select the Hth window with window_start strictly after the final input window_end.",
        required=True,
        available=True,
        aggregation_level="sequence",
        description="Start time of the future forecast target interval.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="target_attack_present",
        data_type="int64",
        source_fields="future window attack_present",
        calculation="Binary attack presence in the Hth eligible strictly future target window.",
        required=True,
        available=True,
        aggregation_level="sequence",
        description="Binary forecasting objective: will attack occur in future horizon?",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="target_dominant_class",
        data_type="string",
        source_fields="future window dominant_class",
        calculation="Dominant attack category in the Hth eligible strictly future target window.",
        required=True,
        available=True,
        aggregation_level="sequence",
        description="Multiclass forecasting objective: future attack category.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="target_dominant_stage",
        data_type="string",
        source_fields="future window dominant_stage",
        calculation="Dominant kill-chain stage in the Hth eligible strictly future target window.",
        required=True,
        available=True,
        aggregation_level="sequence",
        description="Stage forecasting objective: future attack progression stage.",
        available_in=("cic_ids2017",),
    ),
    FeatureSpec(
        feature_name="target_features",
        data_type="array[feature_count]",
        source_fields="future window features",
        calculation="Aggregate feature vector of the Hth eligible strictly future target window.",
        required=False,
        available=True,
        aggregation_level="sequence",
        description="Future network state vector for world model regression forecasting.",
        available_in=("cic_ids2017",),
    ),
]


def get_registry() -> list[FeatureSpec]:
    """Return the complete registry of all registered features."""
    return list(FEATURE_REGISTRY)


def get_features_by_level(level: str) -> list[FeatureSpec]:
    """Filter registry by aggregation_level ('flow', 'host-window', 'sequence')."""
    return [f for f in FEATURE_REGISTRY if f.aggregation_level == level]


def get_unavailable_features() -> list[FeatureSpec]:
    """Return features that are structurally not available in the current source datasets."""
    return [f for f in FEATURE_REGISTRY if not f.available]


def registry_as_markdown(level: str | None = None) -> str:
    """Format registry as GitHub markdown table."""
    entries = FEATURE_REGISTRY if level is None else get_features_by_level(level)
    lines = [
        "| Feature | Level | Source | Type | Required | Available | Calculation | Description |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in entries:
        avail_str = "yes" if f.available else "NO (NA)"
        req_str = "yes" if f.required else "no"
        lines.append(
            f"| `{f.feature_name}` | {f.aggregation_level} | {f.source_fields} | {f.data_type} "
            f"| {req_str} | {avail_str} | {f.calculation} | {f.description} |"
        )
    return "\n".join(lines)
