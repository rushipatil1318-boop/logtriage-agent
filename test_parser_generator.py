from logtriage.generator import generate_dataset
from logtriage.parser import mask, parse_log


def test_mask_replaces_variable_fields_but_keeps_codes():
    assert mask("Downlink sync lost, RSRP=-83dBm SINR=-7.3dB") == "Downlink sync lost, RSRP=<NUM>dBm SINR=<NUM>dB"
    assert "#11" in mask("Attach reject received, EMM cause #11 (PLMN not allowed)")
    assert "T3410" in mask("T3410 expired, attempt 2/5 for attach request")
    assert mask("IP 10.1.2.3 up") == "IP <IP> up"


def test_same_event_different_values_gives_same_template():
    a = parse_log("2026-01-01 10:00:00.000 [ERROR] [PHY] Downlink sync lost, RSRP=-83dBm SINR=-7.3dB")[0]
    b = parse_log("2026-01-01 10:00:09.123 [ERROR] [PHY] Downlink sync lost, RSRP=-101dBm SINR=1.2dB")[0]
    assert a.tid == b.tid


def test_malformed_lines_keep_line_numbers_aligned():
    text = "2026-01-01 10:00:00.000 [INFO] [NAS] ok\n\x00garbage line\n2026-01-01 10:00:01.000 [ERROR] [NAS] boom\n"
    ev = parse_log(text)
    assert [e.line_no for e in ev] == [1, 2, 3]
    assert ev[1].level == "RAW"


def test_generator_is_deterministic():
    assert generate_dataset(5, 42)[3]["text"] == generate_dataset(5, 42)[3]["text"]


def test_ground_truth_is_consistent(dataset):
    for r in dataset:
        lines = r["text"].splitlines()
        if r["label"] == "PASS":
            assert r["root_line"] is None and r["known_defect"] is None
        else:
            assert "[ERROR]" in lines[r["root_line"] - 1] or "[FATAL]" in lines[r["root_line"] - 1]
    labels = {r["label"] for r in dataset}
    assert "PASS" in labels and len(labels) >= 6
