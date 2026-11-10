import pytest

from features.data import DataError, clean_criteria, load_master, parse_icd, parse_smiles, table_hash
from features.tests.conftest import row, write_raw


def test_parse_icd_nested_format():
    s = str(["['D02.20', 'D02.21']", "['c61', 'D02.20']"])
    codes, n_bad, ok = parse_icd(s)
    assert ok and n_bad == 0
    assert codes == ["D02.20", "D02.21", "C61"]       # upper-cased, de-duplicated, order kept


def test_parse_icd_drops_malformed_and_accepts_letter_categories():
    codes, n_bad, ok = parse_icd(str(["['C7A.010', 'banana', '12.3', 'O9A.52']"]))
    assert codes == ["C7A.010", "O9A.52"] and n_bad == 2 and ok


def test_parse_icd_unparseable():
    assert parse_icd("[not python")[2] is False


def test_parse_smiles():
    assert parse_smiles("['CCO', ' CCO ', 'C']") == (["CCO", "C"], True)
    assert parse_smiles("[]") == ([], True)
    assert parse_smiles("['CCO'")[1] is False


def test_clean_criteria_keeps_lines():
    assert clean_criteria("a\r\n  -  b\t\tc  \r\n") == "a\n - b c"


def test_master_order_and_columns(small_master):
    m, rep = small_master
    assert "status" not in m.columns and "why_stop" not in m.columns      # leakage columns never read
    order = list(zip(m.phase, m.split, m.nctid))
    key = {"train": 0, "valid": 1, "test": 2}
    assert order == sorted(order, key=lambda t: (t[0], key[t[1]], t[2]))
    assert m.index.tolist() == list(range(len(m)))
    # a phase 1/2 trial is one row in phase 1 and one in phase 2
    assert sorted(m.loc[m.nctid == "NCT00000010", "phase"]) == [1, 2]
    assert rep["counts"]["1/train"] == 3


def test_phase_comes_from_file_not_raw_string(small_master):
    m, _ = small_master
    r = m[(m.nctid == "NCT00000010")]
    assert set(r.phase_raw) == {"phase 1/phase 2"} and set(r.phase) == {1, 2}


def test_bad_ids_and_labels_dropped(tmp_path):
    raw = write_raw(tmp_path, {("I", "train"): [row("NCT00000001", 1), row("XYZ", 1), row("NCT00000002", 2)]})
    m, rep = load_master(raw, strict_version=False, check_counts=False)
    assert m.nctid.tolist() == ["NCT00000001"]
    assert rep["dropped"] == {"malformed nctid": 1, "label not 0/1": 1}


def test_exact_duplicate_dropped_conflicting_duplicate_fatal(tmp_path):
    raw = write_raw(tmp_path / "a", {("I", "train"): [row("NCT00000001", 1), row("NCT00000001", 1)]})
    m, rep = load_master(raw, strict_version=False, check_counts=False)
    assert len(m) == 1 and rep["dropped"]["exact duplicate (nctid, phase)"] == 1
    raw = write_raw(tmp_path / "b", {("I", "train"): [row("NCT00000001", 1), row("NCT00000001", 0)]})
    with pytest.raises(DataError):
        load_master(raw, strict_version=False, check_counts=False)


def test_test_train_leak_across_phases_is_fatal(tmp_path):
    raw = write_raw(tmp_path, {("I", "train"): [row("NCT00000001", 1)], ("II", "test"): [row("NCT00000001", 1)]})
    with pytest.raises(DataError, match="test for one phase"):
        load_master(raw, strict_version=False, check_counts=False)


def test_cross_phase_label_conflict_is_fatal(tmp_path):
    raw = write_raw(tmp_path, {("I", "train"): [row("NCT00000001", 1)], ("II", "train"): [row("NCT00000001", 0)]})
    with pytest.raises(DataError, match="label"):
        load_master(raw, strict_version=False, check_counts=False)


def test_version_pin_is_enforced(small_raw):
    raw, _ = small_raw
    with pytest.raises(DataError, match="does not match pinned"):
        load_master(raw, strict_version=True)


def test_count_check(small_raw):
    raw, _ = small_raw
    with pytest.raises(DataError, match="row counts"):
        load_master(raw, strict_version=False, check_counts=True)


def test_table_hash_sensitive_to_label_and_order(small_master):
    m, _ = small_master
    h = table_hash(m)
    m2 = m.copy()
    m2.loc[0, "label"] = 1 - m2.loc[0, "label"]
    assert table_hash(m2) != h
    assert table_hash(m.iloc[::-1]) != h
