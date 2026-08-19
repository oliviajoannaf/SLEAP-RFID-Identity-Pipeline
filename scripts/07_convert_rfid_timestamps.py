#!/usr/bin/env python3
"""
Stage 07: Clean an RFID export and convert its timestamps.

The raw LabNetNext RFID CSV contains both:
1. metadata/system rows; and
2. genuine RFID detection events.

This script separates those record types, converts Excel/OLE Automation
timestamps into standard datetimes, and produces clean files for downstream
RFID–SLEAP alignment.

Example
-------
python scripts/07_convert_rfid_timestamps.py \
    incoming/26.07.15spareRFID.csv \
    --output-dir outputs/development_v1/rfid
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("rfid_timestamp_conversion")

OLE_ORIGIN = datetime(1899, 12, 30)
MICROSECONDS_PER_DAY = Decimal("86400000000")
READER_PATTERN = re.compile(r"^R\d+$", flags=re.IGNORECASE)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Separate RFID detections from metadata and convert "
            "Excel/OLE timestamps."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the raw UTF-16 RFID CSV export.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/development_v1/rfid"),
        help="Directory for cleaned outputs.",
    )

    return parser.parse_args()


def read_rfid_csv(path: Path) -> pd.DataFrame:
    """Read the raw UTF-16 RFID export."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    LOGGER.info("Reading raw RFID file: %s", path)

    dataframe = pd.read_csv(
        path,
        encoding="utf-16",
        sep=None,
        engine="python",
        dtype={
            "DateTime": "string",
            "IdRFID": "string",
            "IdLabel": "string",
            "unitLabel": "string",
            "SystemMsg": "string",
            "MsgValue1": "string",
            "MsgValue2": "string",
        },
    )

    dataframe.insert(
        0,
        "raw_row_number",
        range(2, len(dataframe) + 2),
    )

    LOGGER.info(
        "Loaded %d rows and %d original columns.",
        len(dataframe),
        len(dataframe.columns) - 1,
    )

    return dataframe


def normalise_text(series: pd.Series) -> pd.Series:
    """Strip whitespace while retaining pandas string missing values."""
    return series.astype("string").str.strip()


def parse_ole_datetime(value: object) -> pd.Timestamp | pd.NaT:
    """
    Convert one Excel/OLE Automation timestamp to a pandas Timestamp.

    Decimal arithmetic is used instead of binary floating-point arithmetic
    to minimise avoidable timestamp conversion error.
    """
    if pd.isna(value):
        return pd.NaT

    cleaned = str(value).strip().replace(",", ".")

    if not cleaned or cleaned.lower() in {"nan", "none", "<na>"}:
        return pd.NaT

    try:
        ole_days = Decimal(cleaned)
    except InvalidOperation:
        return pd.NaT

    total_microseconds = int(
        (ole_days * MICROSECONDS_PER_DAY).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )

    return pd.Timestamp(
        OLE_ORIGIN + timedelta(microseconds=total_microseconds)
    )


def identify_event_rows(dataframe: pd.DataFrame) -> pd.Series:
    """
    Identify genuine RFID detection events.

    A valid event must have:
    - a convertible numerical DateTime value;
    - a non-empty RFID tag identifier; and
    - a reader label such as R1, R2, ..., R5.
    """
    parsed_timestamp = dataframe["DateTime"].map(parse_ole_datetime)

    tag_id = normalise_text(dataframe["IdRFID"])
    reader = normalise_text(dataframe["unitLabel"])

    valid_tag = tag_id.notna() & tag_id.ne("")
    valid_reader = reader.str.match(READER_PATTERN, na=False)
    valid_timestamp = parsed_timestamp.notna()

    return valid_timestamp & valid_tag & valid_reader


def classify_metadata_row(row: pd.Series) -> str:
    """Assign a descriptive category to a non-event row."""
    datetime_value = str(row.get("DateTime", "")).strip()
    system_message = str(row.get("SystemMsg", "")).strip().lower()
    unit_label = str(row.get("unitLabel", "")).strip().lower()

    if datetime_value == "#ID-Device":
        return "device_definition"

    if system_message == "version":
        return "software_version"

    if system_message == "start" or unit_label == "control":
        return "system_control"

    return "other_metadata"


def clean_events(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the standardised downstream RFID-event table."""
    events = dataframe.copy()

    events["timestamp"] = events["DateTime"].map(parse_ole_datetime)
    events["tag_id"] = normalise_text(events["IdRFID"]).str.upper()
    events["reader"] = normalise_text(events["unitLabel"]).str.upper()

    events["duration_ms"] = pd.to_numeric(
        events["eventDuration"],
        errors="coerce",
    )

    events["rfid_records"] = pd.to_numeric(
        events["senseRFIDrecords"],
        errors="coerce",
    ).astype("Int64")

    events = events.sort_values(
        ["timestamp", "raw_row_number"],
        kind="stable",
    ).reset_index(drop=True)

    recording_start = events["timestamp"].min()

    events["seconds_from_rfid_start"] = (
        events["timestamp"] - recording_start
    ).dt.total_seconds()

    output_columns = [
        "raw_row_number",
        "timestamp",
        "seconds_from_rfid_start",
        "tag_id",
        "reader",
        "duration_ms",
        "rfid_records",
    ]

    return events[output_columns]


def clean_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Preserve non-event rows in a separate metadata table."""
    metadata = dataframe.copy()
    metadata.insert(
        1,
        "metadata_type",
        metadata.apply(classify_metadata_row, axis=1),
    )
    return metadata


def format_duration(seconds: float) -> str:
    """Format seconds as hours, minutes and seconds."""
    rounded_seconds = int(round(seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours} h {minutes} min {secs} s"

    return f"{minutes} min {secs} s"


def write_summary(
    path: Path,
    input_path: Path,
    raw_dataframe: pd.DataFrame,
    events: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    """Write a human-readable quality-control summary."""
    readers = sorted(events["reader"].dropna().unique().tolist())
    tags = sorted(events["tag_id"].dropna().unique().tolist())

    start = events["timestamp"].min()
    end = events["timestamp"].max()
    duration_seconds = (end - start).total_seconds()

    missing_duration = int(events["duration_ms"].isna().sum())
    missing_records = int(events["rfid_records"].isna().sum())

    counts_by_reader = (
        events["reader"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    counts_by_tag = (
        events["tag_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    lines = [
        "RFID PREPROCESSING SUMMARY",
        "=" * 70,
        "",
        f"Input file: {input_path}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "ROW CLASSIFICATION",
        f"Raw rows: {len(raw_dataframe):,}",
        f"RFID event rows: {len(events):,}",
        f"Metadata/system rows: {len(metadata):,}",
        "",
        "RECORDING TIME",
        f"First RFID event: {start.isoformat(sep=' ')}",
        f"Last RFID event:  {end.isoformat(sep=' ')}",
        f"Event time span: {format_duration(duration_seconds)}",
        f"Event time span in seconds: {duration_seconds:.6f}",
        "",
        "READERS",
        f"Number of readers represented: {len(readers)}",
    ]

    for reader in readers:
        lines.append(
            f"{reader}: {counts_by_reader.get(reader, 0):,} events"
        )

    lines.extend(
        [
            "",
            "RFID TAGS",
            f"Number of unique tags: {len(tags)}",
        ]
    )

    for tag in tags:
        lines.append(f"{tag}: {counts_by_tag.get(tag, 0):,} events")

    lines.extend(
        [
            "",
            "MISSING VALUES IN CLEAN EVENT TABLE",
            f"Missing duration_ms: {missing_duration:,}",
            f"Missing rfid_records: {missing_records:,}",
            "",
            "VALIDATION",
            (
                "Row accounting passed: "
                f"{len(events)} events + {len(metadata)} metadata "
                f"= {len(raw_dataframe)} raw rows"
            ),
            "Timestamp conversion passed: all saved events have timestamps",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(
    raw_dataframe: pd.DataFrame,
    events: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    """Stop execution if essential quality checks fail."""
    if len(events) + len(metadata) != len(raw_dataframe):
        raise ValueError(
            "Row-accounting failure: events plus metadata does not equal "
            "the number of raw rows."
        )

    if events.empty:
        raise ValueError("No RFID event rows were identified.")

    if events["timestamp"].isna().any():
        raise ValueError("At least one saved event has no valid timestamp.")

    if events["tag_id"].isna().any():
        raise ValueError("At least one saved event has no RFID tag ID.")

    if events["reader"].isna().any():
        raise ValueError("At least one saved event has no reader label.")

    if not events["timestamp"].is_monotonic_increasing:
        raise ValueError("Clean RFID events are not ordered by timestamp.")


def main() -> None:
    """Run Stage 07."""
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raw_dataframe = read_rfid_csv(args.input_csv)

    event_mask = identify_event_rows(raw_dataframe)

    events = clean_events(raw_dataframe.loc[event_mask].copy())
    metadata = clean_metadata(raw_dataframe.loc[~event_mask].copy())

    validate_outputs(raw_dataframe, events, metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = args.input_csv.stem

    events_path = args.output_dir / f"{input_stem}_rfid_events.csv"
    metadata_path = args.output_dir / f"{input_stem}_rfid_metadata.csv"
    summary_path = args.output_dir / f"{input_stem}_rfid_summary.txt"

    events.to_csv(
        events_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S.%f",
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    write_summary(
        summary_path,
        args.input_csv,
        raw_dataframe,
        events,
        metadata,
    )

    LOGGER.info("RFID event rows: %d", len(events))
    LOGGER.info("Metadata/system rows: %d", len(metadata))
    LOGGER.info("Saved events: %s", events_path)
    LOGGER.info("Saved metadata: %s", metadata_path)
    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Stage 07 completed successfully.")


if __name__ == "__main__":
    main()
