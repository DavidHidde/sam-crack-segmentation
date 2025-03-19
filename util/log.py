import os
from dataclasses import dataclass
from typing import TextIO, Any


@dataclass
class MetricItem:
    """Value which represents the current value of a metric. Simple name-value wrapper."""
    name: str
    value: Any


class CSVLogger:
    """Logger which logs values to a single CSV file."""

    output_file: TextIO
    items: list[MetricItem]
    delimiter: str

    def __init__(self, output_file_path: str, items: list[MetricItem], delimiter: str = ';', write_header: bool = True):
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)

        self.output_file = open(output_file_path, 'a')
        self.items = items
        self.delimiter = delimiter

        if write_header:
            self.output_file.write(self.delimiter.join([item.name for item in self.items]) + '\n')

    def write_line(self) -> None:
        """Write a line to the log file based on the current items."""
        self.output_file.write(self.delimiter.join([str(item.value) for item in self.items]) + '\n')

    def close(self) -> None:
        """Close the log file."""
        self.output_file.flush()
        self.output_file.close()
