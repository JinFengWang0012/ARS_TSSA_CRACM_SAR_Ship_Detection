import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge comparison table txt files without duplicate reruns.')
    parser.add_argument(
        'inputs',
        nargs='+',
        help='input comparison txt files, later files override earlier rows')
    parser.add_argument(
        '--output',
        required=True,
        help='merged output txt path')
    return parser.parse_args()


def parse_table_file(path):
    sections = OrderedDict()
    current_table = None
    current_dataset = None

    for raw_line in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith('TABLE]'):
            current_table = line
            sections.setdefault(current_table, OrderedDict())
            current_dataset = None
            continue
        if line.startswith('Dataset:'):
            current_dataset = line.split(':', 1)[1].strip()
            sections[current_table].setdefault(current_dataset, OrderedDict())
            continue
        if line.startswith('|---'):
            continue
        if not line.startswith('|'):
            continue
        parts = [part.strip() for part in line.split('|')[1:-1]]
        if not parts or parts[0] == 'Method':
            continue
        method = parts[0]
        sections[current_table][current_dataset][method] = line
    return sections


def merge_sections(input_paths):
    merged = OrderedDict()
    for input_path in input_paths:
        parsed = parse_table_file(input_path)
        for table_name, dataset_map in parsed.items():
            merged.setdefault(table_name, OrderedDict())
            for dataset_name, method_map in dataset_map.items():
                merged[table_name].setdefault(dataset_name, OrderedDict())
                for method_name, row in method_map.items():
                    merged[table_name][dataset_name][method_name] = row
    return merged


def format_output(merged, inputs):
    lines = []
    lines.append('Merged Comparison Experiment Summary')
    lines.append(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('Sources:')
    for input_path in inputs:
        lines.append(f'- {Path(input_path).resolve()}')
    lines.append('')

    for table_name, dataset_map in merged.items():
        lines.append(table_name)
        for dataset_name, method_map in dataset_map.items():
            lines.append(f'Dataset: {dataset_name}')
            lines.append('| Method | AP | AP50 | AP75 | Params (M) | FPS | Best Checkpoint |')
            lines.append('|---|---:|---:|---:|---:|---:|---|')
            for _, row in method_map.items():
                lines.append(row)
            lines.append('')
        lines.append('')
    return '\n'.join(lines)


def main():
    args = parse_args()
    merged = merge_sections(args.inputs)
    output_text = format_output(merged, args.inputs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding='utf-8')
    print(f'Saved merged table to: {output_path}')


if __name__ == '__main__':
    main()
