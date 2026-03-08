import argparse
import difflib
import json

# ANSI escape codes for colors
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def main():
    parser = argparse.ArgumentParser(description='Generate colored diffs for JSONL entries.')
    parser.add_argument('--input_file', help='Path to the JSONL input file')
    parser.add_argument('-n', '--number', type=int, required=True, help='Number of items to process')
    args = parser.parse_args()

    # Read the first N entries
    entries = []
    with open(args.input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.number:
                break
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                print(f"Warning: Line {i+1} is not valid JSON. Skipping.")
                continue

    # Process each entry
    for idx, entry in enumerate(entries):
        print(f"Entry {idx + 1}:")
        smelly_sample = entry.get("Smelly Sample", "")
        method_after = entry.get("Method after Refactoring", "")

        smelly_lines = smelly_sample.splitlines(keepends=False)
        method_lines = method_after.splitlines(keepends=False)

        # Generate unified diff
        diff = difflib.unified_diff(
            smelly_lines,
            method_lines,
            fromfile='Smelly Sample',
            tofile='Method after Refactoring',
            lineterm=''
        )

        # Print colored diff
        for line in diff:
            if line.startswith('---'):
                print(f"{RED}{line}{RESET}")
            elif line.startswith('+++'):
                print(f"{GREEN}{line}{RESET}")
            elif line.startswith('@'):
                print(f"{BLUE}{line}{RESET}")
            elif line.startswith('+'):
                print(f"{GREEN}{line}{RESET}")
            elif line.startswith('-'):
                print(f"{RED}{line}{RESET}")
            else:
                print(line)
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()