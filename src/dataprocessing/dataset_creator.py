import os
import json
import argparse
from sklearn.model_selection import train_test_split

def read_jsonl_files(input_dir):
    data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        smelly = entry.get("Smelly Sample", "")
                        refactored = entry.get("Method after Refactoring", "")
                        
                        if smelly and refactored:
                            data.append({"text": smelly, "label": 1})
                            data.append({"text": refactored, "label": 0})
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON in {filename}, line: {line}")
                        continue
    return data

def split_data(data, train_ratio, val_ratio, test_ratio, random_seed):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"
    
    texts = [d['text'] for d in data]
    labels = [d['label'] for d in data]
    
    # Initial split into train and temporary (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, 
        train_size=train_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    # Split temporary into validation and test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_test_ratio,
        stratify=y_temp,
        random_state=random_seed
    )
    
    # Reconstruct dictionaries
    train_data = [{"text": text, "label": label} for text, label in zip(X_train, y_train)]
    val_data = [{"text": text, "label": label} for text, label in zip(X_val, y_val)]
    test_data = [{"text": text, "label": label} for text, label in zip(X_test, y_test)]
    
    return train_data, val_data, test_data

def write_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Create labeled dataset from JSONL files and split into train/val/test')
    parser.add_argument('--input_dir', help='Input directory containing JSONL files')
    parser.add_argument('--output_dir', help='Output directory for split datasets')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Verify ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0 (current sum: {total})")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read and process data
    print("Reading JSONL files...")
    data = read_jsonl_files(args.input_dir)
    print(f"Total samples collected: {len(data)}")
    
    # Split data
    print("Splitting dataset...")
    train_data, val_data, test_data = split_data(
        data, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio,
        args.random_seed
    )
    
    # Write splits to files
    write_jsonl(train_data, os.path.join(args.output_dir, 'train.jsonl'))
    write_jsonl(val_data, os.path.join(args.output_dir, 'validation.jsonl'))
    write_jsonl(test_data, os.path.join(args.output_dir, 'test.jsonl'))
    
    print(f"""
    Dataset split complete!
    Training samples: {len(train_data)}
    Validation samples: {len(val_data)}
    Test samples: {len(test_data)}
    Files saved to: {args.output_dir}
    """)

if __name__ == "__main__":
    main()