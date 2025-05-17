from datasets import load_dataset

def load_wildseek(name: str, config: str, split: str, text_field: str):
    ds = load_dataset(name, config, split=split)
    # Return list of raw text examples
    print(ds.features)           # shows the Features object listing column names & types
    print(ds.column_names)
    return [example[text_field] for example in ds]