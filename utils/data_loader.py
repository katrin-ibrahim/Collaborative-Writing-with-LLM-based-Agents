from datasets import load_dataset

def load_wildseek(path: str = "YuchengJiang/WildSeek"):
    """
    Load the dataset from the specified path.
    Args:
        path (str): Path to the dataset directory.
    Returns:
        List[str]: A list of text entries from the dataset.
    """
    ds = load_dataset(path, split="train")
    texts = [ex['topic'] for ex in ds]
    return texts