from datasets import load_dataset

eng_dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "eng")

def get_as_csv(split: str, path: str):
    if split not in ['train', 'test', 'dev']:
        raise ValueError("split must be one of: train, test, dev")
    return eng_dataset[split].to_csv(path)

def get_as_dict(split: str):
    if split not in ['train', 'test', 'dev']:
        raise ValueError("split must be one of: train, test, dev")
    return eng_dataset[split].to_dict()

def get_as_pandas(split: str):
    if split not in ['train', 'test', 'dev']:
        raise ValueError("split must be one of: train, test, dev")
    df = eng_dataset[split].to_pandas()
    df.fillna(0, inplace=True)
    return df

def get_as_dataset(split: str):
    if split not in ['train', 'test', 'dev']:
        raise ValueError("split must be one of: train, test, dev")
    return eng_dataset[split]

