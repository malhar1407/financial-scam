"""Text preprocessing for scam message classification."""
import re
import pandas as pd
from datasets import load_dataset


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\b\d{10,}\b", "<PHONE>", text)
    text = re.sub(r"[^\w\s<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_sms_spam() -> pd.DataFrame:
    """Load UCI SMS Spam dataset via HuggingFace."""
    ds = load_dataset("ucirvine/sms_spam", split="train")
    df = ds.to_pandas().rename(columns={"sms": "text", "label": "label"})
    df["text"] = df["text"].apply(clean_text)
    return df[["text", "label"]]  # label: 0=ham, 1=spam


def load_phishing_emails() -> pd.DataFrame:
    """Load phishing email dataset."""
    ds = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    df = ds.to_pandas().rename(columns={"Email Text": "text", "Email Type": "label"})
    df["label"] = df["label"].map({"Phishing Email": 1, "Safe Email": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)
    return df[["text", "label"]]


def build_text_dataset(save_path: str = "scam_detection/data/processed/text_dataset.csv"):
    sms = load_sms_spam()
    phishing = load_phishing_emails()
    combined = pd.concat([sms, phishing], ignore_index=True).sample(frac=1, random_state=42)
    combined.to_csv(save_path, index=False)
    print(f"Saved {len(combined)} samples → {save_path}")
    return combined


if __name__ == "__main__":
    build_text_dataset()
