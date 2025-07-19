from transformers import BertTokenizer

def get_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)
