from transformers import BertTokenizer

def get_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)

def tokens_merger(token_strings, attention_logits, method="mean"):
    """this merges the attention logits of the tokens into a single value for particular tokens
    for example, if the token is '##ing', it will merge the attention logits of 'run' and '##ing' into a single value
    using operations like sum, average, max etc."""

    pass