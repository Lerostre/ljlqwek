from collections import Counter, defaultdict
import stanza
stanza.download('ru')
nlp = stanza.Pipeline('ru', processors='tokenize, lemma')


def normalize(text):
    doc = nlp(text)
    words = [word.lemma for sent in doc.sentences for word in sent.words]
    return words


def get_mention_category(data, cat_type):
    mention_categories = data.value_counts(subset=['norm_mention', cat_type])
    mention_categories_dict = defaultdict(dict)
    for key, value in mention_categories.items():
        mention_categories_dict[key[0]][key[1]] = value
    return {k: Counter(v).most_common(1)[0][0] for k, v in mention_categories_dict.items()}


def label_texts(text, mentions, sentiments, max_len=5):
    tokenized = [word for sent in nlp(text).sentences for word in sent.words]
    text_end = len(tokenized)
    for i, token in enumerate(tokenized):
        for l in reversed(range(max_len)):
            if i + l > text_end:
                continue
            span = tokenized[i:i + l]
            key = tuple([t.lemma for t in span])
            if key in mentions:
                start, end = span[0].start_char, span[-1].end_char
                yield mentions[key], text[start:end], start, end, sentiments[key]
                break