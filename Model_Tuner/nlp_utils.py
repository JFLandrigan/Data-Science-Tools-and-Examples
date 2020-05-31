from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer


def clean_up_text(txt=None, remove_stop_words=True, stem_words=False, special_chars=string.punctuation, split_char=' ',
                  keep_tokens=[]):
    """
    Function takes string and returns list of cleaned tokens
    :param txt: expects string to clean
    :param remove_stop_words: Boolean indicator to removr stopwords. Based on nltk.corpus english stopwords
    :param stem_words: Boolean indicator to stem words. If true uses NLTK PorterStemmer
    :param special_chars: special characters to remove from text. Defaults to string.punctuation however can pass in string on characters to remove
    :param split_char: String pattern to split text on. Defaults to ' '
    :param keep_tokens: List of special case tokens to force cleaner to keep
    :return: list of cleaned tokens
    """
    if txt is None:
        print('WARNING no text was passed into clean_up_text')
        return

    # put all the text in lower case
    txt = txt.lower()

    # split the text on white space and remove special characters
    txt = txt.split(split_char)

    # creates a mapping table to map special characters / punctuation to ''
    table = str.maketrans('', '', special_chars)
    txt = [w.translate(table) for w in txt]

    if remove_stop_words:
        txt = [x for x in txt if (x not in stopwords.words('english')) or (x in keep_tokens)]

    if stem_words:
        porter = PorterStemmer()
        txt = [porter.stem(word) for word in txt]

    return (txt)

