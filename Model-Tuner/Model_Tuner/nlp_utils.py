from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import numpy as np


def clean_up_text(
    txt=None,
    remove_stop_words=True,
    stem_words=False,
    special_chars=string.punctuation,
    split_char=" ",
    keep_tokens=[],
):
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
        print("WARNING no text was passed into clean_up_text")
        return

    # put all the text in lower case
    txt = txt.lower()

    # split the text on white space and remove special characters
    txt = txt.split(split_char)

    # creates a mapping table to map special characters / punctuation to ''
    table = str.maketrans("", "", special_chars)
    txt = [w.translate(table) for w in txt]

    if remove_stop_words:
        txt = [
            x
            for x in txt
            if (x not in stopwords.words("english")) or (x in keep_tokens)
        ]

    if stem_words:
        porter = PorterStemmer()
        txt = [porter.stem(word) for word in txt]

    return txt


def avg_word_vec(
    token_set=None, embed_size=None, vec_model=None, mn_or_mx="mean", wts=None
):
    """
    Function to calculate the compressed vector representation of token from gensim model
    :param token_set: list of tokens to get representations
    :param embed_size: int size of the embedding
    :param vec_model: gensim keyed vector model
    :param wts: optional vector of wts to multiply the vectors by
    :return: np array vector representation
    """
    # if the user has at least 1 token then run the wt vec else return np.nan
    if len(token_set) > 0:
        # initialize the matrix for the word vectors
        mat = np.zeros((len(token_set), embed_size))

        # for each word in the token_set try to get the word vector and store it in the matrix
        # if the word wasn't in the vocabulary remove the row from the matrix so that don't end up
        # with a row of all zeros and move on to the next word in the set
        for i in range(len(token_set)):
            try:
                mat[i, :] = vec_model[token_set[i]]
            except Exception as e:
                print("NOT IN MODEL: ", token_set[i])
                continue

        # remove rows with all zeros as these are from rows with no ft info
        # do not equal as opposed to greater then zero cause could sum to neg val
        mat = mat[mat.sum(axis=1) != 0]

        # multiply the matrix with the wts before returning the average vector
        if wts:
            mat = mat * np.array(wts)[:, np.newaxis]

        # return the average of the word vectors
        if mn_or_mx == "mean":
            return mat.mean(axis=0)
        elif mn_or_mx == "max":
            return mat.max(axis=0)
    else:
        return [np.nan]
