def clean(data):
    """
    Cleans data
    """

    # Import statements
    import re
    import string

    # Remove whitespace
    data['text'] = data['text'].str.replace(r'\s+', ' ')

    # Remove chars \n and \t
    data['text'] = data['text'].str.replace(r'\\n', ' ')
    data['text'] = data['text'].str.replace(r'\\t', ' ')

    # Remove rows with Empty Descriptions
    data = data[data['text'] != '']

    # Remove non-English characters from description
    data['text'] = data['text'].str.replace(r'[^\x00-\x7F]+', '')

    # Remove comment code
    data['text'] = data['text'].str.replace("<!--.*-->", "")

    # Return only the rows which do not have NaN in ['text'] column:
    data = data[data['text'].notna()]

    return data
