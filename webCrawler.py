from lxml import etree
import fasttext
import requests


# Web crawler
class LanguageIdentifier(object):
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def predict(self, line):
        line = line.strip()
        labels, probs = self.model.predict(line)
        label = labels[0].split("__")[-1]

        return label

if __name__ == "__main__":
    url = "https://en.wikisource.org/wiki/Grammar_of_the_Burmese_Language"
    # Make a request:
    res = requests.get(url)
    # Read content
    parser = etree.HTMLParser()
    html = etree.HTML(res.content)

    # Obtain text based on this re
    text_nodes = html.xpath(".//p/text() | //span/text()")
    print(text_nodes[1:20]) # test

    # Preprocessing
    for text in text_nodes[:50]:
        if text.strip() != "":
            print(text.strip())

    text_nodes = [text_node.strip() for text_node in text_nodes if text_node.strip() != ""]

    # Web crawler
    lid = LanguageIdentifier(model_path="lid.176.bin") 
    print(lid.predict("Hallo zusammen")) #Example

    english_text = []
    burmese_text = []
    other = []

    for text in text_nodes:
        lang_id_result = lid.predict(text)
    
        if lang_id_result == "en":
            english_text.append(text)
        elif lang_id_result == "my":
            burmese_text.append(text)
        else:
            other.append(text)

    print("English:")
    print(english_text[:10])
    print()
    print("Burmese:")
    print(burmese_text[:10])
    print()
    print("Other:")
    print(other[:10])