import spacy
from spacy import displacy

if __name__ == "__main__":
    text= "I bought Apples at Migros yesterday"

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    #displacy.serve(doc, style="ent")

    # Determine the context of a word
    doc = nlp("""We climbed the north face of Mount Everest.
You seem to face some real difficulties.
My face was glowing red from the heat.
Let’s face it, How I Met your Mother is dumb.""")
    
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    # or for a more descriptive output
    for token in doc:
        print(token.text, token.pos_)
    #displacy.render(doc, style="dep", page=True)