import spacy

class SpacyNLP:
    def __init__(self, spacy_model="en_core_sci_lg"):
        self.nlp = spacy.load(spacy_model)