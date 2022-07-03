from flair.models import TextClassifier
from flair.data import Sentence

# this can be refactored to just a script quite easily
# classifier = TextClassifier.load('en-sentiment')
# sentence = Sentence("sample text")
# classifier.predict(sentence)
# print(sentence.labels[0].value)
# print(sentence.labels[0].score)

def init_classifier():
    classifier = TextClassifier.load('en-sentiment')
    return classifier

def predict(text, classifier):
    sentence = Sentence(text)
    classifier.predict(sentence)
    return sentence.labels[0].value, sentence.labels[0].score
#
# if __name__ == "__main__":
#     classifier = init_classifier()
#     print(predict("I had kfc. was ok. lazed around", classifier))