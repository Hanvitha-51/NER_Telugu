from Model.ner_model import NERModel
from Model.config import Config
from sklearn.metrics import precision_score, recall_score

def predict_test(model, filename_test):
    predicted_tags = []
    original_tags = []

    with open(filename_test, "r") as test_file:
        sentence = []
        orig_tags = []
        for line in test_file:
            line = line.strip().split(' ')
            if len(line) == 1:
                pred_tags = model.predict(sentence) 
                predicted_tags.extend(pred_tags)
                original_tags.extend(orig_tags)
                sentence = []
                orig_tags = []
            else:
                sentence.append(line[0])   
                orig_tags.append(line[-1])

    return predicted_tags, original_tags

def main():
    config = Config()

    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    predicted_tags, original_tags = predict_test(model, config.filename_test)

    # Calculate precision and recall
    precision = precision_score(original_tags, predicted_tags, average='weighted')
    recall = recall_score(original_tags, predicted_tags, average='weighted')

    print("Precision:", precision)
    print("Recall:", recall)

if __name__ == "__main__":
    main()

