from Model.data_utils import Dataset
from Model.NERModel import NERModel
from model.config import Config

def main():
    config = Config()

    model = NERModel(config)
    model.build()

    dev = Dataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = Dataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    model.train(train, dev)

if __name__ == "__main__":
    main()
