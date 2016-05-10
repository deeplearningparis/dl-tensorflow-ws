from dataset import Dataset

if __name__ == '__main__':
    train_set = Dataset('train')
    test_set = Dataset('test')

    n_epochs = 50
    for epoch in xrange(n_epochs):
        for word_source, word_target in train_set:
            # train model
            print word_source, word_target

        for word_source, word_target in test_set:
            # validate model
            print word_source, word_target
