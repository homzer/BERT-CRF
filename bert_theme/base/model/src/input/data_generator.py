class DataGenerator:
    def __init__(self, batch_size, epochs, features, labels):
        self.features = list(features)
        self.labels = list(labels)
        self.batch_size = batch_size
        self.epochs = epochs
        self.total_size = len(self.features)
        self.data_index = 0
        self.epoch_count = 0
        self.batch_count = 0
        self.batches_per_epoch = int(self.total_size / self.batch_size)
        self.total_steps = self.batches_per_epoch * self.epochs

    def next_batch(self):
        """ Recurrently generate features and labels with the size of batch. """
        batch_features = self.features[self.data_index: self.data_index + self.batch_size]
        batch_labels = self.labels[self.data_index: self.data_index + self.batch_size]
        # Exceed
        if self.batch_size + self.data_index > self.total_size:
            batch_features.extend(self.features[0: self.batch_size + self.data_index - self.total_size])
            batch_labels.extend(self.labels[0: self.batch_size + self.data_index - self.total_size])
        self.data_index = (self.data_index + self.batch_size) % self.total_size
        self.__update()
        assert len(batch_features) == self.batch_size
        assert len(batch_labels) == self.batch_size
        return batch_features, batch_labels

    def has_next(self):
        """ If there is the end of dataset. """
        return self.epoch_count < self.epochs

    def __update(self):
        """ update state. """
        self.batch_count += 1
        if self.batch_count >= self.batches_per_epoch:
            self.epoch_count += 1
            self.batch_count = 0
