import os

batch_size = 8
epochs = 10
train_dir = './data/topic_train.txt'
eval_dir = './data/topic_dev.txt'
test_dir = './data/topic_test.txt'
output_dir = 'result'
config_dir = 'config'
vocab_file = os.path.join(config_dir, 'vocab.txt')
vocab_size = 21128
num_encoders = 3
seq_length = 128
hidden_size = 768
num_attention_heads = 12
dropout_prob = 0.1
intermediate_size = 3072
activation = "gelu"
