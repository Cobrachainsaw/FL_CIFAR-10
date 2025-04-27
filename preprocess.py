import pickle
import numpy as np
import pandas as pd
import os

CIFAR10_FOLDER = "C:/Users/vinay/Downloads/cifar-10-python/cifar-10-batches-py"

OUTPUT_CLIENTS_DIR = "./clients_data"
OUTPUT_SERVER_PUBLIC_DIR = "./server_public_data"

os.makedirs(OUTPUT_CLIENTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_SERVER_PUBLIC_DIR, exist_ok=True)

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']          # Shape (10000, 3072)
        labels = batch[b'labels']       # List of labels
        return data, labels

for i in range(1, 6):
    batch_file = os.path.join(CIFAR10_FOLDER, f"data_batch_{i}")
    data, labels = load_batch(batch_file)

    data = data.astype(np.float32) / 255.0

    df = pd.DataFrame(data)
    df['label'] = labels

    client_csv_path = os.path.join(OUTPUT_CLIENTS_DIR, f"client_{i}.csv")
    df.to_csv(client_csv_path, index=False)
    print(f"Saved client_{i}.csv")

test_batch_file = os.path.join(CIFAR10_FOLDER, "test_batch")
test_data, test_labels = load_batch(test_batch_file)

test_data = test_data.astype(np.float32) / 255.0
df_test = pd.DataFrame(test_data)
df_test['label'] = test_labels

server_csv_path = os.path.join(OUTPUT_SERVER_PUBLIC_DIR, "server_public_test.csv")
df_test.to_csv(server_csv_path, index=False)
print(f"Saved server_public_test.csv")
