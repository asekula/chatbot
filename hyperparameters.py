
import pickle

# Hyperparameters
batch_size = 20
learning_rate = 0.001
window_size = 40
embedding_size = 50
hidden_size = 64
n_layers = 2
gru_dropout_p = 0.5
epochs = 10


def get_hyperparameter_dict():
    return {"batch_size": batch_size, "learning_rate": learning_rate, "window_size": window_size, "embedding_size":
            embedding_size, "hidden_size": hidden_size, "n_layers": n_layers, "gru_dropout_p": gru_dropout_p, "epochs":
            epochs}


def get_saved_hyperparameters(filename):
    return pickle.load(open('saved_model/' + filename + '_hyperparameters.p', 'rb'))


if __name__ == '__main__':
    print get_saved_hyperparameters('all_response_chat_data')
