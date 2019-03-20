
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout

class NeuralNetworkConfig:
    
    def __init__(self, input_indices):
    
        self.recursive_depth = 3
        self.epochs = 50
        self.batch_size = 100
        
        self.optimizer = "adam"
        self.loss = "mse"
        self.metrics = ["accuracy"]
        self.activation = "tanh"

        self.architecture = Sequential([
            Dense(10, input_shape=(self.recursive_depth, len(input_indices),)),
            # Activation('relu'),
            Dropout(0.1),
            LSTM(3),
            # Activation('relu'),
            Dense(5),
            # Activation('relu'),
            Dense(1),
            Activation(self.activation)
        ])

        self.architecture.compile(optimizer = self.optimizer,
                                    loss = self.loss,
                                    metrics = self.metrics)




