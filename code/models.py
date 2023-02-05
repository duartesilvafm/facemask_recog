import tensorflow as tf

# constant parameters
image_shape = (64,64,1)

# convolution
units = 32
kernel_size = (3,3)
activation_conv = 'relu'

# compile parameters
loss = 'categorical_crossentropy'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
metrics = ['accuracy']


class ConvModel(tf.keras.Model):

    # initializer method
    def __init__(self):
        
        # super init
        super().__init__()
    

    def __call__(self, conv_architecture: dict, pool_architecture: dict, output_units: int=4):

        # make model as sequential
        self.model = tf.keras.Sequential()

        # add convolutional layer
        for keys_conv, keys_pooling in zip(conv_architecture, pool_architecture):

            conv_layer = conv_architecture[keys_conv]['layer']
            conv_params = conv_architecture[keys_conv]['params']

            pool_layer = pool_architecture[keys_pooling]['layer']
            pool_params = pool_architecture[keys_pooling]['params']

            self.model.add(conv_layer(**conv_params))
            self.model.add(pool_layer(**pool_params))

        # add flat layer
        self.model.add(tf.keras.layers.Flatten())

        # add Dense layer
        self.model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=output_units, activation='softmax'))

    # compile method
    def compile(self, optimizer: object, metrics: list, loss="categorical_crossentropy"):
        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)


class TransferModel(ConvModel):

    # initializer method
    def __init__(self):
        
        # super init
        super().__init__()


    # call method 
    def __call__(self, transfer_model: object, last_layer: str, \
        non_trainable:bool=True, output_units:int=4):
        
        # create attribute
        self.transfer_model=transfer_model

        # make layers non trainable
        if non_trainable:
            for layer in self.transfer_model.layers:
                layer.trainable=False

        # get the output layer
        pre_trained_output_layer = self.output_of_last_layer(layer=last_layer)

        # create model
        self.create_model(pre_trained_output_layer=pre_trained_output_layer, \
            output_units=output_units)


    def output_of_last_layer(self, layer: str):
        """
        Gets the last layer output of a model
        
        Args:
            pre_trained_model (tf.keras Model): model to get the last layer output from
            
        Returns:
            last_output: output of the model's last layer 
        """

        last_desired_layer = self.transfer_model.get_layer(layer)
        pre_trained_output_layer  = last_desired_layer.output

        return pre_trained_output_layer
    

    def create_model(self, pre_trained_output_layer: tf.keras.layers, output_units:int=4):
        
        # Flatten the output layer to 1 dimension
        x = tf.keras.layers.Flatten()(pre_trained_output_layer)

        # add one dense layer
        x = tf.keras.layers.Dense(units=128, activation='relu')(x)

        # add output layer
        x = tf.keras.layers.Dense(units=output_units, activation='softmax')(x)

        # create the complete model by using the Model class
        self.model = tf.keras.Model(inputs=self.transfer_model.input, outputs=x)


# Define a Callback class that stops training once accuracy reaches 99.9%
class MyCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True