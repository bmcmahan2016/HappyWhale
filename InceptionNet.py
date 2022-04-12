import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow.keras as keras

class InceptionNet():
    '''
    creates a GoogLeNet using inception blocks
    '''
    def __init__(self):
        self._l2_penalty = 0.0 # hparams[HP_l2_penalty]

    def input_block(self, img):
        '''
        constructs the input blocks prior to inception blocks
        '''
        # CONVOLUTION
        X = tfl.Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(img)
        #X = tf.ensure_shape(X, [None, 112, 112, 64])

        # MAX POOL
        X = tfl.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(X)
        #X = tf.cast(X, tf.float32)
        #X = tf.nn.local_response_normalization(X)
        sqr_sum = tf.math.reduce_sum(X**2, axis=-1, keepdims=True)
        X = X / (1 + 1*sqr_sum)**0.5
        #X = tf.cast(X, tf.float16)

        # CONVOLUTION
        X = tfl.Conv2D(64, (1,1), activation='relu')(X)
        X = tfl.Conv2D(192, (3,3), padding='same', activation='relu')(X)
        #X = tf.ensure_shape(X, [None, 56, 56, 192])
        #X = tf.cast(X, tf.float32)
        sqr_sum = tf.math.reduce_sum(X**2, axis=-1, keepdims=True)
        X = X / (1 + 1*sqr_sum)**0.5
        #X = tf.cast(X, tf.float16)

        # MAX POOL
        X = tfl.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', dtype=tf.float16)(X)
        #X = tf.ensure_shape(X, [None, 28, 28, 192])
        return X

    def channel_concat(self, inception_heads):
        '''
        concatenates the outputs of the different inception block heads

        Arguments:
        inception_heads -- list of tensor outputs from each inception head.
        Each tensor should have shape (batch_size, height, widht, channels).
        '''
        return tf.concat(inception_heads, axis=-1)  #concatenates tensors along channel dimension

    def inception_block(self, prev_activations, n_filters=[64, 96, 128, 16, 32, 32]):
        '''
        Performs a forward pass of an inception block
        Arguments
        num_filters -- list of how many filters to use for each convolution. First
        number specifies the number of filters to use for the 1x1 convolution
        1x1 filters
        #3x3 reduce
        3x3
        #5x5 reduce
        5x5
        pool projection
        '''
        l2_penalty = self._l2_penalty
        # 1x1 conv 
        output0 = tfl.Conv2D(n_filters[0], (1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l2=l2_penalty), activation='relu')(prev_activations)

        # 1x1 conv --> 3x3 conv 
        output1 = tfl.Conv2D(n_filters[1], (1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l2=l2_penalty), activation='relu')(prev_activations)
        output1 = tfl.Conv2D(n_filters[2], (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(l2=l2_penalty), activation='relu')(output1)
        # 1x1 conv --> 5x5 conv
        output2 = tfl.Conv2D(n_filters[3], (1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l2=l2_penalty), activation='relu')(prev_activations)
        output2 = tfl.Conv2D(n_filters[4], (5,5), padding='same', kernel_regularizer=keras.regularizers.l2(l2=l2_penalty), activation='relu')(output2)
        # maxpool (3x3, s=1) --> 1x1 conv
        output3 = tfl.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(prev_activations)
        output3 = tfl.Conv2D(n_filters[5], (1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l2=l2_penalty), activation='relu')(output3)

        # channel concatenate the inception outputs
        return self.channel_concat([output0, output1, output2, output3])

    def auxillary_head(self, output, name=0):
        '''
        intermediate level output from the network
        '''
        output = tfl.AveragePooling2D(pool_size=(5,5), strides=(3,3))(output)
        output = tfl.Conv2D(128, (1,1), padding='same', activation='relu')(output)              # 128 filters in auxillary output
        output = tfl.Flatten()(output)
        output = tfl.Dense(1024, activation='relu')(output)                  # 1024 FC units
        output = tfl.Dropout(0.7)(output)                                    # 70% of units dropped in auxillary heads
        output_name = "output" + str(name)
        output = tfl.Dense(128, activation='softmax', name=output_name)(output)                # number of classes
        return output

    def output_head(self, output):
        '''
        main model output, this should be used for inference
        '''
        output = tfl.AveragePooling2D(pool_size=(7,7), strides=(1,1))(output)   
        #output = tf.ensure_shape(output, [None, 1, 1, 1024])
        output = tfl.Flatten()(output)
        output = tfl.Dropout(0.4)(output)       # 40% of units dropped from main head
        output = tfl.Dense(128, activation='softmax', name='output2')(output)
        return output

    def assemble_full_model(self, input_shape=(256, 256, 3), classes=128):
        l2_penalty = self._l2_penalty
        X_input = tfl.Input(input_shape)
        # INPUT BLOCK
        X = self.input_block(X_input)
        #X = tf.ensure_shape(X, [None, 28, 28, 192])

        # INCEPTION BLOCK 3
        X = self.inception_block(X, n_filters=[64, 96, 128, 16, 32, 32])
        #X = tf.ensure_shape(X, [None, 28, 28, 256])
        X = self.inception_block(X, n_filters=[128, 128, 192, 32, 96, 64])
        #X = tf.ensure_shape(X, [None, 28, 28, 480])

        X = tfl.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='same')(X)
        #X = tf.ensure_shape(X, [None, 14, 14, 480])

        # INCEPTION BLOCK 4
        X = self.inception_block(X, n_filters=[192, 96, 208, 16, 48, 64])
        #X = tf.ensure_shape(X, [None, 14, 14, 512])
        output0 = self.auxillary_head(X)
        X = self.inception_block(X, n_filters=[160, 112, 224, 24, 64, 64])
        #X = tf.ensure_shape(X, [None, 14, 14, 512])
        X = self.inception_block(X, n_filters=[128, 128, 256, 24, 64, 64])
        #X = tf.ensure_shape(X, [None, 14, 14, 512])
        X = self.inception_block(X, n_filters=[112, 114, 288, 32, 64, 64])
        #X = tf.ensure_shape(X, [None, 14, 14, 528])
        output1 = self.auxillary_head(X, name=1)
        X = self.inception_block(X, n_filters=[256, 160, 320, 32, 128, 128])
        #X = tf.ensure_shape(X, [None, 14, 14, 832])
        
        X = tfl.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='same')(X)
        #X = tf.ensure_shape(X, [None, 7, 7, 832])

        # INCEPTION BLOCK 5
        # need to insert a max pool layer here
        X = self.inception_block(X, n_filters=[256, 160, 320, 32, 128, 128])
        #X = tf.ensure_shape(X, [None, 7, 7, 832])
        X = self.inception_block(X, n_filters=[384, 192, 384, 48, 128, 128])
        #X = tf.ensure_shape(X, [None, 7, 7, 1024])
        output2 = self.output_head(X)

        GoogLeNet = keras.Model(inputs=X_input, outputs=[output0, output1, output2], name="GoogLeNet")
        return GoogLeNet