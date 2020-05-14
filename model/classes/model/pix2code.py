from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'


import keras
from keras.layers import Input, Dense, Dropout, \
                         RepeatVector, LSTM, concatenate, \
                         Conv2D, MaxPooling2D, Flatten#, CuDNNLSTM
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

# to setup tensorboard: https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/#implementing-tensorboard-into-your-keras-model
# We also import the Keras backend for choosing our channels first / channels last approach.
# We import TensorBoard from the Keras callbacks.
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from time import time

from .Config import *
from .AModel import *
from .coordconv import CoordinateChannel2D

#coordconv added or not
is_coordconv = True
# validation split = 0.2 ??
class pix2code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "pix2code"
        self.checkpoint_path=f'{output_path}/checkpoints/testmodel.h5'

        image_model = Sequential()

        init_input_shape = input_shape
        #input_shape changes because of coordconv
        if is_coordconv:
            input_shape = list(input_shape)
            input_shape[2] = 5
            input_shape = tuple(input_shape)

        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=init_input_shape)
        encoded_image = None

        #Add Coord layer before first Conv4
        x = CoordinateChannel2D()(visual_input)
        encoded_image = image_model(x) if is_coordconv else image_model(visual_input)

        language_model = Sequential()
        language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([encoded_image, encoded_text])

        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # Compile  the model

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Define Tensorboard as a Keras callback

        tensorboard = TensorBoard(
        log_dir= "{}/logs".format(self.output_path),
        histogram_freq=0,
        write_images=True
        )
        self.keras_callbacks = [
            tensorboard,
            EarlyStopping(monitor='loss', patience=5, mode='min', min_delta=0.0001),
            ModelCheckpoint("{}/checkpoints/testmodel.h5".format(self.output_path), monitor='loss', save_best_only=True, mode='min')
            ]

    def fit(self, images, partial_captions, next_words):
        self.model.fit([images, partial_captions], next_words,
                        shuffle=False,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_split= 0.2,
                        callbacks=self.keras_callbacks)
        self.save()

    def fit_generator(self, generator, steps_per_epoch):
        self.model.fit_generator(generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=EPOCHS,
                                verbose=1,
                                callbacks=self.keras_callbacks)
        self.save()

# # Generate generalization metrics
# score = model.evaluate(input_test, target_test, verbose=0)
    # def evaluate_generator(self, generator,
    #                        steps=None,
    #                        callbacks=None,
    #                        max_queue_size=10,
    #                        workers=1,
    #                        use_multiprocessing=False,
    #                        verbose=0):
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)
