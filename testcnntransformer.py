import pprint          # for pretty printing
from select_feature import data_x,data_y
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
RANDOM_STATE = 0
import os
import tensorflow as tf
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

pp = pprint.PrettyPrinter()

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
TEST_SIZE = 0.1
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=TEST_SIZE,
                                                  random_state=RANDOM_STATE)
X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)
print("X Train shape: ", X_train.shape)
print("X Test shape: ", X_test.shape)
print("X val shape: ",X_val.shape)
print("Y Train shape: ", y_train.shape)
print("Y test shape: ",y_test.shape)
print("Y val shape:",y_val.shape)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_val = np.reshape(X_val,(X_val.shape[0],X_val.shape[1],1))

# print('Training size: ' + str(X_train.shape))
# print('Validation size: ' + str(X_val.shape))
# print('Test size: ' + str(X_test.shape))
#
# print('Training size: ' + str(y_train.shape))
# print('Validation size: ' + str(y_val.shape))
# print('Test size: ' + str(y_test.shape))
from tensorflow import keras

from keras.optimizers import Adam

from keras.models import *
from keras.layers import *

import tensorflow as tf
import numpy as np
from keras import backend as K

embed_size = 60


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config.update({
            "eps": self.eps,
        })
        return config

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
        self.__name__ = ''

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask

def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

def CnnTransformerModel():
    i = Input(X_train.shape[1:])
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(i)
    x = Convolution1D(8, kernel_size=3, strides=2, activation='relu')(x)


    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(x)

    x = Convolution1D(16, kernel_size=3, strides=2, activation='relu')(x)

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(x)

    x = Convolution1D(32, kernel_size=3, strides=2, activation='relu')(x)

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(x)

    x = Convolution1D(64, kernel_size=3, strides=2, activation='relu')(x)
    x = Dense(units=64,activation='gelu')(x)
    x, slf_attn = MultiHeadAttention(n_head=5, d_model=300, d_k=60, d_v=60, dropout=0.15)(x, x, x)

    avg_pool = GlobalAveragePooling1D()(x)

    avg_pool = Dense(60, activation='relu')(avg_pool)

    y = Dense(1, activation='sigmoid')(avg_pool)

    return Model(inputs=[i], outputs=[y])


model = CnnTransformerModel()
model.compile(optimizer=Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Recall', 'Precision'])
model.summary()
# # from IPython.display import SVG
# # from keras.utils.vis_utils import model_to_dot
# # SVG(model_to_dot(model,show_shapes = True).create(prog='dot', format='svg'))

def create_callbacks(best_model_filepath, tensorboard_logs_filepath):
    callback_checkpoint = ModelCheckpoint(filepath=best_model_filepath,
                                          monitor='val_loss',
                                          verbose=0,
                                          save_weights_only=False,
                                          save_best_only=False,
                                          mode="auto",
                                          save_freq="epoch")

    callback_early_stopping = EarlyStopping(monitor="val_loss",
                                            min_delta=0,
                                            patience=150,
                                            mode="auto",
                                            restore_best_weights=False)

    callback_tensorboard = TensorBoard(log_dir=tensorboard_logs_filepath,
                                       histogram_freq=0,
                                       write_graph=False)

    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-5,
                                           patience=20,
                                           verbose=1)

    return [callback_checkpoint, callback_early_stopping,
            callback_tensorboard,callback_reduce_lr]
# callbacks = [
#     keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=100,mode="auto",restore_best_weights=False
#         # # Stop training when `val_loss` is no longer improving
#         # monitor="val_loss",
#         # # "no longer improving" being defined as "no better than 1e-2 less"
#         # min_delta=1e-4,
#         # # "no longer improving" being further defined as "for at least 2 epochs"
#         # patience=50,
#         # verbose=1,
#     )
#     ,keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
#                                            factor=0.1,
#                                            min_lr=1e-3,
#                                            patience=20,
#                                            verbose=1)
# ]
from sklearn.utils.class_weight import compute_class_weight

EPOCHS = 300
BATCH_SIZE = 64
best_model_filepath = "CNN1D_transformer_Model.ckpt"
tensorboard_logs_filepath = "./CNN1D_transformer_logs/"

# calculate the class weights
class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(y_train),y=y_train)
class_weights = {i : class_weights[i] for i in range(2)}
# class_weights = dict(zip(np.unique(y_train), class_weights))
history_1D = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data = (X_val, y_val),
                       callbacks= create_callbacks(best_model_filepath,tensorboard_logs_filepath) ,class_weight = class_weights,verbose=1)
model.save('CNN1D-Transformers.h5')
# # Support Vector Machine
import matplotlib.pyplot as plt

def plot_progress(history_dict):
    for key in list(history_dict.keys())[:4]:
        plt.clf()  # Clears the figure
        training_values = history_dict[key]
        val_values = history_dict['val_' + key]
        epochs = range(1, len(training_values) + 1)

        plt.plot(epochs, training_values, 'bo', label='Training ' + key)

        plt.plot(epochs, val_values, 'b', label='Validation ' + key)

        if key != 'loss':
            plt.ylim([0., 1.1])

        plt.title('Training and Validation ' + key)

        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
        plt.show()

plot_progress(history_1D.history)
