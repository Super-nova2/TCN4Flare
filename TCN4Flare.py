import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam, Nadam # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tcn import TCN
import datetime
import tensorflow_addons as tfa

# 使用 TensorFlow Addons
loss = tfa.losses.SigmoidFocalCrossEntropy(
    alpha=0.99,  # 正类（1%）的权重
    gamma=3.0
)


# 定义模型
def build_tcn_model(input_shape, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=0.05, nb_stacks=2):
    """
    构建 TCN 模型
    Args:
        input_shape: 输入数据的形状 (seq_len, num_features)
        num_filters: 每层的卷积核数
        kernel_size: 卷积核大小
        dilations: 膨胀系数列表
        dropout_rate: Dropout 概率
        nb_stacks: TCN 堆叠层数
    Returns:
        model: 构建好的 Keras 模型
    """
    inputs = Input(shape=input_shape)
    # norm_output = normalizer(inputs)  # 归一化层
    tcn_output = TCN(
        nb_filters=num_filters,
        nb_stacks=nb_stacks,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout_rate=dropout_rate,
        return_sequences=False,
        use_skip_connections=True,
        kernel_initializer="he_normal",
        use_batch_norm=False,
        use_layer_norm=True,
        use_weight_norm=False,
    )(inputs)
    outputs = Dense(1, activation="sigmoid")(tcn_output)  # 二分类输出
    model = Model(inputs, outputs)
    # 定义优化器时加入梯度裁剪
    optimizer = Nadam(
        learning_rate=1e-4,
        # learning_rate=1e-5,  # 初始学习率建议较低
        clipnorm=1.0        # 按范数裁剪梯度
        # # clipvalue=0.5      # 或按绝对值裁剪
    )
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=["accuracy", F1Score(), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model

# 定义自定义指标F1Score
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-6))  # 防止除零

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# 定义数据生成器
def get_x_y(data, labels, mode='train'):
    for k in range(int(1e9)):
        max_len = data.shape[0]
        x = data[k % max_len, :, 1:]
        x = x[~np.isnan(x).any(axis=1)]
        mean = np.mean(x, axis=0)[0]    
        std = np.mean(x, axis=0)[0]
        x[:,0] = (x[:,0] - mean) / std  # 归一化
        x[:,1] = x[:,1]/std  # 归一化
        x = np.expand_dims(x, axis=0)
        y = labels[k % max_len]
        if k % 10000 == 0 and mode == 'train':
            print('Proceed {:d}e4 samples'.format(k // 10000))
        yield x, np.expand_dims(y, axis=-1)

def validation_data_generator():
    return get_x_y(val_data, val_labels, mode='val')

def train_data_generator():
    return get_x_y(train_data, train_labels, mode='train')

# 加载数据, 并划分训练集和验证集
file_path = r'./TCN4Flare/final_train_data/'
data_all = np.load(os.path.join(file_path, 'train_data_99:1.npz'))   # 训练集, 比例为99:1
data = data_all['data']
labels = data_all['labels']
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)  # 验证集, 比例为0.2:0.8

# 创建tf.data.Dataset对象
train_dataset = tf.data.Dataset.from_generator(
    train_data_generator,
    output_signature=(
        tf.TensorSpec(shape=(1,None,2), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )
)
# 设置多线程处理
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# 构建模型, 需要调整参数
input_shape = (None, 2)  # 时间序列长度和特征维度
nb_filters = 64  # 卷积核数    64, 128
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # 膨胀系数列表    32, 64, 128, 256
nb_stacks = 1  # TCN 堆叠层数   8, 4, 2, 1
dropout_rate = 0.2  # Dropout 概率

model = build_tcn_model(input_shape,
                        num_filters=nb_filters,     #64, 128
                        kernel_size=3,  # 卷积核大小
                        dilations=dilations, #32, 64, 128, 256
                        dropout_rate=dropout_rate,  # Dropout 概率
                        nb_stacks=nb_stacks)     #8, 4, 2, 1
model.summary()
model_name = f'TCN4Flare_{nb_filters}_3_{dilations[-1]}_{nb_stacks}_{dropout_rate}_99:1' # 定义模型名称, 根据设置的超参数和数据集比例进行命名

# 定义回溯点
checkpoint_path = r"./TCN4Flare/model/{}.keras".format(model_name)
n_batches = train_data.shape[0]
# 创建回溯点保存最佳模型
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_f1',
                                                 save_best_only=True, 
                                                 verbose=1,
                                                 mode='max',
                                                 save_freq='epoch')


# tensorboard log
log_dir = r"./TCN4Flare/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + model_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# earlystopping
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',   # 监控验证集损失
#     patience=30,           # 允许连续 5 个 epoch 无改善
#     min_delta=0.0005,      # 认为“提升”的最小变化阈值
#     mode='min',           # 监控指标的方向（这里是损失越小越好）
#     restore_best_weights=True  # 恢复最佳 epoch 的模型权重
# )

# train the model
model.fit(  train_dataset,
        validation_data=validation_data_generator(),
        epochs=100,
        validation_steps=val_data.shape[0],
        steps_per_epoch=n_batches,
        callbacks=[tensorboard_callback, cp_callback],
        verbose=2,
        class_weight={0: 1, 1: 99}  # 训练集采用99：1不平衡样本时，可采用此参数进行样本不均衡处理
        )

# # Save the model
# model.save(r'./TCN4Flare/model/TCN4Flare_32_3_64_1.keras')