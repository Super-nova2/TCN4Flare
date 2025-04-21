import numpy as np
from utils import *
import os
import tensorflow as tf
from tcn import TCN # keras-tcn version==3.5.4
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam, Nadam # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
import tensorflow_addons as tfa
import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



# 定义自定义指标F1Score
class F1Score(tf.keras.metrics.Metric):
    """
    Custom F1 score metric for model evaluation.
    """
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

# 定义自定义回调函数, recalls >= 0.75 and f1 > best_f1
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom model checkpoint callback for saving the best model based on recall and F1 score .
    """
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_f1 = 0  # best F1 score

    def on_epoch_end(self, epoch, logs=None):
        current_recall = logs.get('val_recall')
        current_f1 = logs.get('val_f1')

        # judge if current model is better than the best model
        if current_recall >= 0.75 and current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.model.save(self.filepath)
            print(f'\nSave model: Recall={current_recall:.4f}, F1={current_f1:.4f}')
        else:
            print(f'\nModel not improved.')

class TCN_model():
    """
    定义TCN模型类,包含模型构建，训练，预测，评估等功能
    TCN model class, including model construction, training, prediction, and evaluation functions.
    Parameters:
        model_path: model save path \n
        model_name: model name, default is 'TCN4Flare_128_3_256_1_0.20_10:1.keras', 
        recommened to use this format 'TCN4Flare_num_filters_kernel_size_dilations_nb_stacks_dropout_rate_10:1.keras'
    """
    def __init__(self, model_path=r'./TCN4Flare/model/', model_name='TCN4Flare_128_3_256_1_0.2_10:1.keras'):
        self.model_path = model_path
        self.model_name = model_name
        self.model = tf.keras.models.load_model(os.path.join(self.model_path, self.model_name),custom_objects={'TCN': TCN, 'F1Score':F1Score})
        self.model.summary()

    def predict(self, dataset, thereshold=0.6225):
        """
        预测数据集generator中的数据,返回预测结果,y_pred为预测类别0/1,y_prob为预测概率[0,1]
        Args:
            dataset: datasetr
            thereshold: thereshold for binary classification, default is 0.6225.
        Returns:
            y_pred: predicted class, 0/1
            y_prob: predicted probability, [0,1]
        """
        y_prob = self.model.predict(get_x_normed(dataset))
        y_pred = (y_prob > thereshold).astype(int)
        return y_pred, y_prob
    

    def build_custom_tcn_model(self, input_shape=(None, 2), num_filters=128, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256], dropout_rate=0.20, nb_stacks=1):
        """
        构建 TCN 模型; build a custom TCN model with specified hyperparameters.
        Args:
            input_shape: 输入数据的形状 (seq_len, num_features), shape of input data, default is (None, 2)
            num_filters: 每层的卷积核数, number of filters in each layer, default is 128
            kernel_size: 卷积核大小, number of kernels, default is 3
            dilations: 膨胀系数列表, list of dilations, default is [1, 2, 4, 8, 16, 32, 64, 128, 256]
            dropout_rate: Dropout 概率, default is 0.20
            nb_stacks: TCN 堆叠层数, number of stacks, default is 1
        Note:
            默认参数为训练好的默认模型, default model is the best model trained on the dataset.
        Returns:
            model: 构建好的 Keras 模型, a Keras model.
        """
        inputs = Input(shape=input_shape)
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
            use_layer_norm=True,    # use layer normlization
            use_weight_norm=False)(inputs)
        outputs = Dense(1, activation="sigmoid")(tcn_output)  # 二分类输出, output with sigmoid activation function
        model = Model(inputs, outputs)
        # 定义优化器时加入梯度裁剪, define optimizer with gradient clipping, default using Nadam optimizer, clipnorm=1.0
        optimizer = Nadam(
            learning_rate=1e-4, # 初始学习率建议较低, initial learning rate is suggested to be low
            clipnorm=1.0        # 按范数裁剪梯度, clip gradient by norm
            # # clipvalue=0.5      # 或按绝对值裁剪, clip gradient by value
        )
        # 使用 TensorFlow Addons Focal loss to handle class imbalance
        loss = tfa.losses.SigmoidFocalCrossEntropy(
            alpha=0.9,  # 正类（10%）的权重, ratio of negetive class
            gamma=3.0
        )
        # define metrics, including accuracy, F1 score, precision, recall
        model.compile(optimizer=optimizer, loss=loss,
                    metrics=["accuracy", F1Score(), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
        model.summary()
        # define model name
        model_name = f'TCN4Flare_{num_filters}_3_{dilations[-1]}_{nb_stacks}_{dropout_rate}_10:1' # 定义模型名称, 根据设置的超参数和数据集比例进行命名
        
        self.model = model
        self.model_name = model_name
        
        return model, model_name


    def train(self, train_dataset, train_labels, valid_dataset, valid_labels, model_path=r'./TCN4Flare/model/', log_dir=r'./TCN4Flare/logs/', epochs=100, early_stop = True):
        """
        训练模型，返回训练好的模型
        train_dataset, valid_dataset为训练集和验证集数据集,数据集格式为numpy数组,shape为(N,T,2)
        checkpoint_path为模型保存路径,log_dir为tensorboard日志路径,epochs为训练轮数,early_stop为是否使用early stopping
        train the model, return the trained model.
        Args:
            train_dataset: training dataset, numpy array, shape (N,T,2)
            train_labels: training labels, numpy array, shape (N,)
            valid_dataset: validation dataset, numpy array, shape (N,T,2)
            valid_labels: validation labels, numpy array, shape (N,)
            model_path: model save path, default is './TCN4Flare/model/'
            log_dir: tensorboard log path, default is './TCN4Flare/logs/'
            epochs: number of epochs, default is 100
            early_stop: whether to use early stopping, default is True
        Returns:
            model: trained Keras model.
        """

        # define data generator
        def train_data_generator():
            return get_x_y(train_dataset, train_labels)
        def valid_data_generator():
            return get_x_y(valid_dataset, valid_labels)
        
        # train_data_generator = get_x_y(train_dataset, train_labels)
        # valid_data_generator = get_x_y(valid_dataset, valid_labels)
        train_dataset_from_generator = tf.data.Dataset.from_generator(
            train_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(1,None,2), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.int32)
            )
        )

        # 定义模型保存依据, define custom model checkpoint based on F1 score and recall 
        n_batches = train_dataset.shape[0]
        model_callback = CustomModelCheckpoint(model_path)
        # model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
        #                                                  monitor='val_f1',
        #                                                  save_best_only=True, 
        #                                                  verbose=1,
        #                                                  mode='max',
        #                                                  save_freq='epoch')
        
        # tensorboard log callback
        log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + self.model_name
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # earlystopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',   # 监控验证集损失
            patience=20,           # 允许连续 5 个 epoch 无改善
            min_delta=0.0005,      # 认为“提升”的最小变化阈值
            mode='min',           # 监控指标的方向（这里是损失越小越好）
            restore_best_weights=True  # 恢复最佳 epoch 的模型权重
        )

        # 定义回调函数列表, define callback list
        callbacks = [tensorboard_callback, model_callback, early_stopping] if early_stop else [tensorboard_callback, model_callback]

        # train the model
        self.model.fit(  train_dataset_from_generator,
                validation_data=valid_data_generator(),
                epochs=100,
                validation_steps=valid_dataset.shape[0],
                steps_per_epoch=n_batches,
                callbacks=callbacks,
                verbose=2,
                # class_weight={0: 1, 1: 9}  # 采用focal loss， 无需设置样本权重, already using focal loss, no need to set sample weights
                )
        
        return self.model
    
    def evaluate(self, test_dataset, test_labels, plot_cm=True, plot_roc=False, plot_pr=True):
        """
        评估模型在测试集上的性能，返回评估结果
        test_dataset为测试集数据集,数据集格式为numpy数组,shape为(N,T,2)
        plot_cm, plot_roc, plot_pr为是否绘制混淆矩阵,ROC曲线,PR曲线
        evaluate the model on the test set, return the evaluation results.
        Args:
            test_dataset: test dataset, numpy array, shape (N,T,2)
            test_labels: test labels, numpy array, shape (N,)
            plot_cm: whether to plot confusion matrix, default is True
            plot_roc: whether to plot ROC curve, default is False
            plot_pr: whether to plot PR curve, default is True
        Returns:
            None
        Note:
            the ploted figure will be saved in the './fig/'.
        """

        # define data generator
        test_data_generator = get_x_normed(test_dataset)

        # 使用模型进行预测，得到预测概率, predict probabilities using the model
        y_prob= self.model.predict(test_data_generator)

        # 计算不同阈值下的Precision和Recall, calculate Precision and Recall with different thresholds
        precision, recall, thresholds = precision_recall_curve(test_labels, y_prob)
        if plot_pr:
            # plot PR curve
            pr_aur = auc(recall, precision)
            plt.figure(figsize=(5,5))
            plt.plot(recall, precision, label=f"Model (AUR = {pr_aur:.3f})")
            plt.plot([0, 1], [1, 0], 'k--')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.legend(loc="lower left")
            plt.savefig(r'./fig/PR_curve.png', dpi=300)
            plt.show()
                
        # 寻找最大化F1 Score的阈值, find the threshold that maximizes F1 Score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        optimal_idx = np.argmax(f1_scores)
        # optimal_idx = np.where(np.abs(recall - 0.9).argmin())
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        optimal_recall = recall[optimal_idx]
        optimal_precision = precision[optimal_idx]
        print('The optimal threshold to maximize F1 Score is:', optimal_threshold)
        print('The optimal F1 Score is:', optimal_f1)
        print('The optimal Recall is:', optimal_recall)
        print('The optimal Precision is:', optimal_precision)

        if plot_cm:
            # 绘制混淆矩阵, plot confusion matrix
            y_pred = (y_prob > optimal_threshold).astype(int)
            cm = confusion_matrix(test_labels, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no Flare', 'Flare'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix')
            plt.savefig(r'./fig/Confusion_Matrix.png', dpi=300)
            plt.show()

        if plot_roc:
            # 绘制ROC曲线, plot ROC curve
            fpr, tpr, thresholds = roc_curve(test_labels, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr, label=f"Model (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')
            plt.legend(loc="lower right")
            plt.savefig(r'./fig/ROC_curve.png', dpi=300)
            plt.show()

    def compare_models(self, model_lists, test_dataset, test_labels, plot_cm=True, plot_roc=False, plot_pr=True, fig_name = 'compare_models'):
        """
        比较多个模型在测试集上的性能，返回评估结果
        model_lists为模型列表,每个模型都需要包含模型路径和模型名称,如{'./TCN4Flare/model/TCN4Flare_128_3_256_1_0.2_10:1.keras': 'TCN4Flare_128_3_256_1_0.2_10:1'}
        test_dataset为测试集数据集,数据集格式为numpy数组,shape为(N,T,2)
        plot_cm, plot_roc, plot_pr为是否绘制混淆矩阵,ROC曲线,PR曲线
        compare the performance of multiple models on the test set, return the evaluation results.
        Args:
            model_lists: model list, each model should contain model name,model should be saved in './model/' e.g. {'TCN4Flare_128_3_256_1_0.2_10:1', 'TCN4Flare_64_3_256_1_0.2_10:1'}
            test_dataset: test dataset, numpy array, shape (N,T,2)
            test_labels: test labels, numpy array, shape (N,)
            plot_cm: whether to plot confusion matrix, default is True
            plot_roc: whether to plot ROC curve, default is False
            plot_pr: whether to plot PR curve, default is True
        Returns:
            None
        Note:
            the ploted figure will be saved in the current directory.
        """

        # load models
        models = []
        for model_name in model_lists:
            try:
                model = tf.keras.models.load_model(f'./model/{model_name}.keras', custom_objects={'TCN': TCN, 'F1Score': F1Score})
                models.append(model)
                print(f"成功加载模型, load model successfully: {model_name}")
            except Exception as e:
                print(f"加载模型 {model_name} 时出错, load model {model_name} failed: {e}")

        fig_nums = (int(plot_roc) + int(plot_pr))
        fig, axes = plt.subplots(1, fig_nums, figsize=(5*fig_nums, 5))

        # calculate metrics for each model
        for i, model in enumerate(models):
            # 预测概率
            y_prob = model.predict(get_x_normed(test_dataset))
            # 计算不同阈值下的Precision和Recall, calculate Precision and Recall with different thresholds
            precision, recall, thresholds = precision_recall_curve(test_labels, y_prob)

            # 绘制PR曲线, plot PR curve
            if plot_pr:

                pr_aur = auc(recall, precision)
                axes[0].plot(recall, precision, label=f"{model_lists[i]} (AUR = {pr_aur:.3f})")
                if i == len(models) - 1:
                    axes[0].plot([0, 1], [1, 0], 'k--')
                    axes[0].set_xlabel('Recall')
                    axes[0].set_ylabel('Precision')
                    axes[0].set_title('Precision-Recall curve')
                    axes[0].legend(loc="lower left")
                

            # 寻找最大化F1 Score的阈值, find the threshold that maximizes F1 Score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            optimal_idx = np.argmax(f1_scores)
            # optimal_idx = np.where(np.abs(recall - 0.9).argmin())
            optimal_threshold = thresholds[optimal_idx]
            optimal_f1 = f1_scores[optimal_idx]
            optimal_recall = recall[optimal_idx]
            optimal_precision = precision[optimal_idx]
            print(f'The optimal threshold to maximize F1 Score for {model_lists[i]} is:', optimal_threshold)
            print(f'The optimal F1 Score for {model_lists[i]} is:', optimal_f1)
            print(f'The optimal Recall for {model_lists[i]} is:', optimal_recall)
            print(f'The optimal Precision for {model_lists[i]} is:', optimal_precision)
            
            # 绘制ROC曲线, plot ROC curve
            if plot_roc:

                fpr, tpr, thresholds = roc_curve(test_labels, y_prob)
                roc_auc = auc(fpr, tpr)
                axes[1].plot(fpr, tpr, label=f"{model_lists[i]} (AUC = {roc_auc:.3f})")
                if i == len(models) - 1:
                    axes[1].plot([0, 1], [0, 1], 'k--')
                    axes[1].set_xlabel('False Positive Rate')
                    axes[1].set_ylabel('True Positive Rate')
                    axes[1].set_title('ROC curve')
                    axes[1].legend(loc="lower right")


            if plot_cm:
                # 绘制混淆矩阵, plot confusion matrix
                y_pred = (y_prob > optimal_threshold).astype(int)
                cm = confusion_matrix(test_labels, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no Flare', 'Flare'])
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix for {model_lists[i]}')
                plt.savefig(f'./fig/Confusion_Matrix_{model_lists[i]}.png', dpi=300)

        if not plot_cm and not plot_roc and not plot_pr:
            print("没有绘制任何图形, no figure is plotted.")
        else:
            fig.savefig(f'./fig/compare_models_{fig_name}.png', dpi=300)
            plt.show()