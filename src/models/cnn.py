import tensorflow as tf
from tensorflow.keras import layers, models
class CNNArchitect:
    def __init__(self, input_shape, num_classes, conv_filters, kernel_size, pool_size, dense_units, dropout_rate):
        """
        Khởi tạo kiến trúc 13 lớp.
        :param conv_filters: list các số filter, theo paper là phải truyền như này conv_filters=[64, 128, 256, 256]
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()

        # 1. Batch Normalization (Layer 1)
        model.add(layers.BatchNormalization(input_shape=self.input_shape))

        # 2-9. Lặp qua các khối Conv + Pool (Layer 2 đến 9)
        for filters in self.conv_filters:
            model.add(layers.Conv2D(filters=filters, 
                                    kernel_size=self.kernel_size, 
                                    activation='relu', 
                                    padding='same'))
            model.add(layers.MaxPooling2D(pool_size=self.pool_size))

        # 10. Flatten (Layer 10)
        model.add(layers.Flatten())

        # 11. Dense (Layer 11)
        model.add(layers.Dense(self.dense_units, activation='relu'))

        # 12. Dropout (Layer 12)
        model.add(layers.Dropout(self.dropout_rate))

        # 13. Output Layer (Layer 13)
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return model