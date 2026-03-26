import os
import uuid
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from dataloader import get_data_loaders, create_dataset

class AudioModelTrainer:
    def __init__(self, data_dir, feature_extractor, input_shape, model_name, num_classes=50, batch_size=32):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.input_shape = input_shape
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        """
        session_id: tạo 1 chuỗi id random để ghép với thư mục checkpoint
        Khi tìm được best model (ModelCheckpoint) --> lưu trữ các parameters vào folder checkpoint theo id riêng
        """
        self.session_id = str(uuid.uuid4())[:8]
        self.checkpoint_dir = os.path.join("checkpoints", self.model_name, self.session_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.train_ds = None
        self.val_ds = None
        self.model = None

    def prepare_data(self):
        self.train_ds, self.val_ds = get_data_loaders(
            data_dir=self.data_dir,
            feature_extractor=self.feature_extractor,
            batch_size=self.batch_size
        )

    def model_cnn(self, learning_rate=0.001):
        """
        Build and compile Deep CNN model:
            Layer 1: Batch Normalization
            Layer 2+3: Conv2D (64) + MaxPool (2x2)
            Layer 4+5: Conv2D (128) + MaxPool (2x2)
            Layer 6+7: Conv2D (256) + MaxPool (2x2)
            Layer 8+9: Conv2D (256) + MaxPool (2x2)
            Layer 10->13: Flatten + Dense(256) + Dropout(0.5) + Dense(Output)
        """
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((1,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.summary()

    def _get_callbacks(self):
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor = 0.1, patience=2, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience=6, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(self.checkpoint_dir, "CNN_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
        ]
        return callbacks

    def train(self, num_epochs=50):
        """
        Using MLFlow Tracking to train model CNN
        """      
        mlflow.set_experiment("ESC50_Audio_Classification")
        mlflow.tensorflow.autolog() 

        with mlflow.start_run(run_name=f"{self.model_name}_{self.session_id}"):
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=num_epochs,
                callbacks=self._get_callbacks()
            )
        print("Save model: ",self.checkpoint_dir)
        return history