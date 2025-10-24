# ==============================================================================
# FILE: TRAIN.PY (DEFINITIVE, CLEANED & CORRECTED)
# ==============================================================================
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_nlp
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import os
import json

# --- 1. Global Setup ---
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- 2. Configuration ---
DATA_DIRECTORY = '/kaggle/working/processed_symptoms'
TRAIN_DATA_FILENAME = 'train_symptoms.csv'
EVAL_DATA_FILENAME = 'eval_symptoms.csv'
MODEL_SAVE_DIRECTORY = '/kaggle/working/saved_models'
FINAL_MODEL_NAME = 'symptom_classifier_model_roberta_final.keras'
HISTORY_FILENAME = 'training_history_roberta_final.json'

# --- Model Hyperparameters ---
BATCH_SIZE = 8
MAX_SEQUENCE_LENGTH = 96
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 0.5

# --- Learning Rates & Epochs (ULTRA-CONSERVATIVE) ---
HEAD_EPOCHS = 10
HEAD_LEARNING_RATE = 2e-5
FULL_FINETUNE_EPOCHS = 20
FULL_FINETUNE_LEARNING_RATE = 1e-7
WARMUP_RATIO = 0.2
MIN_LEARNING_RATE = 1e-9

# --- 3. Model Architecture Visualization ---
def visualize_model_architecture(model, save_path=None):
    """Visualize and analyze the model architecture."""
    print("\n🏗️ === MODEL ARCHITECTURE ANALYSIS ===")
    model.summary(line_length=100)
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"📈 Trainable params: {trainable_params:,} | Non-trainable params: {non_trainable_params:,}")
    if save_path:
        try:
            tf.keras.utils.plot_model(model, to_file=save_path, show_shapes=True, expand_nested=True, dpi=96)
            print(f"💾 Architecture diagram saved to: {save_path}")
        except Exception as e:
            print(f"⚠️ Could not save diagram. Install pydot and graphviz. Error: {e}")
    print("🏗️ === END ARCHITECTURE ANALYSIS ===\n")

# --- 4. Custom Model Architecture ---
def build_classifier_model(num_classes):
    """Builds a robust RoBERTa model with an advanced classification head."""
    # Define inputs
    token_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='token_ids')
    padding_mask = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='padding_mask')

    # Reusable Backbone Layer
    backbone = keras_nlp.models.RobertaBackbone.from_preset('roberta_base_en')
    
    # Get output from the backbone
    sequence_output = backbone({'token_ids': token_ids, 'padding_mask': padding_mask})
    cls_output = sequence_output[:, 0, :]  # Use the [CLS] token's output

    # --- CORRECTED & SIMPLIFIED: Enhanced Classification Head ---
    def enhanced_residual_block(x, units, name_prefix):
        shortcut = x
        if x.shape[-1] != units:
            shortcut = tf.keras.layers.Dense(units, name=f'{name_prefix}_shortcut')(shortcut)
            shortcut = tf.keras.layers.LayerNormalization(epsilon=1e-6)(shortcut)
        
        x = tf.keras.layers.Dense(units, name=f'{name_prefix}_dense_1')(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Activation('gelu')(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        return tf.keras.layers.Add()([shortcut, x])

    x = cls_output
    x = enhanced_residual_block(x, 768, 'block1') # Pass through first block
    x = enhanced_residual_block(x, 512, 'block2') # Pass through second block
    
    # CORRECTED: Final output layer logic
    output_units = 1 if num_classes == 2 else num_classes
    activation_func = 'sigmoid' if num_classes == 2 else 'softmax'
    
    outputs = tf.keras.layers.Dense(
        output_units,
        activation=activation_func,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name='classifier_output'
    )(x)
    
    # CORRECTED: The model uses the backbone layer, and we return the layer itself
    model = tf.keras.Model(inputs={'token_ids': token_ids, 'padding_mask': padding_mask}, outputs=outputs)
    return model, backbone

# --- 5. Custom Learning Rate Scheduler ---
class WarmupPolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, end_learning_rate=0.0, power=1.0, name=None):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.end_learning_rate = tf.cast(end_learning_rate, tf.float32)
        self.power = tf.cast(power, tf.float32)
        self.name = name
    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupPolynomialDecay"):
            warmup_steps_float = tf.maximum(1.0, self.warmup_steps)
            global_step_float = tf.cast(step, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done
            is_warmup = global_step_float < self.warmup_steps
            decay_duration = tf.maximum(1.0, self.decay_steps - self.warmup_steps)
            steps_after_warmup = global_step_float - self.warmup_steps
            decay_factor = tf.pow(tf.maximum(0.0, 1.0 - (steps_after_warmup / decay_duration)), self.power)
            decayed_learning_rate = ((self.initial_learning_rate - self.end_learning_rate) * decay_factor) + self.end_learning_rate
            learning_rate = tf.where(is_warmup, warmup_learning_rate, decayed_learning_rate)
            return tf.maximum(learning_rate, self.end_learning_rate)
    def get_config(self):
        return {k: getattr(self, k) for k in ["initial_learning_rate", "decay_steps", "warmup_steps", "end_learning_rate", "power", "name"]}

# --- 6. Helper Functions ---
def get_optimizer(lr_schedule):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, beta_2=0.98, epsilon=1e-6, global_clipnorm=MAX_GRAD_NORM, amsgrad=True)
    return tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

def get_loss_function(num_classes):
    return tf.keras.losses.BinaryCrossentropy() if num_classes == 2 else tf.keras.losses.SparseCategoricalCrossentropy()

def get_metrics(num_classes):
    return [tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name='auc')] if num_classes == 2 else ["accuracy"]

def get_callbacks(model_save_path, patience, monitor):
    return [
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor=monitor, mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, mode='min', verbose=1),
        tf.keras.callbacks.TerminateOnNaN()
    ]

# --- 7. Main Training Pipeline ---
def main():
    """Executes the complete, stabilized training and saving pipeline."""
    print("🚀 --- Starting Model Training (Definitive Corrected Version) ---")

    # [1/5] Load & Preprocess Data
    print("\n📊 Loading and preprocessing data...")
    train_df = pd.read_csv(os.path.join(DATA_DIRECTORY, TRAIN_DATA_FILENAME))
    eval_df = pd.read_csv(os.path.join(DATA_DIRECTORY, EVAL_DATA_FILENAME))
    preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset("roberta_base_en", sequence_length=MAX_SEQUENCE_LENGTH)
    X_train = preprocessor(train_df['processed_text'].astype(str).tolist())
    X_eval = preprocessor(eval_df['processed_text'].astype(str).tolist())
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['department'])
    y_eval = label_encoder.transform(eval_df['department'])
    num_classes = len(label_encoder.classes_)
    os.makedirs(MODEL_SAVE_DIRECTORY, exist_ok=True)
    np.save(os.path.join(MODEL_SAVE_DIRECTORY, 'classes.npy'), label_encoder.classes_)
    print(f"✅ Data loaded. Found {num_classes} classes: {list(label_encoder.classes_)}")
    class_weights_dict = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
    print(f"⚖️ Calculated Class Weights: {class_weights_dict}")

    # [2/5] Build & Visualize Model
    print("\n🏗️ Building and visualizing the model...")
    model, backbone = build_classifier_model(num_classes)
    visualize_model_architecture(model, os.path.join(MODEL_SAVE_DIRECTORY, 'model_architecture.png'))
    
    # [3/5] Stage 1: Head Training
    print("\n🎯 Starting STAGE 1: Training the classification head...")
    backbone.trainable = False
    steps_per_epoch = len(train_df) // BATCH_SIZE
    head_lr_schedule = WarmupPolynomialDecay(initial_learning_rate=HEAD_LEARNING_RATE, decay_steps=(steps_per_epoch * HEAD_EPOCHS), warmup_steps=(int(steps_per_epoch * HEAD_EPOCHS * WARMUP_RATIO)), end_learning_rate=MIN_LEARNING_RATE)
    model.compile(optimizer=get_optimizer(head_lr_schedule), loss=get_loss_function(num_classes), metrics=get_metrics(num_classes))
    
    temp_model_path = os.path.join(MODEL_SAVE_DIRECTORY, 'temp_best_head.keras')
    history_head = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=HEAD_EPOCHS, validation_data=(X_eval, y_eval), callbacks=get_callbacks(temp_model_path, patience=4, monitor='val_loss'), class_weight=class_weights_dict, verbose=1)
    print("✅ --- Stage 1 Complete ---")

    # [4/5] Stage 2: Full Fine-Tuning
    print("\n🔧 Starting STAGE 2: Ultra-conservative full model fine-tuning...")
    model.load_weights(temp_model_path) # Load the BEST weights from Stage 1
    
    backbone.trainable = True
    # Gradual Unfreezing: Freeze the first half of the transformer layers
    num_transformer_layers = len([l for l in backbone.layers if "transformer_layer" in l.name])
    for i, layer in enumerate(backbone.layers):
        if "transformer_layer" in layer.name and i < num_transformer_layers // 2:
            layer.trainable = False
    print("🔓 Gradual Unfreezing Complete. Re-visualizing trainable parameters...")
    visualize_model_architecture(model)

    finetune_lr_schedule = WarmupPolynomialDecay(initial_learning_rate=FULL_FINETUNE_LEARNING_RATE, decay_steps=(steps_per_epoch * FULL_FINETUNE_EPOCHS), warmup_steps=(int(steps_per_epoch * FULL_FINETUNE_EPOCHS * WARMUP_RATIO)), end_learning_rate=MIN_LEARNING_RATE)
    model.optimizer.learning_rate = finetune_lr_schedule # Update LR without recompiling
    print(f"✅ Optimizer LR updated for Stage 2. Peak LR: {FULL_FINETUNE_LEARNING_RATE}")
    
    best_model_path = os.path.join(MODEL_SAVE_DIRECTORY, FINAL_MODEL_NAME)
    history_fine_tune = model.fit(
        X_train, y_train, batch_size=BATCH_SIZE,
        initial_epoch=history_head.epoch[-1] + 1,
        epochs=HEAD_EPOCHS + FULL_FINETUNE_EPOCHS,
        validation_data=(X_eval, y_eval),
        callbacks=get_callbacks(best_model_path, patience=8, monitor='val_loss'),
        class_weight=class_weights_dict, verbose=1
    )
    print("✅ --- Stage 2 Complete ---")

    # [5/5] Save Final Artifacts
    print("\n💾 Saving final artifacts...")
    full_history = {k: v + history_fine_tune.history.get(k, []) for k, v in history_head.history.items()}
    serializable_history = {k: [float(i) for i in v] for k, v in full_history.items()}
    history_path = os.path.join(MODEL_SAVE_DIRECTORY, HISTORY_FILENAME)
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)

    print("\n🎉 --- ✅ TRAINING COMPLETE ---")
    print(f"🏆 Best model saved to: '{best_model_path}'")
    print(f"📊 Full training history saved to: '{history_path}'")

if __name__ == '__main__':
    main()