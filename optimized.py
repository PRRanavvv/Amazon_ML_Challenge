import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
# Install dependencies (uncomment in Colab)
# !pip install -q tensorflow tensorflow_hub tensorflow_addons matplotlib pandas scikit-learn pillow

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image
import time
import json

# Detect GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"  {gpu}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("Running on CPU")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Environment setup complete.\n")

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

CONFIG = {
    'IMAGE_SIZE': 224,
    'BATCH_SIZE': 64,
    'EMBEDDING_DIM': 1280,  # EfficientNetB4 output dim
    'BACKBONE': 'efficientnet_b4',  # 'efficientnet_b4', 'vit', or 'mobilenet'
    'TRAIN_SIZE': 255900,
    'TEST_SIZE': 90600,
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-3,
    'DROPOUT_RATE': 0.3,
    'LOSS_FN': 'huber',  # 'huber' or 'mae' for noisy data
    'DATA_DIR': './data',
    'EMBEDDINGS_DIR': './embeddings',
    'MODEL_SAVE_PATH': './models/regression_head.h5',
}

os.makedirs(CONFIG['DATA_DIR'], exist_ok=True)
os.makedirs(CONFIG['EMBEDDINGS_DIR'], exist_ok=True)
os.makedirs('./models', exist_ok=True)

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================================================
# 3. DATA GENERATION (DUMMY DATA FOR TESTING)
# ============================================================================
# Replace this section with your actual CSV + image directory in production

def create_dummy_dataset():
    """Generate dummy train/test CSVs and random images for testing."""
    print("Generating dummy dataset...")
    
    # Create dummy images directory
    train_img_dir = Path(CONFIG['DATA_DIR']) / 'train_images'
    test_img_dir = Path(CONFIG['DATA_DIR']) / 'test_images'
    train_img_dir.mkdir(exist_ok=True)
    test_img_dir.mkdir(exist_ok=True)
    
    # Generate dummy train CSV
    np.random.seed(SEED)
    n_train = min(500, CONFIG['TRAIN_SIZE'])  # Use 500 for quick testing
    train_ids = np.arange(n_train)
    train_prices = np.random.uniform(10, 1000, n_train) + np.random.normal(0, 50, n_train)
    train_prices = np.clip(train_prices, 1, 10000)
    
    train_df = pd.DataFrame({
        'id': train_ids,
        'image_path': [f'train_images/{i}.jpg' for i in train_ids],
        'price': train_prices,
    })
    train_df.to_csv(Path(CONFIG['DATA_DIR']) / 'train.csv', index=False)
    
    # Generate dummy test CSV
    n_test = min(200, CONFIG['TEST_SIZE'])  # Use 200 for quick testing
    test_ids = np.arange(n_test) + n_train
    test_prices = np.random.uniform(10, 1000, n_test) + np.random.normal(0, 50, n_test)
    test_prices = np.clip(test_prices, 1, 10000)
    
    test_df = pd.DataFrame({
        'id': test_ids,
        'image_path': [f'test_images/{i}.jpg' for i in range(n_test)],
        'price': test_prices,
    })
    test_df.to_csv(Path(CONFIG['DATA_DIR']) / 'test.csv', index=False)
    
    # Generate dummy images (random noise)
    print(f"  Creating {n_train} train images...")
    for i in range(n_train):
        img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        img.save(train_img_dir / f'{i}.jpg')
    
    print(f"  Creating {n_test} test images...")
    for i in range(n_test):
        img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        img.save(test_img_dir / f'{i}.jpg')
    
    print(f"Dummy dataset created: {n_train} train, {n_test} test\n")
    return train_df, test_df

# ============================================================================
# 4. PRECOMPUTE EMBEDDINGS (OFFLINE)
# ============================================================================

def load_and_preprocess_image(image_path, image_size=224):
    """Load and preprocess a single image."""
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (image_size, image_size))
        img = img / 255.0
        return img
    except:
        # Return blank image on error
        return tf.zeros((image_size, image_size, 3))

def create_embedding_model(backbone_name, image_size=224):
    """Create embedding extraction model."""
    if backbone_name == 'efficientnet_b4':
        url = "https://tfhub.dev/google/imagenet/efficientnet_b4/feature_vector/1"
        embedding_dim = 1280
    elif backbone_name == 'efficientnet_b3':
        url = "https://tfhub.dev/google/imagenet/efficientnet_b3/feature_vector/1"
        embedding_dim = 1536
    elif backbone_name == 'mobilenet':
        url = "https://tfhub.dev/google/imagenet/mobilenet_v3_large/feature_vector/1"
        embedding_dim = 1280
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    hub_module = hub.KerasLayer(url, trainable=False)
    embeddings = hub_module(inputs)
    model = tf.keras.Model(inputs, embeddings)
    
    print(f"Embedding model created: {backbone_name} ({embedding_dim}-dim)")
    return model, embedding_dim

def precompute_embeddings(csv_path, embedding_model, subset_name='train', batch_size=64):
    """Extract and save embeddings for all images in dataset."""
    print(f"\nPrecomputing {subset_name} embeddings...")
    
    df = pd.read_csv(csv_path)
    base_dir = Path(csv_path).parent
    
    image_paths = [base_dir / img_path for img_path in df['image_path']]
    prices = df['price'].values
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths.astype(str), prices.astype(np.float32))
    )
    dataset = dataset.map(
        lambda img_path, price: (
            load_and_preprocess_image(img_path, CONFIG['IMAGE_SIZE']),
            price
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Extract embeddings in batches
    all_embeddings = []
    all_prices = []
    
    for batch_imgs, batch_prices in dataset:
        batch_embeddings = embedding_model(batch_imgs, training=False).numpy()
        all_embeddings.append(batch_embeddings)
        all_prices.append(batch_prices.numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    prices = np.concatenate(all_prices, axis=0)
    
    # Save to disk
    emb_path = Path(CONFIG['EMBEDDINGS_DIR']) / f'{subset_name}_embeddings.npz'
    np.savez_compressed(emb_path, embeddings=embeddings, prices=prices)
    
    print(f"Saved {subset_name} embeddings: {embeddings.shape}")
    print(f"  File: {emb_path} ({emb_path.stat().st_size / 1e6:.1f} MB)")
    
    return embeddings, prices

# ============================================================================
# 5. REGRESSION MODEL
# ============================================================================

def create_regression_head(embedding_dim, dropout_rate=0.3):
    """Build lightweight MLP regression head."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1),  # Regression output
    ])
    return model

def compile_model(model, loss_fn='huber', learning_rate=1e-3):
    """Compile regression model with robust loss."""
    if loss_fn == 'huber':
        loss = tf.keras.losses.Huber(delta=100.0)  # Robust to outliers
    elif loss_fn == 'mae':
        loss = 'mae'
    else:
        loss = 'mse'
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])
    return model

# ============================================================================
# 6. EMBEDDING DATASET LOADER (MEMORY-EFFICIENT)
# ============================================================================

def load_embeddings_dataset(embeddings_path, batch_size=64, augment=False):
    """Load precomputed embeddings and return tf.data.Dataset."""
    data = np.load(embeddings_path)
    embeddings = data['embeddings'].astype(np.float32)
    prices = data['prices'].astype(np.float32)
    
    # Optional: add Gaussian noise to embeddings for robustness (pseudo-augmentation)
    if augment:
        noise = np.random.normal(0, 0.01, embeddings.shape).astype(np.float32)
        embeddings = embeddings + noise
    
    dataset = tf.data.Dataset.from_tensor_slices((embeddings, prices))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ============================================================================
# 7. TRAINING
# ============================================================================

def train_regression_head(
    train_embeddings_path,
    val_embeddings_path=None,
    model_save_path='./models/regression_head.h5',
    epochs=50,
    batch_size=64,
):
    """Train regression head on precomputed embeddings."""
    print("\n" + "="*60)
    print("TRAINING REGRESSION HEAD")
    print("="*60 + "\n")
    
    # Load training data
    train_dataset = load_embeddings_dataset(
        train_embeddings_path, batch_size=batch_size, augment=True
    )
    
    # Load validation data (optional split)
    if val_embeddings_path:
        val_dataset = load_embeddings_dataset(val_embeddings_path, batch_size=batch_size)
    else:
        val_dataset = None
    
    # Create model
    model = create_regression_head(CONFIG['EMBEDDING_DIM'], CONFIG['DROPOUT_RATE'])
    model = compile_model(model, loss_fn=CONFIG['LOSS_FN'], learning_rate=CONFIG['LEARNING_RATE'])
    
    print(model.summary())
    print()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss' if val_dataset else 'loss',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_dataset else 'loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_dataset else 'loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    return model, history

# ============================================================================
# 8. INFERENCE
# ============================================================================

def predict_on_embeddings(model, embeddings_path, batch_size=64):
    """Generate predictions using precomputed embeddings."""
    print(f"\nGenerating predictions from {embeddings_path}...")
    
    data = np.load(embeddings_path)
    embeddings = data['embeddings'].astype(np.float32)
    prices_true = data['prices'].astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    t0 = time.time()
    predictions = model.predict(dataset, verbose=0)
    predictions = predictions.flatten()
    elapsed = time.time() - t0
    
    print(f"Predictions generated in {elapsed:.2f}s")
    print(f"  Throughput: {len(predictions) / elapsed:.0f} samples/sec")
    
    return predictions, prices_true

# ============================================================================
# 9. EVALUATION
# ============================================================================

def evaluate_predictions(predictions, prices_true):
    """Compute evaluation metrics."""
    mae = mean_absolute_error(prices_true, predictions)
    rmse = np.sqrt(mean_squared_error(prices_true, predictions))
    mape = np.mean(np.abs((prices_true - predictions) / (prices_true + 1))) * 100
    median_ae = np.median(np.abs(prices_true - predictions))
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"  MAE:           ${mae:.2f}")
    print(f"  RMSE:          ${rmse:.2f}")
    print(f"  MAPE:          {mape:.2f}%")
    print(f"  Median AE:     ${median_ae:.2f}")
    print("="*60 + "\n")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'median_ae': median_ae}

def plot_predictions(predictions, prices_true, n_samples=1000):
    """Plot predicted vs true prices."""
    idx = np.random.choice(len(predictions), min(n_samples, len(predictions)), replace=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(prices_true[idx], predictions[idx], alpha=0.5, s=10)
    max_price = max(prices_true[idx].max(), predictions[idx].max())
    axes[0].plot([0, max_price], [0, max_price], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True Price ($)')
    axes[0].set_ylabel('Predicted Price ($)')
    axes[0].set_title(f'Predicted vs True Prices (n={len(idx)})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    residuals = predictions[idx] - prices_true[idx]
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', label='Zero error')
    axes[1].set_xlabel('Residual (Predicted - True)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./models/predictions_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to ./models/predictions_plot.png\n")

# ============================================================================
# 10. MAIN PIPELINE
# ============================================================================

def main():
    """Execute full pipeline."""
    print("\n" + "="*60)
    print("IMAGE-TO-PRICE REGRESSION PIPELINE")
    print("="*60 + "\n")
    
    # 1. Generate or load dataset
    train_df, test_df = create_dummy_dataset()
    
    # 2. Create embedding model
    embedding_model, embedding_dim = create_embedding_model(CONFIG['BACKBONE'])
    CONFIG['EMBEDDING_DIM'] = embedding_dim
    
    # 3. Precompute embeddings
    train_embeddings_path = Path(CONFIG['EMBEDDINGS_DIR']) / 'train_embeddings.npz'
    test_embeddings_path = Path(CONFIG['EMBEDDINGS_DIR']) / 'test_embeddings.npz'
    
    if not train_embeddings_path.exists():
        precompute_embeddings(
            Path(CONFIG['DATA_DIR']) / 'train.csv',
            embedding_model,
            subset_name='train',
            batch_size=CONFIG['BATCH_SIZE'],
        )
    else:
        print("Train embeddings already exist, skipping...")
    
    if not test_embeddings_path.exists():
        precompute_embeddings(
            Path(CONFIG['DATA_DIR']) / 'test.csv',
            embedding_model,
            subset_name='test',
            batch_size=CONFIG['BATCH_SIZE'],
        )
    else:
        print("Test embeddings already exist, skipping...")
    
    # 4. Train regression head
    model, history = train_regression_head(
        train_embeddings_path,
        val_embeddings_path=None,
        model_save_path=CONFIG['MODEL_SAVE_PATH'],
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
    )
    
    # 5. Inference on test set
    predictions, prices_true = predict_on_embeddings(
        model, test_embeddings_path, batch_size=CONFIG['BATCH_SIZE']
    )
    
    # 6. Evaluate
    metrics = evaluate_predictions(predictions, prices_true)
    plot_predictions(predictions, prices_true, n_samples=min(1000, len(predictions)))
    
    # 7. Save results
    results = {
        'config': CONFIG,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'model_path': CONFIG['MODEL_SAVE_PATH'],
    }
    with open('./models/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Pipeline complete! Results saved to ./models/results.json\n")
    return model

if __name__ == '__main__':
    model = main()
