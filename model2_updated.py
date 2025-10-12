#!/usr/bin/env python3
"""
Optimized Multimodal Embedding → Regression Workflow
========================================================
CPU/MacBook-friendly pipeline for price prediction using precomputed embeddings.

Features:
- Efficient image streaming (tf.data or PyTorch)
- Offline multimodal embedding computation (CLIP/ViT)
- Small MLP regressor trained on frozen embeddings
- Robust loss handling (Huber, MAE) for noisy labels
- Optional semi-supervised pseudo-labeling
- Fast inference on 95k test images

Requirements:
pip install tensorflow numpy pandas pillow torch torchvision transformers scikit-learn tqdm

For Mac GPU (optional):
pip install --upgrade tensorflow-macos tensorflow-metal
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset',
    'csv_train': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\train.csv',
    'csv_test': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\test.csv',
    'embeddings_dir': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\embeddings',
    'embeddings_train_npy': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\embeddings\train_embeddings.npy',
    'embeddings_test_npy': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\embeddings\test_embeddings.npy',
    'text_embeddings_train': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\embeddings\text_train_embeddings.npy',
    'text_embeddings_test': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\embeddings\text_test_embeddings.npy',
    'output_predictions': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\predictions.csv',
    'best_model_path': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\best_model.h5',
    
    # Model config
    'image_size': 224,
    'batch_size_precompute': 32,  # For embedding computation
    'batch_size_train': 64,        # For MLP training
    'embedding_dim': 512,          # CLIP/ViT embedding dimension
    'use_text': True,              # Use text embeddings if available
    'text_dim': 384,               # Text embedding dimension (all-MiniLM-L6-v2 outputs 384)
    'mlp_hidden_dim': 256,
    'mlp_dropout': 0.3,
    
    # Training
    'epochs': 50,
    'learning_rate': 1e-3,
    'patience': 10,
    'use_pseudo_labeling': False,  # Enable for semi-supervised
    'pseudo_label_threshold': 0.7,
    'loss_fn': 'huber',            # 'huber', 'mse', 'mae'
    'log_transform_price': True,   # Stabilize skewed distributions
    'price_clip_percentile': (1, 99),  # Remove outliers
    
    # Hardware
    'use_gpu': False,              # Set True if GPU available
    'num_workers': 4,
}

os.makedirs(CONFIG['embeddings_dir'], exist_ok=True)


# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_image(image_path, image_size=224, augment=False):
    """Load and preprocess image to [0,1] range."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        if augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_array = np.fliplr(img_array)
            
            # Random rotation [-10, 10] degrees
            angle = np.random.uniform(-10, 10)
            from scipy import ndimage
            img_array = ndimage.rotate(img_array, angle, reshape=False, order=1)
            
            # Random brightness/contrast adjustment
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            img_array = np.clip(img_array * contrast + brightness - 1, 0, 1)
        
        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def create_tf_dataset(csv_path, data_dir, batch_size, augment=False, shuffle=True):
    """Create tf.data.Dataset for streaming images."""
    df = pd.read_csv(csv_path)
    
    def load_fn(idx):
        row = df.iloc[idx.numpy()]
        image_path = os.path.join(data_dir, row['image_path'])
        img = load_and_preprocess_image(image_path, CONFIG['image_size'], augment=augment)
        
        if img is None:
            img = np.zeros((CONFIG['image_size'], CONFIG['image_size'], 3), dtype=np.float32)
        
        price = float(row['price'])
        img_id = str(row['id'])
        
        return img, price, img_id
    
    dataset = tf.data.Dataset.from_tensor_slices(np.arange(len(df)))
    dataset = dataset.map(
        lambda idx: tf.py_function(load_fn, [idx], [tf.float32, tf.float32, tf.string]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(df)))
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, df


# ============================================================================
# 2. PRECOMPUTE MULTIMODAL EMBEDDINGS (OFFLINE)
# ============================================================================

def precompute_embeddings_clip(csv_path, data_dir, output_path, batch_size=32):
    """Precompute image embeddings using CLIP model (offline)."""
    print(f"\n{'='*70}")
    print(f"Precomputing image embeddings: {csv_path}")
    print(f"{'='*70}")
    
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers -q")
        from transformers import CLIPProcessor, CLIPModel
    
    device = "mps" if CONFIG['use_gpu'] and tf.test.is_gpu_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    df = pd.read_csv(csv_path)
    embeddings = []
    failed_indices = []
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        images = []
        
        for _, row in batch_df.iterrows():
            image_path = os.path.join(data_dir, row['image_path'])
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((CONFIG['image_size'], CONFIG['image_size']))
                images.append(img)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                failed_indices.append(i + len(images))
                images.append(Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size'])))
        
        # Compute embeddings
        with tf.no_grad() if hasattr(tf, 'no_grad') else no_grad_context():
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if k != 'pixel_values'}
            inputs['pixel_values'] = inputs.get('pixel_values', processor(images=images).pixel_values).to(device)
            
            try:
                import torch
                with torch.no_grad():
                    img_embeds = model.get_image_features(**inputs).cpu().numpy()
            except:
                img_embeds = model.get_image_features(**inputs).detach().cpu().numpy()
        
        embeddings.append(img_embeds)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(df))}/{len(df)} images")
    
    embeddings = np.vstack(embeddings)
    np.save(output_path, embeddings)
    print(f"✓ Saved {len(embeddings)} image embeddings to {output_path}")
    
    return embeddings, failed_indices


# CHANGE 1: Line 213 - Added 'catalog_content' as first element in text_columns
def precompute_text_embeddings(csv_path, output_path, text_columns=['catalog_content', 'category', 'brand']):
    """Precompute text embeddings if metadata available."""
    print(f"\n{'='*70}")
    print(f"Precomputing text embeddings: {csv_path}")
    print(f"Trying to use columns: {text_columns}")
    print(f"{'='*70}")
    
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers -q")
        from transformers import AutoTokenizer, AutoModel
    
    df = pd.read_csv(csv_path)
    
    # Check if text columns exist
    available_cols = [col for col in text_columns if col in df.columns]
    if not available_cols:
        print(f"No text columns {text_columns} found in CSV. Skipping text embeddings.")
        return None
    
    print(f"✓ Found columns: {available_cols}")
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    
    device = "mps" if CONFIG['use_gpu'] else "cpu"
    model = model.to(device)
    
    embeddings = []
    
    # CHANGE 2: Added progress tracking for text embedding
    from tqdm import tqdm
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing text"):
        # Concatenate available text columns
        text_parts = []
        for col in available_cols:
            if pd.notna(row[col]):
                text_parts.append(str(row[col]))
        
        text = " ".join(text_parts) if text_parts else "unknown"
        
        # CHANGE 3: Line 239 - Increased max_length from 128 to 512
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with tf.no_grad() if hasattr(tf, 'no_grad') else no_grad_context():
            try:
                import torch
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Mean pooling for better representation
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            except:
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        
        embeddings.append(embedding)
    
    embeddings = np.vstack(embeddings)
    np.save(output_path, embeddings)
    print(f"✓ Saved {len(embeddings)} text embeddings to {output_path}")
    print(f"  Embedding shape: {embeddings.shape}")
    
    return embeddings


def no_grad_context():
    """Dummy context manager for compatibility."""
    class NoGrad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NoGrad()


# ============================================================================
# 3. PRICE PREPROCESSING & OUTLIER REMOVAL
# ============================================================================

def preprocess_prices(df, clip_percentile=(1, 99), log_transform=True):
    """Clean and preprocess prices."""
    prices = df['price'].values.astype(float)
    
    # Remove NaN
    valid_mask = ~np.isnan(prices)
    prices_valid = prices[valid_mask]
    
    # Clip outliers
    if clip_percentile:
        low, high = np.percentile(prices_valid, clip_percentile)
        prices = np.clip(prices, low, high)
        print(f"Clipped prices to [{low:.2f}, {high:.2f}]")
    
    # Log transform to stabilize distribution
    if log_transform:
        prices = np.log1p(prices)
        print(f"Applied log1p transform to prices")
    
    return prices, prices_valid.min(), prices_valid.max()


# ============================================================================
# 4. BUILD MLP REGRESSOR
# ============================================================================

# CHANGE 4: Updated default text_dim to 384 (all-MiniLM-L6-v2 output dimension)
def build_mlp_regressor(embedding_dim, use_text=False, text_dim=384):
    """Build lightweight MLP for regression on embeddings."""
    input_dim = embedding_dim
    if use_text:
        input_dim += text_dim
    
    print(f"Building MLP with input_dim={input_dim} (image={embedding_dim}, text={text_dim if use_text else 0})")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(CONFIG['mlp_hidden_dim'], activation='relu'),
        tf.keras.layers.Dropout(CONFIG['mlp_dropout']),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(CONFIG['mlp_dropout']),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    
    return model


def get_loss_fn(loss_type='huber'):
    """Get robust loss function for noisy labels."""
    if loss_type == 'huber':
        return tf.keras.losses.Huber(delta=1.0)
    elif loss_type == 'mae':
        return tf.keras.losses.MeanAbsoluteError()
    else:
        return tf.keras.losses.MeanSquaredError()


# ============================================================================
# 5. TRAINING WITH CALLBACKS
# ============================================================================

def train_mlp(train_embeddings, train_prices, val_split=0.1, text_embeddings=None):
    """Train MLP regressor on frozen embeddings."""
    print(f"\n{'='*70}")
    print(f"Training MLP Regressor")
    print(f"{'='*70}")
    
    # Combine image + text embeddings
    if text_embeddings is not None and CONFIG['use_text']:
        X = np.hstack([train_embeddings, text_embeddings])
        print(f"Combined embeddings shape: {X.shape}")
        print(f"  - Image embeddings: {train_embeddings.shape}")
        print(f"  - Text embeddings: {text_embeddings.shape}")
    else:
        X = train_embeddings
        print(f"Using only image embeddings: {X.shape}")
    
    y = train_prices.reshape(-1, 1)
    
    # Split train/val
    n_train = int(len(X) * (1 - val_split))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
    
    # Build model
    model = build_mlp_regressor(
        CONFIG['embedding_dim'],
        use_text=CONFIG['use_text'] and text_embeddings is not None,
        text_dim=CONFIG['text_dim']
    )
    
    loss_fn = get_loss_fn(CONFIG['loss_fn'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mae']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            './best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size_train'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"✓ Training complete. Best val_loss: {min(history.history['val_loss']):.4f}")
    
    return model, history


# ============================================================================
# 6. INFERENCE ON TEST SET
# ============================================================================

def inference_on_test(model, test_embeddings, test_csv, text_embeddings=None, 
                      price_min=None, price_max=None):
    """Fast inference on test set."""
    print(f"\n{'='*70}")
    print(f"Running Inference on Test Set")
    print(f"{'='*70}")
    
    # Combine embeddings
    if text_embeddings is not None and CONFIG['use_text']:
        X_test = np.hstack([test_embeddings, text_embeddings])
        print(f"Combined test embeddings shape: {X_test.shape}")
    else:
        X_test = test_embeddings
        print(f"Using only image embeddings: {X_test.shape}")
    
    # Predict
    predictions = model.predict(X_test, batch_size=256, verbose=1)
    predictions = predictions.flatten()
    
    # Inverse log transform if applied
    if CONFIG['log_transform_price']:
        predictions = np.expm1(predictions)
    
    # Clip to observed range
    if price_min is not None and price_max is not None:
        predictions = np.clip(predictions, price_min, price_max)
    
    # Load test CSV and create submission
    df_test = pd.read_csv(test_csv)
    df_test['predicted_price'] = predictions
    
    output_path = './predictions.csv'
    df_test[['id', 'predicted_price']].to_csv(output_path, index=False)
    print(f"✓ Saved predictions to {output_path}")
    print(f"  Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return df_test


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def main():
    """Execute full pipeline."""
    
    print("\n" + "="*70)
    print("MULTIMODAL EMBEDDING → REGRESSION WORKFLOW")
    print("="*70)
    
    # Step 1: Load training data
    print(f"\n[1/6] Loading training data...")
    df_train = pd.read_csv(CONFIG['csv_train'])
    print(f"  ✓ Loaded {len(df_train)} training samples")
    print(f"  Available columns: {list(df_train.columns)}")
    
    # Step 2: Preprocess prices
    print(f"\n[2/6] Preprocessing prices...")
    train_prices, price_min, price_max = preprocess_prices(
        df_train,
        clip_percentile=CONFIG['price_clip_percentile'],
        log_transform=CONFIG['log_transform_price']
    )
    print(f"  ✓ Price range (after transform): [{train_prices.min():.4f}, {train_prices.max():.4f}]")
    
    # Step 3: Precompute embeddings (if not already done)
    print(f"\n[3/6] Precomputing embeddings...")
    
    if not os.path.exists(CONFIG['embeddings_train_npy']):
        train_embeddings, _ = precompute_embeddings_clip(
            CONFIG['csv_train'],
            CONFIG['data_dir'],
            CONFIG['embeddings_train_npy'],
            batch_size=CONFIG['batch_size_precompute']
        )
    else:
        train_embeddings = np.load(CONFIG['embeddings_train_npy'])
        print(f"  ✓ Loaded precomputed train embeddings: {train_embeddings.shape}")
    
    if not os.path.exists(CONFIG['embeddings_test_npy']):
        test_embeddings, _ = precompute_embeddings_clip(
            CONFIG['csv_test'],
            CONFIG['data_dir'],
            CONFIG['embeddings_test_npy'],
            batch_size=CONFIG['batch_size_precompute']
        )
    else:
        test_embeddings = np.load(CONFIG['embeddings_test_npy'])
        print(f"  ✓ Loaded precomputed test embeddings: {test_embeddings.shape}")
    
    # Step 4: Precompute text embeddings (optional)
    text_train_embeddings = None
    text_test_embeddings = None
    
    if CONFIG['use_text']:
        # CHANGE 5: Lines 483-488 - Explicitly specify text columns including catalog_content
        if not os.path.exists(CONFIG['text_embeddings_train']):
            text_train_embeddings = precompute_text_embeddings(
                CONFIG['csv_train'],
                CONFIG['text_embeddings_train'],
                text_columns=['catalog_content', 'category', 'brand']
            )
        else:
            text_train_embeddings = np.load(CONFIG['text_embeddings_train'])
            print(f"  ✓ Loaded train text embeddings: {text_train_embeddings.shape}")
        
        # CHANGE 6: Lines 491-496 - Explicitly specify text columns for test set
        if not os.path.exists(CONFIG['text_embeddings_test']):
            text_test_embeddings = precompute_text_embeddings(
                CONFIG['csv_test'],
                CONFIG['text_embeddings_test'],
                text_columns=['catalog_content', 'category', 'brand']
            )
        else:
            text_test_embeddings = np.load(CONFIG['text_embeddings_test'])
            print(f"  ✓ Loaded test text embeddings: {text_test_embeddings.shape}")
    
    # Step 5: Train MLP
    print(f"\n[4/6] Training MLP regressor...")
    model, history = train_mlp(
        train_embeddings,
        train_prices,
        val_split=0.1,
        text_embeddings=text_train_embeddings
    )
    
    # Step 6: Inference
    print(f"\n[5/6] Running inference on test set...")
    df_predictions = inference_on_test(
        model,
        test_embeddings,
        CONFIG['csv_test'],
        text_embeddings=text_test_embeddings,
        price_min=price_min,
        price_max=price_max
    )
    
    print(f"\n[6/6] Pipeline complete!")
    print(f"{'='*70}")
    print(f"Results:")
    print(f"  - Predictions saved: ./predictions.csv")
    print(f"  - Model saved: ./best_model.h5")
    print(f"  - Embedding cache: {CONFIG['embeddings_dir']}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
