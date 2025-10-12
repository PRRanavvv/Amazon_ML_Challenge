#!/usr/bin/env python3
"""
FIXED: Multimodal Embedding → Regression Workflow
Key fix: Corrected text embedding dimension from 512 to 384
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
# CONFIGURATION - FIXED TEXT DIMENSION
# ============================================================================

CONFIG = {
    # UPDATE THESE PATHS TO MATCH YOUR ACTUAL DATA LOCATION
    'data_dir': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset',
    'csv_train': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\train.csv',
    'csv_test': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\test.csv',
    
    # Sample limits
    'max_train_samples': 300,  # Use 300 images for training
    'max_test_samples': 100,   # Use 100 images for prediction
    
    'embeddings_dir': './embeddings',
    'embeddings_train_npy': './embeddings/train_embeddings.npy',
    'embeddings_test_npy': './embeddings/test_embeddings.npy',
    'text_embeddings_train': './embeddings/text_train_embeddings.npy',
    'text_embeddings_test': './embeddings/text_test_embeddings.npy',
    
    # Model config - FIXED!
    'image_size': 224,
    'batch_size_precompute': 32,
    'batch_size_train': 64,
    'embedding_dim': 512,          # CLIP embedding dimension
    'use_text': True,
    'text_dim': 384,               # FIXED: all-MiniLM-L6-v2 outputs 384 dims, not 512!
    'mlp_hidden_dim': 256,
    'mlp_dropout': 0.3,
    
    # Training
    'epochs': 50,
    'learning_rate': 1e-3,
    'patience': 10,
    'use_pseudo_labeling': False,
    'pseudo_label_threshold': 0.7,
    'loss_fn': 'huber',
    'log_transform_price': True,
    'price_clip_percentile': (1, 99),
    
    # Hardware
    'use_gpu': True,
    'num_workers': 4,
}

os.makedirs(CONFIG['embeddings_dir'], exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_path(row, data_dir):
    """Extract image path from CSV row."""
    if 'image_path' in row.index:
        image_ref = row['image_path']
    elif 'image_link' in row.index:
        image_ref = row['image_link']
    else:
        raise ValueError("CSV must have either 'image_path' or 'image_link' column")
    
    if '/' in str(image_ref):
        filename = str(image_ref).split('/')[-1]
    else:
        filename = str(image_ref)
    
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        base_name = filename.rsplit('.', 1)[0]
        image_path = os.path.join(data_dir, 'images', f"{base_name}{ext}")
        
        if os.path.exists(image_path):
            return image_path
        
        image_path = os.path.join(data_dir, f"{base_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    
    return None


def precompute_embeddings_clip(csv_path, data_dir, output_path, batch_size=32, max_samples=None):
    """Precompute image embeddings using CLIP model."""
    print(f"\n{'='*70}")
    print(f"Precomputing image embeddings: {csv_path}")
    print(f"{'='*70}")
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
    except ImportError:
        print("Installing transformers and torch...")
        os.system("pip install transformers torch torchvision -q")
        from transformers import CLIPProcessor, CLIPModel
        import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    df = pd.read_csv(csv_path)
    
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"  Limited to {len(df)} samples")
    
    embeddings = []
    failed_count = 0
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        images = []
        
        for _, row in batch_df.iterrows():
            image_path = get_image_path(row, data_dir)
            
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert('RGB')
                    img = img.resize((CONFIG['image_size'], CONFIG['image_size']))
                    images.append(img)
                except Exception as e:
                    failed_count += 1
                    images.append(Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size'])))
            else:
                failed_count += 1
                images.append(Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size'])))
        
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_embeds = model.get_image_features(**inputs).cpu().numpy()
        
        embeddings.append(img_embeds)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(df))}/{len(df)} images")
    
    embeddings = np.vstack(embeddings)
    np.save(output_path, embeddings)
    print(f"✓ Saved {len(embeddings)} image embeddings to {output_path}")
    print(f"  Embedding shape: {embeddings.shape}")
    if failed_count > 0:
        print(f"  ⚠ {failed_count} images failed to load")
    
    return embeddings


def precompute_text_embeddings(csv_path, output_path, max_samples=None):
    """Precompute text embeddings (384 dimensions)."""
    print(f"\n{'='*70}")
    print(f"Precomputing text embeddings: {csv_path}")
    print(f"{'='*70}")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        os.system("pip install sentence-transformers -q")
        from sentence_transformers import SentenceTransformer
    
    df = pd.read_csv(csv_path)
    
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"  Limited to {len(df)} samples")
    
    text_columns = []
    possible_columns = ['catalog_content', 'category', 'brand', 'product_name', 'description']
    
    for col in possible_columns:
        if col in df.columns:
            text_columns.append(col)
    
    if not text_columns:
        print(f"  ⚠ No text columns found. Skipping text embeddings.")
        return None
    
    print(f"  Using text columns: {text_columns}")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = []
    for _, row in df.iterrows():
        text_parts = []
        for col in text_columns:
            if pd.notna(row[col]):
                text_parts.append(str(row[col]))
        
        text = " ".join(text_parts) if text_parts else "unknown product"
        texts.append(text)
    
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    np.save(output_path, embeddings)
    print(f"✓ Saved {len(embeddings)} text embeddings to {output_path}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Dimension: {embeddings.shape[1]} (should be 384)")
    
    return embeddings


def preprocess_prices(df, clip_percentile=(1, 99), log_transform=True):
    """Clean and preprocess prices."""
    prices = df['price'].values.astype(float)
    
    valid_mask = ~np.isnan(prices)
    prices_valid = prices[valid_mask]
    
    if clip_percentile:
        low, high = np.percentile(prices_valid, clip_percentile)
        prices = np.clip(prices, low, high)
        print(f"Clipped prices to [{low:.2f}, {high:.2f}]")
    
    if log_transform:
        prices = np.log1p(prices)
        print(f"Applied log1p transform to prices")
    
    return prices, prices_valid.min(), prices_valid.max()


def build_mlp_regressor(embedding_dim, use_text=False, text_dim=384):
    """Build MLP regressor with correct dimensions."""
    input_dim = embedding_dim
    if use_text:
        input_dim += text_dim
    
    print(f"  Building MLP with input_dim={input_dim} (image:{embedding_dim} + text:{text_dim if use_text else 0})")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(CONFIG['mlp_hidden_dim'], activation='relu'),
        tf.keras.layers.Dropout(CONFIG['mlp_dropout']),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(CONFIG['mlp_dropout']),
        tf.keras.layers.Dense(1)
    ])
    
    return model


def get_loss_fn(loss_type='huber'):
    """Get robust loss function."""
    if loss_type == 'huber':
        return tf.keras.losses.Huber(delta=1.0)
    elif loss_type == 'mae':
        return tf.keras.losses.MeanAbsoluteError()
    else:
        return tf.keras.losses.MeanSquaredError()


def train_mlp(train_embeddings, train_prices, val_split=0.1, text_embeddings=None):
    """Train MLP regressor."""
    print(f"\n{'='*70}")
    print(f"Training MLP Regressor")
    print(f"{'='*70}")
    
    # Combine embeddings
    if text_embeddings is not None and CONFIG['use_text']:
        print(f"  Image embeddings: {train_embeddings.shape}")
        print(f"  Text embeddings: {text_embeddings.shape}")
        X = np.hstack([train_embeddings, text_embeddings])
        print(f"  Combined embeddings: {X.shape}")
    else:
        X = train_embeddings
        print(f"  Using only image embeddings: {X.shape}")
    
    y = train_prices.reshape(-1, 1)
    
    # Split train/val
    n_train = int(len(X) * (1 - val_split))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"  Train set: {X_train.shape}, Val set: {X_val.shape}")
    
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


def inference_on_test(model, test_embeddings, test_csv, text_embeddings=None, 
                      price_min=None, price_max=None):
    """Run inference on test set."""
    print(f"\n{'='*70}")
    print(f"Running Inference on Test Set")
    print(f"{'='*70}")
    
    if text_embeddings is not None and CONFIG['use_text']:
        # Safety check: ensure same number of samples
        if test_embeddings.shape[0] != text_embeddings.shape[0]:
            print(f"  ⚠ WARNING: Size mismatch detected!")
            print(f"    Image embeddings: {test_embeddings.shape}")
            print(f"    Text embeddings: {text_embeddings.shape}")
            
            # Truncate to minimum size
            min_size = min(test_embeddings.shape[0], text_embeddings.shape[0])
            test_embeddings = test_embeddings[:min_size]
            text_embeddings = text_embeddings[:min_size]
            print(f"  ✓ Truncated both to {min_size} samples")
        
        X_test = np.hstack([test_embeddings, text_embeddings])
        print(f"  Combined test embeddings: {X_test.shape}")
    else:
        X_test = test_embeddings
        print(f"  Using only image embeddings: {X_test.shape}")
    
    predictions = model.predict(X_test, batch_size=256, verbose=1)
    predictions = predictions.flatten()
    
    if CONFIG['log_transform_price']:
        predictions = np.expm1(predictions)
    
    if price_min is not None and price_max is not None:
        predictions = np.clip(predictions, price_min, price_max)
    
    df_test = pd.read_csv(test_csv)
    
    # CRITICAL FIX: Truncate df_test to match the number of predictions
    if len(df_test) > len(predictions):
        print(f"  ⚠ Truncating test CSV from {len(df_test)} to {len(predictions)} rows to match predictions")
        df_test = df_test.head(len(predictions))
    
    if 'sample_id' in df_test.columns:
        sample_ids = df_test['sample_id'].values
    elif 'id' in df_test.columns:
        sample_ids = df_test['id'].values
    else:
        sample_ids = np.arange(len(predictions))
    
    output_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Verify sizes match
    assert len(sample_ids) == len(predictions), f"Size mismatch: {len(sample_ids)} IDs vs {len(predictions)} predictions"
    print(f"  ✓ Created output with {len(output_df)} predictions")
    
    output_path = os.path.join(CONFIG['data_dir'], 'predictions.csv')
    output_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved predictions to {output_path}")
    print(f"  Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return df_test


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute full pipeline."""
    
    print("\n" + "="*70)
    print("FIXED: MULTIMODAL EMBEDDING → REGRESSION WORKFLOW")
    print("="*70)
    
    # Step 1: Load data
    print(f"\n[1/6] Loading training data...")
    df_train = pd.read_csv(CONFIG['csv_train'])
    print(f"  Total training samples: {len(df_train)}")
    
    if 'max_train_samples' in CONFIG and CONFIG['max_train_samples']:
        df_train = df_train.head(CONFIG['max_train_samples'])
        print(f"  ✓ Limited to {len(df_train)} training samples")
    
    # Step 2: Preprocess prices
    print(f"\n[2/6] Preprocessing prices...")
    train_prices, price_min, price_max = preprocess_prices(
        df_train,
        clip_percentile=CONFIG['price_clip_percentile'],
        log_transform=CONFIG['log_transform_price']
    )
    
    # Ensure prices match the limited sample size
    if len(train_prices) > len(df_train):
        train_prices = train_prices[:len(df_train)]
        print(f"  ✓ Truncated prices to match {len(train_prices)} samples")
    
    # Step 3: Precompute image embeddings
    print(f"\n[3/6] Precomputing image embeddings...")
    
    if not os.path.exists(CONFIG['embeddings_train_npy']):
        train_embeddings = precompute_embeddings_clip(
            CONFIG['csv_train'],
            CONFIG['data_dir'],
            CONFIG['embeddings_train_npy'],
            batch_size=CONFIG['batch_size_precompute'],
            max_samples=CONFIG.get('max_train_samples', None)
        )
    else:
        train_embeddings = np.load(CONFIG['embeddings_train_npy'])
        print(f"  ✓ Loaded precomputed train embeddings: {train_embeddings.shape}")
        # Truncate to match max_train_samples if needed
        if 'max_train_samples' in CONFIG and CONFIG['max_train_samples']:
            train_embeddings = train_embeddings[:CONFIG['max_train_samples']]
            print(f"  ✓ Truncated to {train_embeddings.shape[0]} samples")
    
    df_test = pd.read_csv(CONFIG['csv_test'])
    if 'max_test_samples' in CONFIG and CONFIG['max_test_samples']:
        df_test = df_test.head(CONFIG['max_test_samples'])
    
    if not os.path.exists(CONFIG['embeddings_test_npy']):
        test_embeddings = precompute_embeddings_clip(
            CONFIG['csv_test'],
            CONFIG['data_dir'],
            CONFIG['embeddings_test_npy'],
            batch_size=CONFIG['batch_size_precompute'],
            max_samples=CONFIG.get('max_test_samples', None)
        )
    else:
        test_embeddings = np.load(CONFIG['embeddings_test_npy'])
        print(f"  ✓ Loaded precomputed test embeddings: {test_embeddings.shape}")
        # Truncate to match max_test_samples if needed
        if 'max_test_samples' in CONFIG and CONFIG['max_test_samples']:
            test_embeddings = test_embeddings[:CONFIG['max_test_samples']]
            print(f"  ✓ Truncated to {test_embeddings.shape[0]} samples")
    
    # Step 4: Precompute text embeddings
    text_train_embeddings = None
    text_test_embeddings = None
    
    if CONFIG['use_text']:
        if not os.path.exists(CONFIG['text_embeddings_train']):
            text_train_embeddings = precompute_text_embeddings(
                CONFIG['csv_train'],
                CONFIG['text_embeddings_train'],
                max_samples=CONFIG.get('max_train_samples', None)
            )
        else:
            text_train_embeddings = np.load(CONFIG['text_embeddings_train'])
            print(f"  ✓ Loaded train text embeddings: {text_train_embeddings.shape}")
            # Truncate to match max_train_samples if needed
            if 'max_train_samples' in CONFIG and CONFIG['max_train_samples']:
                text_train_embeddings = text_train_embeddings[:CONFIG['max_train_samples']]
                print(f"  ✓ Truncated to {text_train_embeddings.shape[0]} samples")
        
        if not os.path.exists(CONFIG['text_embeddings_test']):
            text_test_embeddings = precompute_text_embeddings(
                CONFIG['csv_test'],
                CONFIG['text_embeddings_test'],
                max_samples=CONFIG.get('max_test_samples', None)
            )
        else:
            text_test_embeddings = np.load(CONFIG['text_embeddings_test'])
            print(f"  ✓ Loaded test text embeddings: {text_test_embeddings.shape}")
            # Truncate to match max_test_samples if needed
            if 'max_test_samples' in CONFIG and CONFIG['max_test_samples']:
                text_test_embeddings = text_test_embeddings[:CONFIG['max_test_samples']]
                print(f"  ✓ Truncated to {text_test_embeddings.shape[0]} samples")
    
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
    print(f"✓ Predictions saved to: {os.path.join(CONFIG['data_dir'], 'predictions.csv')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
