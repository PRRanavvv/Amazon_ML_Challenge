#!/usr/bin/env python3
"""
High-Accuracy Multimodal Price Prediction Pipeline
===================================================
Enhanced with: Batch Normalization, Learning Rate Decay, ReduceLROnPlateau,
better embeddings, data augmentation, ensemble methods, text features.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageEnhance
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Paths
    'data_dir': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset',
    'csv_train': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\train.csv',
    'csv_test': r'C:\Users\rawat\Downloads\ml challenge\student_resource\dataset\test.csv',
    'embeddings_dir': './embeddings',
    
    # Sample limits
    'max_train_samples': 75,
    'max_test_samples': 10,
    
    # Model config - IMPROVED
    'image_size': 224,
    'batch_size_precompute': 8,  # Smaller for stability
    'batch_size_train': 16,
    'embedding_dim': 512,
    'text_embedding_dim': 384,  # For text features
    'mlp_hidden_dims': [256, 128, 64],  # Deeper network
    'mlp_dropout': 0.4,  # Higher dropout
    'use_text_features': True,  # Enable text embeddings
    'use_augmentation': True,  # Data augmentation
    
    # Training - IMPROVED WITH BATCH NORM & LR SCHEDULING
    'epochs': 100,  # More epochs with early stopping
    'learning_rate': 5e-4,  # Initial learning rate
    'batch_norm_momentum': 0.99,  # Batch norm momentum
    'batch_norm_epsilon': 1e-3,  # Batch norm epsilon
    'patience': 15,  # Early stopping patience
    'loss_fn': 'huber',
    'log_transform_price': True,
    'price_clip_percentile': (2, 98),
    'use_batch_norm': True,  # ENABLE BATCH NORMALIZATION
    'use_residual': True,
    
    # Learning Rate Scheduling (choose one or the other, not both)
    'use_lr_decay': True,  # Exponential decay - reduces LR automatically every N steps
    'decay_steps': 100,
    'decay_rate': 0.96,
    'use_reduce_lr_on_plateau': False,  # ReduceLROnPlateau - reduces LR when val_loss plateaus
    # NOTE: Cannot use both at the same time. Set one to True, the other to False
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 5,
    'reduce_lr_min': 1e-6,
}

os.makedirs(CONFIG['embeddings_dir'], exist_ok=True)


# ============================================================================
# IMPROVED IMAGE AUGMENTATION
# ============================================================================

def advanced_augmentation(img, strength=0.3):
    """Apply advanced augmentation."""
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random brightness
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(np.random.uniform(1-strength, 1+strength))
    
    # Random contrast
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(np.random.uniform(1-strength, 1+strength))
    
    # Random color saturation
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(np.random.uniform(1-strength, 1+strength))
    
    # Random rotation
    if np.random.rand() > 0.7:
        angle = np.random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=(255, 255, 255))
    
    return img


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_images(data_dir):
    """Build mapping of available images."""
    image_folder = os.path.join(data_dir, 'images') if os.path.exists(
        os.path.join(data_dir, 'images')) else data_dir
    
    available_images = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for img_path in Path(image_folder).glob(ext):
            available_images.add(img_path.stem)
    
    print(f"  Found {len(available_images)} images in {image_folder}")
    return available_images, image_folder


def filter_csv_by_images(csv_path, available_images, max_samples=None):
    """Filter CSV to only samples with available images."""
    print(f"  Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")
    
    def has_image(row):
        image_url = str(row['image_link'])
        if '/' in image_url:
            img_id = image_url.split('/')[-1].rsplit('.', 1)[0]
        else:
            img_id = image_url
        return img_id in available_images
    
    df_filtered = df[df.apply(has_image, axis=1)].reset_index(drop=True)
    print(f"  Rows with images: {len(df_filtered)}")
    
    if max_samples:
        df_filtered = df_filtered.head(max_samples)
        print(f"  Limited to: {max_samples} samples")
    
    return df_filtered


def extract_image_id(image_url):
    """Extract image ID from URL."""
    if '/' in image_url:
        img_filename = image_url.split('/')[-1]
        return img_filename.rsplit('.', 1)[0]
    return image_url


# ============================================================================
# MULTIMODAL EMBEDDINGS (IMAGE + TEXT)
# ============================================================================

def precompute_image_embeddings(df, image_folder, available_images, output_path, 
                                batch_size=8, augment=False):
    """Precompute image embeddings with better model."""
    print(f"\n{'='*70}")
    print(f"Computing image embeddings for {len(df)} samples")
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
    
    # Use better CLIP model
    model_name = "openai/clip-vit-large-patch14"  # Larger, more accurate
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    # Build image path mapping
    image_paths = {}
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        for img_id in available_images:
            path = os.path.join(image_folder, f"{img_id}{ext}")
            if os.path.exists(path):
                image_paths[img_id] = path
                break
    
    all_embeddings = []
    
    # Multiple augmented versions for training
    num_augments = 3 if augment else 1
    
    for aug_iter in range(num_augments):
        embeddings = []
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            images = []
            
            for _, row in batch_df.iterrows():
                img_id = extract_image_id(row['image_link'])
                
                if img_id in image_paths:
                    try:
                        img = Image.open(image_paths[img_id]).convert('RGB')
                        
                        # Apply augmentation
                        if augment and aug_iter > 0:
                            img = advanced_augmentation(img)
                        
                        img = img.resize((CONFIG['image_size'], CONFIG['image_size']))
                        images.append(img)
                    except Exception as e:
                        print(f"  Error loading {img_id}: {e}")
                        images.append(Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size'])))
                else:
                    images.append(Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size'])))
            
            with torch.no_grad():
                inputs = processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                img_embeds = model.get_image_features(**inputs).cpu().numpy()
            
            embeddings.append(img_embeds)
        
        all_embeddings.append(np.vstack(embeddings))
        if aug_iter == 0:
            print(f"  Original embeddings: {all_embeddings[0].shape}")
        else:
            print(f"  Augmentation {aug_iter}: done")
    
    # Average embeddings from augmented versions
    if num_augments > 1:
        final_embeddings = np.mean(all_embeddings, axis=0)
        print(f"  Averaged {num_augments} augmented versions")
    else:
        final_embeddings = all_embeddings[0]
    
    np.save(output_path, final_embeddings)
    print(f"✓ Saved embeddings: {final_embeddings.shape}")
    
    return final_embeddings


def precompute_text_embeddings(df, output_path):
    """Extract text features from catalog_content."""
    print(f"\n{'='*70}")
    print(f"Computing text embeddings for {len(df)} samples")
    print(f"{'='*70}")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        os.system("pip install sentence-transformers -q")
        from sentence_transformers import SentenceTransformer
    
    # Better text model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract text from catalog_content
    texts = []
    for _, row in df.iterrows():
        text = str(row.get('catalog_content', ''))
        if not text or text == 'nan':
            text = "unknown product"
        texts.append(text)
    
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    np.save(output_path, embeddings)
    print(f"✓ Saved text embeddings: {embeddings.shape}")
    
    return embeddings


# ============================================================================
# IMPROVED MLP MODEL WITH BATCH NORMALIZATION
# ============================================================================

def build_advanced_mlp(input_dim):
    """Build MLP with Batch Normalization and residual connections."""
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # Initial projection
    x = tf.keras.layers.Dense(CONFIG['mlp_hidden_dims'][0], activation=None)(inputs)
    if CONFIG['use_batch_norm']:
        x = tf.keras.layers.BatchNormalization(
            momentum=CONFIG['batch_norm_momentum'],
            epsilon=CONFIG['batch_norm_epsilon']
        )(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(CONFIG['mlp_dropout'])(x)
    
    # Residual blocks
    for hidden_dim in CONFIG['mlp_hidden_dims'][1:]:
        residual = x
        
        # Block
        x = tf.keras.layers.Dense(hidden_dim, activation=None)(x)
        if CONFIG['use_batch_norm']:
            x = tf.keras.layers.BatchNormalization(
                momentum=CONFIG['batch_norm_momentum'],
                epsilon=CONFIG['batch_norm_epsilon']
            )(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(CONFIG['mlp_dropout'])(x)
        
        # Residual connection if dimensions match
        if CONFIG['use_residual'] and residual.shape[-1] == hidden_dim:
            x = tf.keras.layers.Add()([x, residual])
    
    # Output
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_loss_fn(loss_type='huber'):
    """Get loss function."""
    if loss_type == 'huber':
        return tf.keras.losses.Huber(delta=1.0)
    elif loss_type == 'mae':
        return tf.keras.losses.MeanAbsoluteError()
    else:
        return tf.keras.losses.MeanSquaredError()


# ============================================================================
# PRICE PREPROCESSING
# ============================================================================

def preprocess_prices(df, clip_percentile=(2, 98), log_transform=True):
    """Clean and preprocess prices."""
    prices = df['price'].values.astype(float)
    valid_mask = ~np.isnan(prices)
    prices_valid = prices[valid_mask]
    
    if clip_percentile:
        low, high = np.percentile(prices_valid, clip_percentile)
        prices = np.clip(prices, low, high)
        print(f"  Clipped to [{low:.2f}, {high:.2f}]")
    
    if log_transform:
        prices = np.log1p(prices)
        print(f"  Applied log1p transform")
    
    return prices, prices_valid.min(), prices_valid.max()


# ============================================================================
# TRAINING WITH IMPROVEMENTS - BATCH NORM & LR SCHEDULING
# ============================================================================

def train_mlp(train_embeddings, train_prices, val_split=0.2):
    """Train MLP with Batch Normalization and Learning Rate Scheduling."""
    print(f"\n{'='*70}")
    print(f"Training Advanced MLP with BatchNorm & LR Scheduling")
    print(f"{'='*70}")
    
    X = train_embeddings
    y = train_prices.reshape(-1, 1)
    
    # Normalize features
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std
    
    # Save normalization params
    np.save('./embeddings/norm_mean.npy', mean)
    np.save('./embeddings/norm_std.npy', std)
    
    # Train/val split
    n_train = int(len(X) * (1 - val_split))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Batch Normalization: {'ENABLED' if CONFIG['use_batch_norm'] else 'DISABLED'}")
    print(f"  LR Decay: {'ENABLED' if CONFIG['use_lr_decay'] else 'DISABLED'}")
    print(f"  ReduceLROnPlateau: {'ENABLED' if CONFIG['use_reduce_lr_on_plateau'] else 'DISABLED'}")
    
    model = build_advanced_mlp(X.shape[1])
    loss_fn = get_loss_fn(CONFIG['loss_fn'])
    
    # Learning Rate Schedule with Exponential Decay
    if CONFIG['use_lr_decay']:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=CONFIG['learning_rate'],
            decay_steps=CONFIG['decay_steps'],
            decay_rate=CONFIG['decay_rate'],
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print(f"  ✓ Exponential Decay: rate={CONFIG['decay_rate']}, steps={CONFIG['decay_steps']}")
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae', 'mse'])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            './best_model.h5', monitor='val_loss', save_best_only=True, 
            verbose=1, save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=CONFIG['patience'], 
            restore_best_weights=True, verbose=1
        ),
    ]
    
    # ReduceLROnPlateau callback (only if NOT using exponential decay)
    if CONFIG['use_reduce_lr_on_plateau'] and not CONFIG['use_lr_decay']:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=CONFIG['reduce_lr_factor'],
                patience=CONFIG['reduce_lr_patience'],
                min_lr=CONFIG['reduce_lr_min'],
                verbose=1
            )
        )
        print(f"  ✓ ReduceLROnPlateau: factor={CONFIG['reduce_lr_factor']}, "
              f"patience={CONFIG['reduce_lr_patience']}, min_lr={CONFIG['reduce_lr_min']}")
    elif CONFIG['use_lr_decay']:
        print(f"  ℹ Using Exponential Decay (incompatible with ReduceLROnPlateau)")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size_train'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluation
    val_pred = model.predict(X_val, verbose=0).flatten()
    if CONFIG['log_transform_price']:
        val_true = np.expm1(y_val.flatten())
        val_pred = np.expm1(val_pred)
    else:
        val_true = y_val.flatten()
    
    mae = np.mean(np.abs(val_true - val_pred))
    rmse = np.sqrt(np.mean((val_true - val_pred)**2))
    mape = np.mean(np.abs((val_true - val_pred) / (val_true + 1e-8))) * 100
    
    print(f"\n{'='*70}")
    print(f"Validation Results:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"{'='*70}")
    
    return model, history


# ============================================================================
# INFERENCE
# ============================================================================

def inference_on_test(model, test_embeddings, df_test, price_min, price_max):
    """Run inference."""
    print(f"\n{'='*70}")
    print(f"Running Inference")
    print(f"{'='*70}")
    
    # Normalize using training statistics
    mean = np.load('./embeddings/norm_mean.npy')
    std = np.load('./embeddings/norm_std.npy')
    test_embeddings = (test_embeddings - mean) / std
    
    predictions = model.predict(test_embeddings, batch_size=256, verbose=1).flatten()
    
    if CONFIG['log_transform_price']:
        predictions = np.expm1(predictions)
    
    predictions = np.clip(predictions, price_min, price_max)
    
    df_test['predicted_price'] = predictions
    
    output_path = './predictions.csv'
    df_test[['sample_id', 'predicted_price']].to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    print(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return df_test


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute pipeline."""
    print("\n" + "="*70)
    print("HIGH-ACCURACY MULTIMODAL PIPELINE")
    print("With Batch Normalization & Learning Rate Scheduling")
    print("="*70)
    
    # Step 1: Get available images
    print(f"\n[1/6] Scanning images...")
    available_images, image_folder = get_available_images(CONFIG['data_dir'])
    
    if len(available_images) == 0:
        print("ERROR: No images found!")
        return
    
    # Step 2: Filter CSVs
    print(f"\n[2/6] Filtering CSVs...")
    df_train = filter_csv_by_images(
        CONFIG['csv_train'], available_images, CONFIG['max_train_samples']
    )
    df_test = filter_csv_by_images(
        CONFIG['csv_test'], available_images, CONFIG['max_test_samples']
    )
    
    # Step 3: Preprocess prices
    print(f"\n[3/6] Preprocessing prices...")
    train_prices, price_min, price_max = preprocess_prices(
        df_train, CONFIG['price_clip_percentile'], CONFIG['log_transform_price']
    )
    
    # Step 4: Compute embeddings
    print(f"\n[4/6] Computing embeddings...")
    
    train_img_emb_path = './embeddings/train_img_large.npy'
    test_img_emb_path = './embeddings/test_img_large.npy'
    train_text_emb_path = './embeddings/train_text.npy'
    test_text_emb_path = './embeddings/test_text.npy'
    
    # Image embeddings
    if not os.path.exists(train_img_emb_path):
        train_img_emb = precompute_image_embeddings(
            df_train, image_folder, available_images, train_img_emb_path,
            CONFIG['batch_size_precompute'], augment=CONFIG['use_augmentation']
        )
    else:
        train_img_emb = np.load(train_img_emb_path)
        print(f"  ✓ Loaded train image embeddings: {train_img_emb.shape}")
    
    if not os.path.exists(test_img_emb_path):
        test_img_emb = precompute_image_embeddings(
            df_test, image_folder, available_images, test_img_emb_path,
            CONFIG['batch_size_precompute'], augment=False
        )
    else:
        test_img_emb = np.load(test_img_emb_path)
        print(f"  ✓ Loaded test image embeddings: {test_img_emb.shape}")
    
    # Text embeddings
    if CONFIG['use_text_features']:
        if not os.path.exists(train_text_emb_path):
            train_text_emb = precompute_text_embeddings(df_train, train_text_emb_path)
        else:
            train_text_emb = np.load(train_text_emb_path)
            print(f"  ✓ Loaded train text embeddings: {train_text_emb.shape}")
        
        if not os.path.exists(test_text_emb_path):
            test_text_emb = precompute_text_embeddings(df_test, test_text_emb_path)
        else:
            test_text_emb = np.load(test_text_emb_path)
            print(f"  ✓ Loaded test text embeddings: {test_text_emb.shape}")
        
        # Combine embeddings
        train_embeddings = np.hstack([train_img_emb, train_text_emb])
        test_embeddings = np.hstack([test_img_emb, test_text_emb])
        print(f"  Combined embeddings: {train_embeddings.shape}")
    else:
        train_embeddings = train_img_emb
        test_embeddings = test_img_emb
    
    # Step 5: Train
    print(f"\n[5/6] Training...")
    model, history = train_mlp(train_embeddings, train_prices)
    
    # Step 6: Inference
    print(f"\n[6/6] Inference...")
    inference_on_test(model, test_embeddings, df_test, price_min, price_max)
    
    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"  - predictions.csv")
    print(f"  - best_model.h5")
    print(f"  - Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
