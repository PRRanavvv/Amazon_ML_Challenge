#!/usr/bin/env python3
"""
High-Accuracy Multimodal Price Prediction Pipeline with Ensemble + OCR
========================================================================
Enhanced with: 2-Model Ensemble, Batch Normalization, Learning Rate Decay,
OCR text extraction, better embeddings, data augmentation, text features.
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
    'batch_size_precompute': 8,
    'batch_size_train': 16,
    'embedding_dim': 512,
    'text_embedding_dim': 384,
    
    # Ensemble configuration
    'use_ensemble': True,
    'ensemble_weights': [0.5, 0.5],  # Equal weights for both models
    
    # Model 1: Deep MLP with BatchNorm
    'model1_hidden_dims': [256, 128, 64],
    'model1_dropout': 0.4,
    'model1_use_batch_norm': True,
    'model1_use_residual': True,
    
    # Model 2: Wide MLP with L2 Regularization
    'model2_hidden_dims': [512, 256, 128],
    'model2_dropout': 0.3,
    'model2_use_batch_norm': True,
    'model2_use_residual': False,
    'model2_l2_reg': 1e-4,
    
    'use_text_features': True,
    'use_ocr_features': True,  # NEW: Enable OCR
    'use_augmentation': True,
    
    # OCR Configuration
    'ocr_backend': 'easyocr',  # Options: 'easyocr', 'pytesseract'
    'ocr_languages': ['en'],
    'ocr_confidence_threshold': 0.3,
    
    # Training - IMPROVED WITH BATCH NORM & LR SCHEDULING
    'epochs': 100,
    'learning_rate': 5e-4,
    'batch_norm_momentum': 0.99,
    'batch_norm_epsilon': 1e-3,
    'patience': 15,
    'loss_fn': 'huber',
    'log_transform_price': True,
    'price_clip_percentile': (2, 98),
    
    # Learning Rate Scheduling
    'use_lr_decay': True,
    'decay_steps': 100,
    'decay_rate': 0.96,
    'use_reduce_lr_on_plateau': False,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 5,
    'reduce_lr_min': 1e-6,
}

os.makedirs(CONFIG['embeddings_dir'], exist_ok=True)


# ============================================================================
# OCR SETUP AND EXTRACTION
# ============================================================================

# def setup_ocr():
#     """Setup OCR engine based on configuration."""
#     print(f"\n{'='*70}")
#     print(f"Setting up OCR Engine: {CONFIG['ocr_backend']}")
#     print(f"{'='*70}")
    
#     if CONFIG['ocr_backend'] == 'easyocr':
#         try:
#             import easyocr
#             reader = easyocr.Reader(CONFIG['ocr_languages'], gpu=True)
#             print(f"  ✓ EasyOCR initialized with languages: {CONFIG['ocr_languages']}")
#             return reader, 'easyocr'
#         except ImportError:
#             print("  Installing EasyOCR (this may take a few minutes)...")
#             import subprocess
#             import sys
#             try:
#                 # Try with verbose output to see what's happening
#                 print("  Installing dependencies...")
#                 subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
#                 print("  Installing easyocr...")
#                 subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
#                 print("  ✓ Installation complete. Initializing...")
#                 import easyocr
#                 reader = easyocr.Reader(CONFIG['ocr_languages'], gpu=True)
#                 print(f"  ✓ EasyOCR initialized with languages: {CONFIG['ocr_languages']}")
#                 return reader, 'easyocr'
#             except Exception as install_error:
#                 print(f"  ✗ EasyOCR installation failed: {install_error}")
#                 print(f"  Falling back to Pytesseract...")
#                 CONFIG['ocr_backend'] = 'pytesseract'
#                 return setup_ocr()
#         except Exception as e:
#             print(f"  Error initializing EasyOCR: {e}")
#             print(f"  Falling back to Pytesseract...")
#             CONFIG['ocr_backend'] = 'pytesseract'
#             return setup_ocr()
    
#     elif CONFIG['ocr_backend'] == 'pytesseract':
#         try:
#             import pytesseract
#             from PIL import Image
#             # Test if pytesseract is properly configured
#             pytesseract.get_tesseract_version()
#             print(f"  ✓ Pytesseract initialized")
#             return None, 'pytesseract'
#         except ImportError:
#             print("  Installing pytesseract...")
#             import subprocess
#             import sys
#             try:
#                 subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
#                 import pytesseract
#                 print(f"  ✓ Pytesseract installed")
#                 return None, 'pytesseract'
#             except Exception as e:
#                 print(f"  Error installing pytesseract: {e}")
#                 CONFIG['use_ocr_features'] = False
#                 return None, None
#         except Exception as e:
#             print(f"  Warning: Tesseract binary not found.")
#             print(f"  Install from: https://github.com/tesseract-ocr/tesseract")
#             print(f"  Disabling OCR features and continuing without OCR...")
#             CONFIG['use_ocr_features'] = False
#             return None, None
    
#     return None, None

# def extract_text_from_image(img_path, ocr_reader, ocr_backend):
#     """Extract text from image using OCR."""
#     try:
#         if ocr_backend == 'easyocr':
#             result = ocr_reader.readtext(img_path, detail=1)
#             # Filter by confidence
#             texts = [text for bbox, text, conf in result 
#                     if conf >= CONFIG['ocr_confidence_threshold']]
#             return ' '.join(texts)
        
#         elif ocr_backend == 'pytesseract':
#             import pytesseract
#             img = Image.open(img_path)
#             text = pytesseract.image_to_string(img)
#             return text.strip()
        
#     except Exception as e:
#         # Silently fail for individual images
#         return ""
    
#     return ""


# def precompute_ocr_features(df, image_folder, available_images, output_path, ocr_reader, ocr_backend):
#     """Extract OCR text from all images and create embeddings."""
#     print(f"\n{'='*70}")
#     print(f"Extracting OCR features for {len(df)} samples")
#     print(f"{'='*70}")
    
#     # Build image paths
#     image_paths = {}
#     for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
#         for img_id in available_images:
#             path = os.path.join(image_folder, f"{img_id}{ext}")
#             if os.path.exists(path):
#                 image_paths[img_id] = path
#                 break
    
#     # Extract OCR text
#     ocr_texts = []
#     successful_ocr = 0
    
#     for i, row in df.iterrows():
#         img_id = extract_image_id(row['image_link'])
        
#         if img_id in image_paths:
#             text = extract_text_from_image(image_paths[img_id], ocr_reader, ocr_backend)
#             if text:
#                 successful_ocr += 1
#             ocr_texts.append(text if text else "no text detected")
#         else:
#             ocr_texts.append("no text detected")
        
#         if (i + 1) % 10 == 0:
#             print(f"  Processed {i+1}/{len(df)} images ({successful_ocr} with text)")
    
#     print(f"  ✓ OCR successful on {successful_ocr}/{len(df)} images")
    
#     # Create embeddings from OCR text
#     try:
#         from sentence_transformers import SentenceTransformer
#     except ImportError:
#         print("  Installing sentence-transformers...")
#         os.system("pip install sentence-transformers -q")
#         from sentence_transformers import SentenceTransformer
    
#     print(f"  Creating embeddings from OCR text...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     ocr_embeddings = model.encode(ocr_texts, show_progress_bar=True, batch_size=32)
    
#     np.save(output_path, ocr_embeddings)
#     print(f"  ✓ Saved OCR embeddings: {ocr_embeddings.shape}")
    
#     # Save raw OCR text for inspection
#     ocr_text_path = output_path.replace('.npy', '_raw.txt')
#     with open(ocr_text_path, 'w', encoding='utf-8') as f:
#         for i, text in enumerate(ocr_texts):
#             f.write(f"Image {i}: {text}\n")
#     print(f"  ✓ Saved raw OCR text to: {ocr_text_path}")
    
#     return ocr_embeddings


# ============================================================================
# IMPROVED IMAGE AUGMENTATION
# ============================================================================

def advanced_augmentation(img, strength=0.3):
    """Apply advanced augmentation."""
    if np.random.rand() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(np.random.uniform(1-strength, 1+strength))
    
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(np.random.uniform(1-strength, 1+strength))
    
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(np.random.uniform(1-strength, 1+strength))
    
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
# MULTIMODAL EMBEDDINGS (IMAGE + TEXT + OCR)
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
    
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    image_paths = {}
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        for img_id in available_images:
            path = os.path.join(image_folder, f"{img_id}{ext}")
            if os.path.exists(path):
                image_paths[img_id] = path
                break
    
    all_embeddings = []
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
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
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
# ENSEMBLE MODEL ARCHITECTURES
# ============================================================================

def build_model_1(input_dim):
    """Build Model 1: Deep MLP with Batch Normalization and Residual Connections."""
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(CONFIG['model1_hidden_dims'][0], activation=None)(inputs)
    if CONFIG['model1_use_batch_norm']:
        x = tf.keras.layers.BatchNormalization(
            momentum=CONFIG['batch_norm_momentum'],
            epsilon=CONFIG['batch_norm_epsilon']
        )(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(CONFIG['model1_dropout'])(x)
    
    for hidden_dim in CONFIG['model1_hidden_dims'][1:]:
        residual = x
        
        x = tf.keras.layers.Dense(hidden_dim, activation=None)(x)
        if CONFIG['model1_use_batch_norm']:
            x = tf.keras.layers.BatchNormalization(
                momentum=CONFIG['batch_norm_momentum'],
                epsilon=CONFIG['batch_norm_epsilon']
            )(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(CONFIG['model1_dropout'])(x)
        
        if CONFIG['model1_use_residual'] and residual.shape[-1] == hidden_dim:
            x = tf.keras.layers.Add()([x, residual])
    
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Model_1_Deep_ResNet')
    return model


def build_model_2(input_dim):
    """Build Model 2: Wide MLP with L2 Regularization."""
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(
        CONFIG['model2_hidden_dims'][0], 
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['model2_l2_reg'])
    )(inputs)
    if CONFIG['model2_use_batch_norm']:
        x = tf.keras.layers.BatchNormalization(
            momentum=CONFIG['batch_norm_momentum'],
            epsilon=CONFIG['batch_norm_epsilon']
        )(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(CONFIG['model2_dropout'])(x)
    
    for hidden_dim in CONFIG['model2_hidden_dims'][1:]:
        x = tf.keras.layers.Dense(
            hidden_dim, 
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(CONFIG['model2_l2_reg'])
        )(x)
        if CONFIG['model2_use_batch_norm']:
            x = tf.keras.layers.BatchNormalization(
                momentum=CONFIG['batch_norm_momentum'],
                epsilon=CONFIG['batch_norm_epsilon']
            )(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(CONFIG['model2_dropout'])(x)
    
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Model_2_Wide_L2')
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
# ENSEMBLE TRAINING
# ============================================================================

def train_ensemble(train_embeddings, train_prices, val_split=0.2):
    """Train ensemble of 2 models."""
    print(f"\n{'='*70}")
    print(f"Training 2-Model Ensemble")
    print(f"{'='*70}")
    
    X = train_embeddings
    y = train_prices.reshape(-1, 1)
    
    # Normalize features
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std
    
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
    
    loss_fn = get_loss_fn(CONFIG['loss_fn'])
    models = []
    histories = []
    val_predictions = []
    
    # Train Model 1
    print(f"\n{'='*70}")
    print(f"Training Model 1: Deep ResNet with BatchNorm")
    print(f"  Architecture: {CONFIG['model1_hidden_dims']}")
    print(f"  Dropout: {CONFIG['model1_dropout']}, Residual: {CONFIG['model1_use_residual']}")
    print(f"{'='*70}")
    
    model1 = build_model_1(X.shape[1])
    
    if CONFIG['use_lr_decay']:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=CONFIG['learning_rate'],
            decay_steps=CONFIG['decay_steps'],
            decay_rate=CONFIG['decay_rate'],
            staircase=True
        )
        optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer1 = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    
    model1.compile(optimizer=optimizer1, loss=loss_fn, metrics=['mae', 'mse'])
    
    callbacks1 = [
        tf.keras.callbacks.ModelCheckpoint(
            './best_model_1.h5', monitor='val_loss', save_best_only=True, 
            verbose=1, save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=CONFIG['patience'], 
            restore_best_weights=True, verbose=1
        ),
    ]
    
    history1 = model1.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size_train'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks1,
        verbose=1
    )
    
    val_pred1 = model1.predict(X_val, verbose=0).flatten()
    models.append(model1)
    histories.append(history1)
    val_predictions.append(val_pred1)
    
    # Train Model 2
    print(f"\n{'='*70}")
    print(f"Training Model 2: Wide Network with L2 Regularization")
    print(f"  Architecture: {CONFIG['model2_hidden_dims']}")
    print(f"  Dropout: {CONFIG['model2_dropout']}, L2: {CONFIG['model2_l2_reg']}")
    print(f"{'='*70}")
    
    model2 = build_model_2(X.shape[1])
    
    if CONFIG['use_lr_decay']:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=CONFIG['learning_rate'],
            decay_steps=CONFIG['decay_steps'],
            decay_rate=CONFIG['decay_rate'],
            staircase=True
        )
        optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer2 = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    
    model2.compile(optimizer=optimizer2, loss=loss_fn, metrics=['mae', 'mse'])
    
    callbacks2 = [
        tf.keras.callbacks.ModelCheckpoint(
            './best_model_2.h5', monitor='val_loss', save_best_only=True, 
            verbose=1, save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=CONFIG['patience'], 
            restore_best_weights=True, verbose=1
        ),
    ]
    
    history2 = model2.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size_train'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks2,
        verbose=1
    )
    
    val_pred2 = model2.predict(X_val, verbose=0).flatten()
    models.append(model2)
    histories.append(history2)
    val_predictions.append(val_pred2)
    
    # Ensemble predictions
    print(f"\n{'='*70}")
    print(f"Ensemble Validation Results")
    print(f"{'='*70}")
    
    weights = np.array(CONFIG['ensemble_weights'])
    weights = weights / weights.sum()  # Normalize
    
    ensemble_pred = np.zeros_like(val_pred1)
    for i, pred in enumerate(val_predictions):
        ensemble_pred += weights[i] * pred
        
        if CONFIG['log_transform_price']:
            pred_original = np.expm1(pred)
            true_original = np.expm1(y_val.flatten())
        else:
            pred_original = pred
            true_original = y_val.flatten()
        
        mae = np.mean(np.abs(true_original - pred_original))
        rmse = np.sqrt(np.mean((true_original - pred_original)**2))
        print(f"  Model {i+1} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Weight: {weights[i]:.2f}")
    
    if CONFIG['log_transform_price']:
        ensemble_pred_original = np.expm1(ensemble_pred)
        val_true = np.expm1(y_val.flatten())
    else:
        ensemble_pred_original = ensemble_pred
        val_true = y_val.flatten()
    
    mae = np.mean(np.abs(val_true - ensemble_pred_original))
    rmse = np.sqrt(np.mean((val_true - ensemble_pred_original)**2))
    mape = np.mean(np.abs((val_true - ensemble_pred_original) / (val_true + 1e-8))) * 100
    
    print(f"\n  ENSEMBLE - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    print(f"{'='*70}")
    
    return models, histories


# ============================================================================
# ENSEMBLE INFERENCE
# ============================================================================

def inference_ensemble(models, test_embeddings, df_test, price_min, price_max):
    """Run inference with ensemble."""
    print(f"\n{'='*70}")
    print(f"Running Ensemble Inference")
    print(f"{'='*70}")
    
    mean = np.load('./embeddings/norm_mean.npy')
    std = np.load('./embeddings/norm_std.npy')
    test_embeddings = (test_embeddings - mean) / std
    
    weights = np.array(CONFIG['ensemble_weights'])
    weights = weights / weights.sum()
    
    ensemble_pred = np.zeros(len(test_embeddings))
    
    for i, model in enumerate(models):
        pred = model.predict(test_embeddings, batch_size=256, verbose=0).flatten()
        ensemble_pred += weights[i] * pred
        print(f"  Model {i+1} prediction range: [{pred.min():.2f}, {pred.max():.2f}] (weight: {weights[i]:.2f})")
    
    if CONFIG['log_transform_price']:
        ensemble_pred = np.expm1(ensemble_pred)
    
    ensemble_pred = np.clip(ensemble_pred, price_min, price_max)
    
    df_test['predicted_price'] = ensemble_pred
    
    output_path = './predictions.csv'
    df_test[['sample_id', 'predicted_price']].to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    print(f"  Ensemble range: [{ensemble_pred.min():.2f}, {ensemble_pred.max():.2f}]")
    
    return df_test


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute pipeline."""
    print("\n" + "="*70)
    print("HIGH-ACCURACY MULTIMODAL ENSEMBLE PIPELINE WITH OCR")
    print("2-Model Ensemble + Image + Text + OCR Features")
    print("="*70)
    
    print(f"\n[1/7] Scanning images...")
    available_images, image_folder = get_available_images(CONFIG['data_dir'])
    
    if len(available_images) == 0:
        print("ERROR: No images found!")
        return
    
    print(f"\n[2/7] Filtering CSVs...")
    df_train = filter_csv_by_images(
        CONFIG['csv_train'], available_images, CONFIG['max_train_samples']
    )
    df_test = filter_csv_by_images(
        CONFIG['csv_test'], available_images, CONFIG['max_test_samples']
    )
    
    print(f"\n[3/7] Preprocessing prices...")
    train_prices, price_min, price_max = preprocess_prices(
        df_train, CONFIG['price_clip_percentile'], CONFIG['log_transform_price']
    )
    
    print(f"\n[4/7] Computing embeddings...")
    
    # Image embeddings
    train_img_emb_path = './embeddings/train_img_large.npy'
    test_img_emb_path = './embeddings/test_img_large.npy'
    
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
    
    # Text embeddings from catalog content
    train_text_emb_path = './embeddings/train_text.npy'
    test_text_emb_path = './embeddings/test_text.npy'
    
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
    
    # OCR embeddings
    train_ocr_emb_path = './embeddings/train_ocr.npy'
    test_ocr_emb_path = './embeddings/test_ocr.npy'
    
    if CONFIG['use_ocr_features']:
        ocr_reader, ocr_backend = setup_ocr()
        
        if not os.path.exists(train_ocr_emb_path):
            train_ocr_emb = precompute_ocr_features(
                df_train, image_folder, available_images, train_ocr_emb_path,
                ocr_reader, ocr_backend
            )
        else:
            train_ocr_emb = np.load(train_ocr_emb_path)
            print(f"  ✓ Loaded train OCR embeddings: {train_ocr_emb.shape}")
        
        if not os.path.exists(test_ocr_emb_path):
            test_ocr_emb = precompute_ocr_features(
                df_test, image_folder, available_images, test_ocr_emb_path,
                ocr_reader, ocr_backend
            )
        else:
            test_ocr_emb = np.load(test_ocr_emb_path)
            print(f"  ✓ Loaded test OCR embeddings: {test_ocr_emb.shape}")
    
    # Combine all embeddings
    print(f"\n[5/7] Combining all embeddings...")
    embedding_list_train = [train_img_emb]
    embedding_list_test = [test_img_emb]
    
    if CONFIG['use_text_features']:
        embedding_list_train.append(train_text_emb)
        embedding_list_test.append(test_text_emb)
        print(f"  ✓ Added catalog text embeddings")
    
    if CONFIG['use_ocr_features']:
        embedding_list_train.append(train_ocr_emb)
        embedding_list_test.append(test_ocr_emb)
        print(f"  ✓ Added OCR text embeddings")
    
    train_embeddings = np.hstack(embedding_list_train)
    test_embeddings = np.hstack(embedding_list_test)
    
    print(f"  Final combined embeddings shape:")
    print(f"    Train: {train_embeddings.shape}")
    print(f"    Test: {test_embeddings.shape}")
    
    feature_breakdown = []
    if len(embedding_list_train) > 0:
        feature_breakdown.append(f"Image: {train_img_emb.shape[1]}")
    if CONFIG['use_text_features']:
        feature_breakdown.append(f"Text: {train_text_emb.shape[1]}")
    if CONFIG['use_ocr_features']:
        feature_breakdown.append(f"OCR: {train_ocr_emb.shape[1]}")
    print(f"  Feature breakdown: {', '.join(feature_breakdown)}")
    
    print(f"\n[6/7] Training Ensemble...")
    models, histories = train_ensemble(train_embeddings, train_prices)
    
    print(f"\n[7/7] Ensemble Inference...")
    inference_ensemble(models, test_embeddings, df_test, price_min, price_max)
    
    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"  - predictions.csv")
    print(f"  - best_model_1.h5 (Deep ResNet)")
    print(f"  - best_model_2.h5 (Wide L2)")
    print(f"  - Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"  - Features: Image + Text + OCR")
    print(f"  - Total feature dim: {train_embeddings.shape[1]}")
    print(f"  - Ensemble weights: {CONFIG['ensemble_weights']}")
    print(f"  - OCR Backend: {CONFIG['ocr_backend']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
