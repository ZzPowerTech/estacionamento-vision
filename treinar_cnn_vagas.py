"""
Treinamento de CNN para Classificação de Vagas de Estacionamento
Baseado em: https://www.kaggle.com/code/rizwanrizwannazir/empty-car-parking-spot-detecting-system
Melhorado com: data augmentation, callbacks, e otimizações
"""
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)


def criar_modelo_cnn(input_shape=(48, 48, 1)):
    """
    Cria modelo CNN simplificado para melhor generalização
    
    Args:
        input_shape: Forma da imagem de entrada
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        # Primeira camada convolucional
        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Segunda camada convolucional
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Terceira camada convolucional
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Camadas densas
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        
        # Camada de saída (2 classes: vazia/ocupada)
        Dense(2, activation='softmax')
    ])
    
    # Compilar com otimizador Adam
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )
    
    return model


def carregar_imagens(pastas_treino):
    """
    Carrega imagens de vagas vazias e ocupadas
    
    Args:
        pastas_treino: Lista com [pasta_vazias, pasta_ocupadas]
        
    Returns:
        images: Array numpy com imagens
        labels: Array numpy com labels (0=vazia, 1=ocupada)
    """
    images = []
    labels = []
    
    print("📂 Carregando imagens...")
    
    for i, folder in enumerate(pastas_treino):
        if not Path(folder).exists():
            print(f"⚠️  Pasta não encontrada: {folder}")
            continue
            
        label = i  # 0 para vazia, 1 para ocupada
        arquivos = list(Path(folder).glob('*.jpg')) + list(Path(folder).glob('*.png'))
        
        print(f"  {'Vazias' if i == 0 else 'Ocupadas'}: {len(arquivos)} imagens")
        
        for img_path in arquivos:
            try:
                # Ler em escala de cinza
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Redimensionar para 48x48
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label)
                
            except Exception as e:
                print(f"⚠️  Erro ao carregar {img_path.name}: {e}")
    
    print(f"✅ Total de imagens carregadas: {len(images)}")
    return np.array(images), np.array(labels)


def preprocessar_dados(images, labels, test_size=0.2):
    """
    Preprocessa e divide os dados
    
    Args:
        images: Array de imagens
        labels: Array de labels
        test_size: Proporção para teste
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n🔄 Preprocessando dados...")
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Reshape e normalizar
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32') / 255
    
    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    
    print(f"  Treino: {X_train.shape[0]} imagens")
    print(f"  Teste: {X_test.shape[0]} imagens")
    
    return X_train, X_test, y_train, y_test


def criar_data_augmentation():
    """
    Cria gerador de data augmentation moderado
    
    Returns:
        ImageDataGenerator configurado
    """
    return ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def treinar_modelo(pastas_treino, epochs=50, batch_size=64):
    """
    Treina o modelo CNN completo
    
    Args:
        pastas_treino: Lista com pastas [vazias, ocupadas]
        epochs: Número de épocas
        batch_size: Tamanho do batch
        
    Returns:
        model: Modelo treinado
        history: Histórico do treinamento
    """
    print("=" * 70)
    print("🚗 TREINAMENTO CNN - DETECÇÃO DE VAGAS VAZIAS/OCUPADAS")
    print("=" * 70)
    
    # Carregar e preprocessar dados
    images, labels = carregar_imagens(pastas_treino)
    
    if len(images) == 0:
        print("\n❌ Nenhuma imagem carregada! Verifique os caminhos.")
        return None, None
    
    X_train, X_test, y_train, y_test = preprocessar_dados(images, labels)
    
    # Criar modelo
    print("\n🏗️  Criando modelo CNN...")
    model = criar_modelo_cnn()
    model.summary()
    
    # Callbacks
    callbacks = [
        # Salvar melhor modelo
        ModelCheckpoint(
            'modelos/vaga_detector_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduzir learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Data augmentation
    print("\n📊 Configurando data augmentation...")
    datagen = criar_data_augmentation()
    datagen.fit(X_train)
    
    # Treinar
    print(f"\n🚀 Iniciando treinamento ({epochs} épocas)...")
    print("=" * 70)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Avaliar
    print("\n" + "=" * 70)
    print("📊 AVALIAÇÃO FINAL")
    print("=" * 70)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Salvar modelo final
    model.save('modelos/vaga_detector_final.h5')
    print("\n✅ Modelo salvo em:")
    print("  - modelos/vaga_detector_best.h5 (melhor)")
    print("  - modelos/vaga_detector_final.h5 (final)")
    
    return model, history


def plotar_historico(history):
    """
    Plota gráficos do histórico de treinamento
    
    Args:
        history: Histórico retornado pelo fit()
    """
    import matplotlib.pyplot as plt
    
    # Acurácia
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('modelos/training_history.png')
    print("\n📈 Gráficos salvos em: modelos/training_history.png")
    plt.show()


def main():
    """Função principal"""
    # CONFIGURE OS CAMINHOS AQUI
    pastas_treino = [
        "dataset_vagas/empty",      # Vagas vazias
        "dataset_vagas/occupied"    # Vagas ocupadas
    ]
    
    # Criar pasta de modelos
    Path("modelos").mkdir(exist_ok=True)
    
    # Treinar
    model, history = treinar_modelo(
        pastas_treino=pastas_treino,
        epochs=50,
        batch_size=64
    )
    
    if model and history:
        # Plotar resultados
        plotar_historico(history)
        
        print("\n" + "=" * 70)
        print("✅ TREINAMENTO CONCLUÍDO!")
        print("=" * 70)
        print("\n📋 Próximos passos:")
        print("  1. Use o modelo: modelos/vaga_detector_best.h5")
        print("  2. Execute: python sistema_vagas_cnn.py")


if __name__ == "__main__":
    main()
