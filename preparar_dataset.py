"""
Script para converter Stanford Car Dataset para formato YOLO
Estrutura esperada do Stanford Dataset:
stanford-car-dataset/
├── train/
│   ├── Acura Integra Type R 2001/
│   ├── Audi A4 2012/
│   └── ...
└── test/
    ├── Acura Integra Type R 2001/
    └── ...
"""
import os
import shutil
from pathlib import Path
import yaml
from PIL import Image

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


def criar_estrutura_yolo(dataset_origem, dataset_destino):
    """
    Cria estrutura de pastas para YOLO
    
    Args:
        dataset_origem: Caminho para o Stanford Car Dataset
        dataset_destino: Caminho onde criar o dataset YOLO
    """
    print("📁 Criando estrutura de diretórios YOLO...")
    
    # Cria estrutura
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            path = Path(dataset_destino) / split / folder
            path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Estrutura criada!")


def encontrar_pasta_train(dataset_origem):
    """
    Encontra a pasta train no dataset, lidando com diferentes estruturas
    
    Args:
        dataset_origem: Caminho raiz do dataset
        
    Returns:
        Caminho para a pasta train ou None
    """
    base_path = Path(dataset_origem)
    
    # Possíveis localizações
    possiveis = [
        base_path / 'train',
        base_path / 'car_data' / 'train',
        base_path / 'car_data' / 'car_data' / 'train',
    ]
    
    for path in possiveis:
        if path.exists() and path.is_dir():
            return path
    
    return None


def processar_imagens(dataset_origem, dataset_destino, split='train', val_split=0.2):
    """
    Processa imagens e cria anotações YOLO
    
    Args:
        dataset_origem: Caminho do Stanford Dataset
        dataset_destino: Caminho de destino YOLO
        split: 'train' ou 'test'
        val_split: Porcentagem para validação (0.2 = 20%)
    """
    # Tenta encontrar a pasta train automaticamente
    if split == 'train':
        origem_path = encontrar_pasta_train(dataset_origem)
        if origem_path is None:
            print(f"❌ Pasta 'train' não encontrada em nenhuma estrutura conhecida")
            print(f"   Caminho base: {dataset_origem}")
            return None
    else:
        origem_path = Path(dataset_origem) / split
        if not origem_path.exists():
            print(f"❌ Pasta não encontrada: {origem_path}")
            return None
    
    print(f"\n🔄 Processando {split}...")
    print(f"📂 Caminho: {origem_path}")
    
    # Lista todas as classes (pastas)
    classes = sorted([d.name for d in origem_path.iterdir() if d.is_dir()])
    
    if not classes:
        print(f"❌ Nenhuma classe (pasta) encontrada em: {origem_path}")
        return None
    
    print(f"📊 Encontradas {len(classes)} classes de carros")
    
    total_images = 0
    train_count = 0
    val_count = 0
    
    # Processa cada classe
    for class_idx, class_name in enumerate(classes):
        class_path = origem_path / class_name
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        
        if not images:
            continue
        
        # Divide em train/val
        num_val = int(len(images) * val_split)
        val_images = images[:num_val]
        train_images = images[num_val:]
        
        # Processa imagens de treino
        for img_path in train_images:
            processar_imagem_unica(
                img_path, 
                dataset_destino, 
                'train', 
                class_idx
            )
            train_count += 1
        
        # Processa imagens de validação
        for img_path in val_images:
            processar_imagem_unica(
                img_path, 
                dataset_destino, 
                'val', 
                class_idx
            )
            val_count += 1
        
        total_images += len(images)
        
        if (class_idx + 1) % 10 == 0:
            print(f"  Processadas {class_idx + 1}/{len(classes)} classes...")
    
    print(f"\n✅ Processamento completo!")
    print(f"  📊 Total: {total_images} imagens")
    print(f"  🎯 Treino: {train_count} imagens")
    print(f"  ✔️  Validação: {val_count} imagens")
    
    return classes


def processar_imagem_unica(img_path, dataset_destino, split, class_id):
    """
    Processa uma imagem individual e cria anotação YOLO
    
    Args:
        img_path: Caminho da imagem
        dataset_destino: Destino YOLO
        split: 'train' ou 'val'
        class_id: ID da classe
    """
    # Copia imagem
    img_dest = Path(dataset_destino) / split / 'images' / img_path.name
    shutil.copy2(img_path, img_dest)
    
    # Cria anotação YOLO (carro ocupa toda a imagem)
    # Formato: <class_id> <x_center> <y_center> <width> <height>
    # Valores normalizados (0.0 - 1.0)
    
    # Como o Stanford Dataset tem fotos de carros centralizados,
    # assumimos que o carro ocupa a maior parte da imagem
    try:
        with Image.open(img_path) as img:
            w, h = img.size
        
        # Caixa que cobre ~80% da imagem centralizada
        x_center = 0.5  # centro horizontal
        y_center = 0.5  # centro vertical
        width = 0.8     # 80% da largura
        height = 0.8    # 80% da altura
        
        # Cria arquivo de anotação
        label_dest = Path(dataset_destino) / split / 'labels' / f"{img_path.stem}.txt"
        with open(label_dest, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    except Exception as e:
        print(f"⚠️  Erro ao processar {img_path.name}: {e}")


def baixar_dataset_kaggle():
    """
    Baixa o Stanford Car Dataset do Kaggle automaticamente
    
    Returns:
        Caminho do dataset baixado ou None se falhar
    """
    if not KAGGLE_AVAILABLE:
        print("\n⚠️  kagglehub não instalado!")
        print("Para baixar automaticamente do Kaggle, instale:")
        print("  pip install kagglehub")
        return None
    
    try:
        print("\n📥 Baixando Stanford Car Dataset do Kaggle...")
        print("⏱️  Isso pode demorar alguns minutos (dataset ~1-2 GB)")
        print("\nNota: Você precisa estar autenticado no Kaggle.")
        print("Configure com: kaggle configure")
        
        path = kagglehub.dataset_download("jutrera/stanford-car-dataset-by-classes-folder")
        
        print(f"\n✅ Dataset baixado com sucesso!")
        print(f"📂 Localização: {path}")
        
        return path
    
    except Exception as e:
        print(f"\n❌ Erro ao baixar dataset: {e}")
        print("\nAlternativas:")
        print("1. Baixe manualmente do Kaggle:")
        print("   https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder")
        print("2. Configure a API do Kaggle: kaggle configure")
        return None


def criar_yaml_config(dataset_destino, classes, nome_arquivo='dataset.yaml'):
    """
    Cria arquivo YAML de configuração para YOLO
    
    Args:
        dataset_destino: Caminho do dataset YOLO
        classes: Lista de nomes das classes
        nome_arquivo: Nome do arquivo YAML
    """
    config = {
        'path': str(Path(dataset_destino).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = Path(dataset_destino) / nome_arquivo
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Arquivo de configuração criado: {yaml_path}")
    print(f"  Classes: {len(classes)}")
    
    return yaml_path


def main():
    """Função principal"""
    print("=" * 60)
    print("🚗 CONVERSOR STANFORD CAR DATASET → YOLO")
    print("=" * 60)
    
    # Onde criar o dataset no formato YOLO
    DATASET_DESTINO = r"C:\Users\Murilo\Documents\Faculdade\Projetos\estacionamento-vision\dataset_yolo"
    
    # Pergunta se quer baixar automaticamente
    print("\n📋 Opções:")
    print("1 - Baixar automaticamente do Kaggle (recomendado)")
    print("2 - Usar dataset já baixado manualmente")
    
    opcao = input("\nEscolha uma opção (1/2): ").strip()
    
    if opcao == '1':
        # Baixa automaticamente
        dataset_path = baixar_dataset_kaggle()
        if dataset_path:
            DATASET_ORIGEM = dataset_path
        else:
            print("\n⚠️  Baixe o dataset manualmente e execute novamente com opção 2.")
            return
    else:
        # Usa caminho manual
        print("\n📂 Digite o caminho do Stanford Car Dataset:")
        print("   Exemplo: C:\\Users\\Murilo\\Downloads\\stanford-car-dataset")
        DATASET_ORIGEM = input("\nCaminho: ").strip().strip('"').strip("'")
        
        if not DATASET_ORIGEM:
            DATASET_ORIGEM = r"C:\caminho\para\stanford-car-dataset"
    
    print(f"\n📂 Origem: {DATASET_ORIGEM}")
    print(f"📂 Destino: {DATASET_DESTINO}")
    
    # Verifica se a origem existe
    if not Path(DATASET_ORIGEM).exists():
        print("\n❌ ERRO: Dataset de origem não encontrado!")
        print(f"Caminho verificado: {DATASET_ORIGEM}")
        print("\nVerifique se:")
        print("1. O caminho está correto")
        print("2. O dataset foi baixado completamente")
        print("3. A pasta contém as subpastas 'train' e/ou 'test'")
        return
    
    # Pergunta confirmação
    print("\n⚠️  ATENÇÃO: Este processo pode demorar alguns minutos.")
    resposta = input("Deseja continuar? (s/n): ")
    if resposta.lower() != 's':
        print("Operação cancelada.")
        return
    
    # 1. Criar estrutura
    criar_estrutura_yolo(DATASET_ORIGEM, DATASET_DESTINO)
    
    # 2. Processar imagens
    classes = processar_imagens(DATASET_ORIGEM, DATASET_DESTINO, 'train', val_split=0.2)
    
    if classes:
        # 3. Criar arquivo YAML
        yaml_path = criar_yaml_config(DATASET_DESTINO, classes)
        
        print("\n" + "=" * 60)
        print("✅ CONVERSÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print(f"\n📋 Próximo passo: Treinar o modelo")
        print(f"Execute: python treinar_modelo.py")
        print(f"\nOu use o arquivo YAML diretamente:")
        print(f"  {yaml_path}")
    else:
        print("\n❌ Nenhuma classe processada. Verifique a estrutura do dataset.")


if __name__ == "__main__":
    main()
