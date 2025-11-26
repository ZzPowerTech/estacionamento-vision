"""
Download e preparação do dataset de vagas vazias/ocupadas
Dataset: https://www.kaggle.com/datasets/rizwanrizwannazir/parking
"""
import kagglehub
import shutil
from pathlib import Path


def baixar_dataset_vagas():
    """
    Baixa dataset de vagas do Kaggle
    
    Returns:
        Caminho do dataset baixado
    """
    print("=" * 70)
    print("📥 DOWNLOAD DATASET DE VAGAS - KAGGLE")
    print("=" * 70)
    
    print("\n📥 Baixando dataset...")
    print("   Dataset: rizwanrizwannazir/parking")
    print("   Conteúdo: Fotos de vagas vazias e ocupadas")
    
    try:
        path = kagglehub.dataset_download("rizwanrizwannazir/parking")
        
        print(f"\n✅ Download concluído!")
        print(f"📂 Localização: {path}")
        
        return path
        
    except Exception as e:
        print(f"\n❌ Erro ao baixar: {e}")
        print("\nVerifique:")
        print("1. Você está autenticado no Kaggle")
        print("2. Token configurado corretamente")
        return None


def organizar_dataset(dataset_path, destino='dataset_vagas'):
    """
    Organiza dataset na estrutura esperada pelo treinamento
    
    Args:
        dataset_path: Caminho do dataset baixado
        destino: Pasta de destino
    """
    print(f"\n🔄 Organizando dataset...")
    
    source_path = Path(dataset_path)
    dest_path = Path(destino)
    
    # Remove destino se já existir
    if dest_path.exists():
        print(f"   Removendo pasta anterior: {destino}")
        shutil.rmtree(dest_path)
    
    # Procura pelas pastas empty e occupied
    # O dataset pode estar em diferentes estruturas
    empty_sources = list(source_path.rglob('empty'))
    occupied_sources = list(source_path.rglob('occupied'))
    
    if not empty_sources or not occupied_sources:
        print("\n⚠️  Explorando estrutura do dataset...")
        print(f"   Conteúdo de {dataset_path}:")
        for item in source_path.rglob('*'):
            if item.is_dir():
                num_files = len(list(item.glob('*.jpg'))) + len(list(item.glob('*.png')))
                if num_files > 0:
                    print(f"   📁 {item.relative_to(source_path)} ({num_files} imagens)")
    
    if empty_sources and occupied_sources:
        empty_source = empty_sources[0]
        occupied_source = occupied_sources[0]
        
        print(f"   Encontrado: {empty_source.relative_to(source_path)}")
        print(f"   Encontrado: {occupied_source.relative_to(source_path)}")
        
        # Cria estrutura de destino
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Copia pastas
        print(f"\n📋 Copiando imagens...")
        
        # Empty
        dest_empty = dest_path / 'empty'
        if dest_empty.exists():
            shutil.rmtree(dest_empty)
        shutil.copytree(empty_source, dest_empty)
        num_empty = len(list(dest_empty.glob('*.jpg'))) + len(list(dest_empty.glob('*.png')))
        print(f"   ✅ Vagas vazias: {num_empty} imagens")
        
        # Occupied
        dest_occupied = dest_path / 'occupied'
        if dest_occupied.exists():
            shutil.rmtree(dest_occupied)
        shutil.copytree(occupied_source, dest_occupied)
        num_occupied = len(list(dest_occupied.glob('*.jpg'))) + len(list(dest_occupied.glob('*.png')))
        print(f"   ✅ Vagas ocupadas: {num_occupied} imagens")
        
        print(f"\n✅ Dataset organizado em: {dest_path.absolute()}")
        print(f"   Total: {num_empty + num_occupied} imagens")
        
        return True
    else:
        print("\n❌ Estrutura do dataset não reconhecida")
        print(f"   Pasta empty: {'✅' if empty_sources else '❌'}")
        print(f"   Pasta occupied: {'✅' if occupied_sources else '❌'}")
        return False


def verificar_dataset(destino='dataset_vagas'):
    """
    Verifica se o dataset está pronto para treino
    
    Args:
        destino: Pasta do dataset
        
    Returns:
        True se válido
    """
    dest_path = Path(destino)
    
    empty_path = dest_path / 'empty'
    occupied_path = dest_path / 'occupied'
    
    if not empty_path.exists() or not occupied_path.exists():
        return False
    
    num_empty = len(list(empty_path.glob('*.jpg'))) + len(list(empty_path.glob('*.png')))
    num_occupied = len(list(occupied_path.glob('*.jpg'))) + len(list(occupied_path.glob('*.png')))
    
    return num_empty > 0 and num_occupied > 0


def main():
    """Função principal"""
    print("=" * 70)
    print("🚗 PREPARAÇÃO DATASET DE VAGAS")
    print("=" * 70)
    
    # Verifica se já existe
    if verificar_dataset('dataset_vagas'):
        print("\n📂 Dataset já existe em: dataset_vagas/")
        resposta = input("Deseja baixar novamente? (s/n): ")
        if resposta.lower() != 's':
            print("\n✅ Usando dataset existente")
            print("\n📋 Próximo passo:")
            print("   python treinar_cnn_vagas.py")
            return
    
    # Baixa dataset
    dataset_path = baixar_dataset_vagas()
    
    if not dataset_path:
        print("\n❌ Falha no download")
        return
    
    # Organiza
    sucesso = organizar_dataset(dataset_path, 'dataset_vagas')
    
    if sucesso:
        print("\n" + "=" * 70)
        print("✅ PREPARAÇÃO CONCLUÍDA!")
        print("=" * 70)
        print("\n📋 Estrutura criada:")
        print("   dataset_vagas/")
        print("   ├── empty/      (vagas vazias)")
        print("   └── occupied/   (vagas ocupadas)")
        print("\n🚀 Próximo passo: Treinar o modelo")
        print("   python treinar_cnn_vagas.py")
    else:
        print("\n⚠️  Verifique a estrutura do dataset manualmente")
        print(f"   Localização: {dataset_path}")


if __name__ == "__main__":
    main()
