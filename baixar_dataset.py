"""
Script simplificado para baixar Stanford Car Dataset do Kaggle
"""
import kagglehub

def main():
    print("=" * 60)
    print("📥 DOWNLOAD STANFORD CAR DATASET")
    print("=" * 60)
    
    print("\n⚠️  IMPORTANTE:")
    print("1. Você precisa ter uma conta no Kaggle")
    print("2. Configure a API do Kaggle primeiro:")
    print("   - Vá em: https://www.kaggle.com/settings")
    print("   - Clique em 'Create New Token'")
    print("   - Salve o arquivo kaggle.json em:")
    print("     Windows: C:\\Users\\<seu_usuario>\\.kaggle\\kaggle.json")
    print("     Linux/Mac: ~/.kaggle/kaggle.json")
    
    input("\nPressione ENTER para continuar...")
    
    print("\n📥 Baixando Stanford Car Dataset do Kaggle...")
    print("⏱️  Tamanho: ~1-2 GB - pode demorar alguns minutos\n")
    
    try:
        # Download do dataset
        path = kagglehub.dataset_download("jutrera/stanford-car-dataset-by-classes-folder")
        
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD CONCLUÍDO!")
        print("=" * 60)
        print(f"\n📂 Dataset salvo em: {path}")
        print(f"\n📋 Próximo passo:")
        print(f"   Execute: python preparar_dataset.py")
        print(f"   E escolha a opção 2 (usar dataset já baixado)")
        print(f"   Cole o caminho: {path}")
        
        # Salva o caminho em arquivo para referência
        with open('dataset_path.txt', 'w') as f:
            f.write(path)
        print(f"\n💾 Caminho salvo em: dataset_path.txt")
        
    except Exception as e:
        print(f"\n❌ ERRO ao baixar: {e}")
        print("\nPossíveis soluções:")
        print("1. Verifique se configurou a API do Kaggle corretamente")
        print("2. Verifique sua conexão com a internet")
        print("3. Baixe manualmente:")
        print("   https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder")

if __name__ == "__main__":
    main()
