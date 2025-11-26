"""
Script para treinar modelo YOLO customizado com Stanford Car Dataset
Execute após preparar o dataset com preparar_dataset.py
"""
from ultralytics import YOLO
from pathlib import Path
import torch


def verificar_gpu():
    """Verifica se GPU está disponível"""
    if torch.cuda.is_available():
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠️  GPU não detectada. Treinamento será feito em CPU (mais lento)")
        return False


def treinar_modelo(
    yaml_path='dataset_yolo/dataset.yaml',
    modelo_base='yolov8n.pt',
    epochs=50,
    imgsz=640,
    batch=16,
    nome_projeto='car_detection'
):
    """
    Treina modelo YOLO
    
    Args:
        yaml_path: Caminho para dataset.yaml
        modelo_base: Modelo base YOLO (n/s/m/l/x)
        epochs: Número de épocas de treinamento
        imgsz: Tamanho da imagem
        batch: Tamanho do batch
        nome_projeto: Nome do projeto
    """
    
    print("=" * 70)
    print("🚀 TREINAMENTO DE MODELO YOLO - DETECÇÃO DE CARROS")
    print("=" * 70)
    
    # Verifica GPU
    has_gpu = verificar_gpu()
    
    # Verifica se dataset existe
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"\n❌ ERRO: Arquivo {yaml_path} não encontrado!")
        print("Execute primeiro: python preparar_dataset.py")
        return
    
    print(f"\n📊 Configurações:")
    print(f"  Dataset: {yaml_path}")
    print(f"  Modelo base: {modelo_base}")
    print(f"  Épocas: {epochs}")
    print(f"  Tamanho imagem: {imgsz}x{imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Dispositivo: {'GPU' if has_gpu else 'CPU'}")
    
    # Confirmação
    print("\n⏱️  Tempo estimado:")
    if has_gpu:
        print(f"  ~{epochs * 2} a {epochs * 5} minutos (com GPU)")
    else:
        print(f"  ~{epochs * 10} a {epochs * 30} minutos (sem GPU)")
    
    resposta = input("\nIniciar treinamento? (s/n): ")
    if resposta.lower() != 's':
        print("Treinamento cancelado.")
        return
    
    print("\n🔄 Carregando modelo base...")
    model = YOLO(modelo_base)
    
    print("\n🚀 Iniciando treinamento...\n")
    print("=" * 70)
    
    try:
        # Treina modelo
        results = model.train(
            data=str(yaml_file.absolute()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=nome_projeto,
            name='train',
            patience=10,  # Early stopping
            save=True,
            plots=True,
            verbose=True,
            device=0 if has_gpu else 'cpu'
        )
        
        print("\n" + "=" * 70)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
        
        # Informações do modelo treinado
        model_path = Path(nome_projeto) / 'train' / 'weights' / 'best.pt'
        print(f"\n📦 Modelo salvo em: {model_path}")
        
        print("\n📊 Métricas finais:")
        print(f"  Arquivo de resultados: {nome_projeto}/train/results.csv")
        print(f"  Gráficos: {nome_projeto}/train/")
        
        print("\n🎯 Para usar o modelo treinado:")
        print(f"  1. Em main.py, altere:")
        print(f"     detector = VehicleDetector(model_path='{model_path}')")
        print(f"\n  2. Ou teste diretamente:")
        print(f"     python testar_modelo.py")
        
    except Exception as e:
        print(f"\n❌ ERRO durante treinamento: {e}")
        import traceback
        traceback.print_exc()


def treinar_rapido():
    """Treinamento rápido para testes (poucas épocas)"""
    print("⚡ MODO TREINAMENTO RÁPIDO (para testes)")
    treinar_modelo(
        epochs=10,
        batch=8,
        nome_projeto='car_detection_test'
    )


def treinar_completo():
    """Treinamento completo com mais épocas"""
    print("🎯 MODO TREINAMENTO COMPLETO")
    treinar_modelo(
        epochs=100,
        batch=16,
        nome_projeto='car_detection_full'
    )


def main():
    """Menu principal"""
    print("\n🚗 TREINAMENTO DE MODELO YOLO")
    print("\nEscolha o modo de treinamento:")
    print("1 - Treinamento RÁPIDO (10 épocas - para testes)")
    print("2 - Treinamento COMPLETO (100 épocas - melhor precisão)")
    print("3 - Treinamento CUSTOMIZADO")
    print("0 - Sair")
    
    escolha = input("\nOpção: ")
    
    if escolha == '1':
        treinar_rapido()
    elif escolha == '2':
        treinar_completo()
    elif escolha == '3':
        print("\n⚙️  Configuração customizada:")
        
        modelo = input("Modelo base (yolov8n/s/m/l/x) [yolov8n]: ").strip() or 'yolov8n.pt'
        if not modelo.endswith('.pt'):
            modelo += '.pt'
        
        try:
            epochs = int(input("Número de épocas [50]: ") or "50")
            batch = int(input("Batch size [16]: ") or "16")
            imgsz = int(input("Tamanho da imagem [640]: ") or "640")
        except ValueError:
            print("❌ Valores inválidos. Usando padrões.")
            epochs, batch, imgsz = 50, 16, 640
        
        treinar_modelo(
            modelo_base=modelo,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz
        )
    elif escolha == '0':
        print("Saindo...")
    else:
        print("❌ Opção inválida")


if __name__ == "__main__":
    main()
