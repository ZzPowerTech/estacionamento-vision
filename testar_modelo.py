"""
Script para testar modelo YOLO treinado
Testa com imagens ou webcam
"""
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse


def testar_com_imagem(model_path, image_path):
    """
    Testa modelo com uma imagem
    
    Args:
        model_path: Caminho do modelo .pt
        image_path: Caminho da imagem
    """
    print(f"🔄 Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    print(f"📷 Processando imagem: {image_path}")
    results = model(image_path)
    
    # Mostra resultado
    annotated = results[0].plot()
    
    cv2.imshow('Detecção - Pressione qualquer tecla para sair', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Exibe informações
    print("\n📊 Detecções:")
    for i, box in enumerate(results[0].boxes):
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results[0].names[cls]
        print(f"  {i+1}. {class_name} - Confiança: {conf:.2%}")


def testar_com_webcam(model_path, camera_id=0):
    """
    Testa modelo com webcam em tempo real
    
    Args:
        model_path: Caminho do modelo .pt
        camera_id: ID da câmera
    """
    print(f"🔄 Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    print(f"📹 Abrindo câmera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir câmera")
        return
    
    print("✅ Câmera aberta! Pressione ESC para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detecta
        results = model(frame, verbose=False)
        
        # Anota frame
        annotated = results[0].plot()
        
        # Mostra FPS
        cv2.putText(
            annotated,
            "Pressione ESC para sair",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow('Teste do Modelo - ESC para sair', annotated)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Teste finalizado")


def testar_com_video(model_path, video_path):
    """
    Testa modelo com arquivo de vídeo
    
    Args:
        model_path: Caminho do modelo .pt
        video_path: Caminho do vídeo
    """
    print(f"🔄 Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    print(f"🎬 Abrindo vídeo: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir vídeo")
        return
    
    print("✅ Vídeo aberto! Pressione ESC para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⏹️  Fim do vídeo")
            break
        
        # Detecta
        results = model(frame, verbose=False)
        
        # Anota frame
        annotated = results[0].plot()
        
        cv2.imshow('Teste do Modelo - ESC para sair', annotated)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Teste finalizado")


def encontrar_modelo_treinado():
    """Encontra o modelo treinado mais recente"""
    # Procura em car_detection/train/weights/best.pt
    possiveis = [
        Path('car_detection/train/weights/best.pt'),
        Path('car_detection_full/train/weights/best.pt'),
        Path('car_detection_test/train/weights/best.pt'),
    ]
    
    for path in possiveis:
        if path.exists():
            return str(path)
    
    return None


def main():
    """Menu principal"""
    print("=" * 60)
    print("🧪 TESTE DE MODELO YOLO TREINADO")
    print("=" * 60)
    
    # Procura modelo automaticamente
    model_path = encontrar_modelo_treinado()
    
    if model_path:
        print(f"\n✅ Modelo encontrado: {model_path}")
        usar = input("Usar este modelo? (s/n): ")
        if usar.lower() != 's':
            model_path = input("Digite o caminho do modelo .pt: ")
    else:
        print("\n⚠️  Modelo treinado não encontrado automaticamente")
        model_path = input("Digite o caminho do modelo .pt: ")
    
    if not Path(model_path).exists():
        print("❌ Modelo não encontrado!")
        return
    
    # Menu de opções
    print("\n📋 Escolha o tipo de teste:")
    print("1 - Testar com WEBCAM")
    print("2 - Testar com IMAGEM")
    print("3 - Testar com VÍDEO")
    print("0 - Sair")
    
    escolha = input("\nOpção: ")
    
    if escolha == '1':
        camera = input("ID da câmera [0]: ").strip() or "0"
        testar_com_webcam(model_path, int(camera))
    
    elif escolha == '2':
        img_path = input("Caminho da imagem: ")
        if Path(img_path).exists():
            testar_com_imagem(model_path, img_path)
        else:
            print("❌ Imagem não encontrada!")
    
    elif escolha == '3':
        video_path = input("Caminho do vídeo: ")
        if Path(video_path).exists():
            testar_com_video(model_path, video_path)
        else:
            print("❌ Vídeo não encontrado!")
    
    elif escolha == '0':
        print("Saindo...")
    
    else:
        print("❌ Opção inválida")


if __name__ == "__main__":
    main()
