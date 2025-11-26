"""
Script rápido para testar modelo treinado com webcam
"""
import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    print("=" * 60)
    print("🧪 TESTE RÁPIDO DO MODELO TREINADO")
    print("=" * 60)
    
    # Caminho do modelo
    model_path = "car_detection_test/train/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"\n❌ Modelo não encontrado: {model_path}")
        print("\nTreine o modelo primeiro:")
        print("  python treinar_modelo.py")
        return
    
    print(f"\n🔄 Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    print("📹 Abrindo webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir webcam")
        print("\nTente testar com uma imagem:")
        print("  python testar_modelo.py")
        return
    
    print("\n✅ Sistema rodando!")
    print("📝 Controles:")
    print("   ESC - Sair")
    print("   ESPAÇO - Pausar/Continuar")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detecta
            results = model(frame, verbose=False)
            
            # Anota
            annotated = results[0].plot()
            
            # Conta detecções
            num_detections = len(results[0].boxes)
            
            # Info no topo
            cv2.putText(
                annotated,
                f"Deteccoes: {num_detections} | ESC=Sair ESPACO=Pausar",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            current_frame = annotated
        
        cv2.imshow('Modelo Customizado - Stanford Cars', current_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 32:  # ESPAÇO
            paused = not paused
            status = "PAUSADO" if paused else "RODANDO"
            print(f"⏸️  {status}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Teste finalizado!")

if __name__ == "__main__":
    main()
