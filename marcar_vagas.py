"""
Ferramenta para marcar vagas de estacionamento na imagem
Baseado em: https://www.kaggle.com/code/rizwanrizwannazir/empty-car-parking-spot-detecting-system
Adaptado para OpenCV com interface simples
"""
import cv2
import json
import numpy as np
from pathlib import Path


class MarcadorVagas:
    """Ferramenta interativa para marcar vagas"""
    
    def __init__(self, image_path):
        """
        Inicializa o marcador
        
        Args:
            image_path: Caminho da imagem do estacionamento
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        self.original = self.image.copy()
        self.vagas = []
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        
        cv2.namedWindow('Marcar Vagas')
        cv2.setMouseCallback('Marcar Vagas', self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos do mouse"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Inicia desenho
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Desenha retângulo temporário
            if self.drawing:
                self.current_rect = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finaliza desenho
            self.drawing = False
            
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Garantir que x1 < x2 e y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Adiciona vaga
                vaga_id = len(self.vagas) + 1
                self.vagas.append({
                    'id': vaga_id,
                    'coords': [x1, y1, x2, y2],
                    'descricao': f'Vaga {vaga_id}'
                })
                
                print(f"✅ Vaga #{vaga_id} adicionada: ({x1}, {y1}) -> ({x2}, {y2})")
                
                self.start_point = None
                self.current_rect = None
    
    def desenhar_vagas(self):
        """Desenha todas as vagas marcadas"""
        self.image = self.original.copy()
        
        # Desenha vagas salvas
        for vaga in self.vagas:
            x1, y1, x2, y2 = vaga['coords']
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label com ID
            label = f"#{vaga['id']}"
            cv2.putText(
                self.image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        # Desenha retângulo temporário
        if self.drawing and self.start_point and self.current_rect:
            x1, y1 = self.start_point
            x2, y2 = self.current_rect
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Info no topo
        info = f"Vagas marcadas: {len(self.vagas)} | ESC=Sair | S=Salvar | R=Reiniciar | Z=Desfazer"
        cv2.putText(
            self.image, info, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            self.image, info, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
    
    def salvar_vagas(self, output_path='vagas.json'):
        """
        Salva vagas em arquivo JSON
        
        Args:
            output_path: Caminho do arquivo de saída
        """
        config = {
            'descricao': 'Configuração das vagas de estacionamento',
            'imagem_origem': self.image_path,
            'total_vagas': len(self.vagas),
            'vagas': self.vagas
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ {len(self.vagas)} vagas salvas em: {output_path}")
    
    def executar(self):
        """Loop principal da ferramenta"""
        print("=" * 70)
        print("📐 MARCADOR DE VAGAS DE ESTACIONAMENTO")
        print("=" * 70)
        print("\n📝 Instruções:")
        print("  1. Clique e arraste para marcar uma vaga")
        print("  2. Pressione S para salvar")
        print("  3. Pressione Z para desfazer última vaga")
        print("  4. Pressione R para reiniciar")
        print("  5. Pressione ESC para sair")
        print("\n🚀 Comece a marcar as vagas...")
        
        while True:
            self.desenhar_vagas()
            cv2.imshow('Marcar Vagas', self.image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            elif key == ord('s') or key == ord('S'):
                # Salvar
                if len(self.vagas) > 0:
                    self.salvar_vagas()
                    print("💾 Pressione ESC para sair ou continue marcando")
                else:
                    print("⚠️  Nenhuma vaga marcada!")
            
            elif key == ord('z') or key == ord('Z'):
                # Desfazer
                if len(self.vagas) > 0:
                    removida = self.vagas.pop()
                    print(f"↩️  Vaga #{removida['id']} removida")
                else:
                    print("⚠️  Nenhuma vaga para desfazer")
            
            elif key == ord('r') or key == ord('R'):
                # Reiniciar
                self.vagas = []
                print("🔄 Vagas reiniciadas")
        
        cv2.destroyAllWindows()
        
        # Perguntar se quer salvar ao sair
        if len(self.vagas) > 0:
            print(f"\n📊 Você marcou {len(self.vagas)} vagas")
            resposta = input("Deseja salvar? (s/n): ")
            if resposta.lower() == 's':
                self.salvar_vagas()


def main():
    """Função principal"""
    import sys
    
    print("=" * 70)
    print("📐 FERRAMENTA DE MARCAÇÃO DE VAGAS")
    print("=" * 70)
    
    # Solicita caminho da imagem
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\n📷 Digite o caminho da imagem do estacionamento:")
        print("   Exemplo: estacionamento.jpg")
        image_path = input("\nCaminho: ").strip().strip('"').strip("'")
    
    if not Path(image_path).exists():
        print(f"\n❌ Imagem não encontrada: {image_path}")
        print("\nUso:")
        print("  python marcar_vagas.py imagem.jpg")
        print("  ou execute sem argumentos para digitar o caminho")
        return
    
    try:
        marcador = MarcadorVagas(image_path)
        marcador.executar()
        
        print("\n✅ Ferramenta encerrada")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")


if __name__ == "__main__":
    main()
