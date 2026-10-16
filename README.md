
# Hands Control

Controle do mouse do Windows por gestos das mãos, utilizando
uma webcam, MediaPipe, OpenCV e PyAutoGUI.

## Funcionalidades

- Reconhecimento de gestos das mãos em tempo real.
- Movimentação do cursor pelo dedo indicador.
- Clique esquerdo e direito.
- Clique duplo.
- Arrastar e soltar (click and drag).
- Rolagem vertical.
- Controle usando combinacoes de gestos das duas maos.
- Area de controle calibrada para a tela do monitor.

## Tecnologias

- Python
- MediaPipe
- OpenCV
- NumPy
- PyAutoGUI

## Requisitos

- Windows.
- Python 3.14, ou a versao usada para gerar
  o requirements.txt.
- Webcam.
- Arquivo de modelo `gesture_recognizer.task`.

## Instalacao

Clone o repositorio e entre na pasta do projeto:

```powershell
git clone URL_DO_SEU_REPOSITORIO
cd hands-control
```

Crie e ative um ambiente virtual:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependencias:

```powershell
python -m pip install -r requirements.txt
```

## Modelo do MediaPipe

O arquivo `gesture_recognizer.task` precisa estar na mesma
pasta do script Python.

Obtenha o modelo compativel pela documentacao oficial:
https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python

## Execucao

```powershell
python hands_controls2.py
```

Pressione ESC na janela da webcam para encerrar o programa.

## Seguranca

Teste inicialmente em um ambiente seguro. A automacao pode
movimentar o mouse e interagir com janelas do Windows.
