import math

from imutils.perspective import four_point_transform
from skimage.metrics import structural_similarity as compare_ssim
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from pyzbar.pyzbar import decode
import time
import matplotlib.pyplot as plt

import ccl


def eliminaSombras(image):
    planos_rgb = cv2.split(image)
    resultado_planos_normalizados = []
    for plane in planos_rgb:
        imagem_dilatada = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(imagem_dilatada, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)

        imagem_normalizada = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        resultado_planos_normalizados.append(imagem_normalizada)

    return cv2.merge(resultado_planos_normalizados)


def detectaQRcode(image):
        for code in decode(image):
            qrCode = code
            dado = code.data.decode('utf-8')
            pts = np.array([qrCode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            pontoInferior = pts[1][0]

            return dado, image[pontoInferior[1] + 20:image.shape[0], 0:image.shape[1]]
        return None, image



def preProcessamento(image):
    cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converte em cinza
    turva = cv2.GaussianBlur(cinza, (5, 5), 0)  # desfoca para para reduzir o ruido de alta frequencia
    borda = cv2.Canny(turva, 75, 200)  # Detecta bordas
    thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cinza, turva, borda, thresh

def processamentoImagem(image):
    (cinza, turva, borda, thresh) = preProcessamento(image)


    contornosQuadrados = []

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )

    for c in cnts:
        #Obtem dados do ponto
        peri = cv2.arcLength(c, True)
        periPorcentagem = peri / image.shape[1] * 100
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
        (x, y, w, h) = cv2.boundingRect(c)

        # Carrega a imagem do modelo do quadrado de delimitação para otimizar a identificação
        img2 = cv2.imread('C:\\OneDrive\\TCC\\Imagens\\modelo-quadrado.jpeg')
        # Cria uma cópia de imagem para isolar o ponto
        img1 = image.copy()

        # Converte para tons de cinza
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Binariza as imagens dos pontos
        img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


        #Obtem os dados do ponto considerando sua perspectiva
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)


        # Define os pontos para alterar a perspectiva
        src_pts = np.array(box, np.float32)
        dst_pts = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32)
        ret = cv2.getPerspectiveTransform(src_pts, dst_pts)

        #Redimensiona a imagem para ficar no mesmo tamanho do ponto (20x20)
        img2 = cv2.resize(img2, (20,20))

        # Faz a transformação da perspectiva do ponto
        warp = cv2.warpPerspective(img1, ret, (20, 20))

        # Faz a erosão das imagens para facilitar a identificação
        kernel = np.ones((5, 5), np.uint8)
        img2 = cv2.erode(img2, kernel, iterations=1)
        warp = cv2.erode(warp, kernel, iterations=1)

        # Salva a imagem para comparação
        cv2.imwrite("C:\\OneDrive\\TCC\\Imagens\\img1.png", warp)

        # Compara ambas as imagens e retorna a semelhança entre elas
        (score, diff) = compare_ssim(warp, img2, full=True)
        diff = (diff * 255).astype("uint8")

        # Verifica se o ponto atual é igual ao quadrado alvo
        if score > 0.9 and score < 1.2 and len(aprox) <= 6 and len(aprox) >=4 and periPorcentagem > 3 and periPorcentagem < 10 and math.isclose(w, h, abs_tol=3):

            cv2.drawContours(image, [c], -1, 255, -1)

            #Adiciona na lista de contornos
            contornosQuadrados.append((x, y, w, h))


    return contornosQuadrados

def defineAreaQuestoes(contornosQuadrados, image):
    if len(contornosQuadrados) != 4:
        print("Apenas " + str(len(contornosQuadrados)) + " quadrados foram identificados. 4 são esperados.")
    else:
        areaQuestoes = (contornosQuadrados[0][0], contornosQuadrados[0][1] + contornosQuadrados[0][3]),\
                       (contornosQuadrados[1][0]  + contornosQuadrados[1][2], contornosQuadrados[1][1] + contornosQuadrados[1][3]),\
                       (contornosQuadrados[2][0], contornosQuadrados[2][1]),\
                       (contornosQuadrados[3][0] + contornosQuadrados[3][2], contornosQuadrados[3][1])

        paper = four_point_transform(image, np.array(areaQuestoes))
        cv2.imwrite("C:\\OneDrive\\TCC\\Imagens\\foto-teste.jpeg", paper)
        ccl.iniciaProcessamento(paper);


if __name__ == "__main__":

    #carrega a imagem
    image = cv2.imread("C:\\OneDrive\\TCC\\Imagens\\Testes\\Mariana\\imagem 4.jpeg")

    if image.shape[1] > 1500:
        image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

    imagemAuxiliar = image
    (dado, image) = detectaQRcode(image)
    if dado is None:
        imagemAuxiliar = eliminaSombras(image)
        (dado, imagemAuxiliar) = detectaQRcode(imagemAuxiliar)

    dado = 0
    if dado is None:
        print("Não foi possível detectar o QR Code. Tire a foto novamente mais próximo possível da folha, sem cortar os pontos")
    else:
        contornoQuadrados = processamentoImagem(image)

        if(len(contornoQuadrados) != 4):
            image = eliminaSombras(imagemAuxiliar)

            contornoQuadrados = processamentoImagem(image)


        defineAreaQuestoes(contornoQuadrados, image)
