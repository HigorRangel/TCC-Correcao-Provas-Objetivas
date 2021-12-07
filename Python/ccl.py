import time

import cv2
import random
import argparse
import imutils
import math
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import csv


def corrigirQuestoes(questoesMarcadas, questoesDivididas, image):

    with open('Questões corretas.csv', mode='r') as infile:
        reader = csv.reader(infile)
        respostas = {int(rows[0]): rows[1] for rows in reader}

    print("Respostas Corretas: ", respostas )
    print("Respostas Marcadas: ", questoesMarcadas)
    corretas = 0
    naoMarcadas = 0
    erradas = 0
    numQuestao = 0

    for questao in questoesMarcadas:

        ponto = questoesDivididas[numQuestao]
        caracteresQuestoes = ['A', 'B', 'C', 'D', 'E']
        numAlternativa = None if questoesMarcadas[questao] == None else caracteresQuestoes.index(questoesMarcadas[questao])
        numAlternativaCorreta = caracteresQuestoes.index(respostas[questao])
        if questoesMarcadas[questao] == respostas[questao]:
            (index, x, y, w, h, area, cX, cY) = questoesDivididas[numQuestao][numAlternativa]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            corretas += 1
        elif questoesMarcadas[questao] is None:
            naoMarcadas += 1
        else:
            if(len(questoesDivididas[numQuestao]) == 5):
                (index, x, y, w, h, area, cX, cY) = questoesDivididas[numQuestao][numAlternativa]
                cv2.rectangle(image, (x, y), (x + w, y + h), (110, 110, 110), 2)

                (index, x, y, w, h, area, cX, cY) = questoesDivididas[numQuestao][numAlternativaCorreta]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            erradas += 1
        numQuestao += 1

    cv2.imshow("Correção", image)
    cv2.waitKey()

    print(corretas, naoMarcadas, erradas)



def separaImagensEmTres(img, pontosImagens):
    imagensDivididas = []
    (x, y, w, h) = pontosImagens
    imagensDivididas.append(img[y:y + h, x:(x + int(w / 3))])
    imagensDivididas.append(img[y:y + h, (x + int(w / 3)):(x + int((w / 3) * 2))])
    imagensDivididas.append(img[y:y + h, (x + int((w / 3) * 2)):(x + int((w / 3) * 3))])

    return imagensDivididas


def separaQuestoes(pontosImagensDivididas, fundo, rotulos, imagem):
    pontos = getPontosQuestoes(pontosImagensDivididas)
    #Filtra todos os pontos pela área (para eliminar pontos menores que à area das alternativas, como pequenas rasuras)

    pontos = list(filter(lambda c: ((c[3] * c[4]) / (fundo[2] * fundo[3]) * 100 > 0.08) and
                                   ((c[3] * c[4]) / (fundo[2] * fundo[3]) * 100 < 2)
                                   and math.isclose(c[3], c[4], abs_tol=6), pontos))

    #ordena as questões d
    questoesDivididas = divideQuestoes(pontos, 45)
    #respostasMarcadas = getRespostasMarcadasMetodo1(fundo, questoesDivididas)
    respostasMarcadas = getRespostasMarcadasMetodo2(fundo, questoesDivididas, imagem)

    return (respostasMarcadas, questoesDivididas)


def getRespostasMarcadasMetodo2(fundo, questoesDivididas, image):
    if len(questoesDivididas) == 45:
        ordenaQuestoesLeftToRight(questoesDivididas)
        respostasMarcadas = {}

        img2 = cv2.imread('C:\\OneDrive\\TCC\\Imagens\\modelo-circulo-preenchido.jpg')
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        caracteresQuestoes = ['A', 'B', 'C', 'D', 'E']

        for indiceIteraQuestao in range(0, len(questoesDivididas)):
            if (len(questoesDivididas[indiceIteraQuestao]) == 5):
                respostasMarcadas[indiceIteraQuestao + 1] = None

                for indiceIteraAlternativa in range(len(questoesDivididas[indiceIteraQuestao])):
                    alternativaAtual = questoesDivididas[indiceIteraQuestao][indiceIteraAlternativa]

                    img1 = image.copy()[alternativaAtual[2]:alternativaAtual[2] + alternativaAtual[4],
                           alternativaAtual[1]:alternativaAtual[1] + alternativaAtual[3]]

                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                    img1 = cv2.resize(img1, (20, 20))
                    img2 = cv2.resize(img2, (20, 20))

                    #cv2.imwrite("C:\\Users\\higor\\OneDrive\\TCC\\Imagens\\metodo-2\\" + str(indiceIteraQuestao + 1) + "-" + str(indiceIteraAlternativa) + ".png", img1)


                    (score, diff) = compare_ssim(img1, img2, full=True)
                    diff = (diff * 255).astype("uint8")

                    if score > 0.18:
                        if respostasMarcadas[indiceIteraQuestao + 1] == None:
                            respostasMarcadas[indiceIteraQuestao + 1] = caracteresQuestoes[indiceIteraAlternativa]
                        else:
                            respostasMarcadas[indiceIteraQuestao + 1] = None
                            break
            else:
                respostasMarcadas[indiceIteraQuestao + 1] = None
        return respostasMarcadas





def getRespostasMarcadasMetodo1(fundo, questoesDivididas):
    if len(questoesDivididas) == 45:
        ordenaQuestoesLeftToRight(questoesDivididas)
        respostasMarcadas = {}


        for indiceIteraQuestao in range(0, len(questoesDivididas)):
            # print(indiceIteraQuestao, len(questoesDivididas[indiceIteraQuestao]))

            if len(questoesDivididas[indiceIteraQuestao]) < 5:
                respostasMarcadas[indiceIteraQuestao + 1] = None
            else:
                respostasMarcadas[indiceIteraQuestao + 1] = None
                for indiceMarcados in range(0, len(questoesDivididas[indiceIteraQuestao])):
                    if (questoesDivididas[indiceIteraQuestao][indiceMarcados][5] / fundo[4] * 100 > 0.07):
                        caracteresQuestoes = ['A', 'B', 'C', 'D', 'E']

                        if respostasMarcadas[indiceIteraQuestao + 1] == None:
                            respostasMarcadas[indiceIteraQuestao + 1] = caracteresQuestoes[indiceMarcados]

                        else:
                            respostasMarcadas[indiceIteraQuestao + 1] = None
                            break

    else:
        print("Não foram encontradas 45 questões. Capture a imagem novamente. ({} encontradas".format(
            len(questoesDivididas)))
    return respostasMarcadas


def getPontosQuestoes(pontosImagensDivididas):
    pontos = []
    # Itera pelos pontos das 3 imagens divididas
    # (1º img = questões de 1 a 15 || 2º img = questões de 16 à 30 || 3º img = questões de 31 à 45
    for i in range(0, len(pontosImagensDivididas)):

        # Itera por todos os pontos encontrados na parte da imagem em questão
        for j in range(0, len(pontosImagensDivididas[i])):
            pontos.append(pontosImagensDivididas[i][j])

    return pontos


def ordenaQuestoesLeftToRight(questoesDivididas):
    contador = 0
    for questao in questoesDivididas:
        questao.sort(key=lambda c: c[1])


def divideQuestoes(pontos, qtdQuestoes):
    questoesDivididas = []
    len_l = len(pontos)

    questao = []

    indiceOrdenacao = 0
    restoFechaQuestao = 0

    while indiceOrdenacao <= len_l:

        if indiceOrdenacao % 5 != restoFechaQuestao and indiceOrdenacao != len_l:

                pontoY = pontos[indiceOrdenacao][7]
                pontoYanterior = pontos[indiceOrdenacao - 1][7]
                if pontoYanterior >= pontoY - 10 and pontoYanterior <= pontoY + 10:
                    questao.append(pontos[indiceOrdenacao])
                else:
                    if len(questao) < 5:
                        numQuestoesRasuradas = 5 - len(questao)
                        for indiceSubtracao in range(numQuestoesRasuradas):
                            if restoFechaQuestao == 0:
                                restoFechaQuestao = 5
                            restoFechaQuestao -= 1

                    questoesDivididas.append(questao)
                    questao = []
                    questao.append(pontos[indiceOrdenacao])
        else:
            if indiceOrdenacao != 0:
                questoesDivididas.append(questao)
                questao = []

            if indiceOrdenacao != len_l:
                questao.append(pontos[indiceOrdenacao])

        indiceOrdenacao += 1

    return questoesDivididas

def getPontos(imagem, thresh, args):
    output = cv2.connectedComponentsWithStats(
        thresh, args["connectivity"], cv2.CV_64F)
    (numRotulos, rotulos, stats, centros) = output

    listaPontos = []
    fundo = {}
    imagensDivididas = []
    pontosImagensDivididas = [[], [], []]
    fundo = (stats[0, cv2.CC_STAT_LEFT], stats[0, cv2.CC_STAT_TOP], stats[0, cv2.CC_STAT_WIDTH], stats[0, cv2.CC_STAT_HEIGHT], stats[0, cv2.CC_STAT_AREA], centros[0][0], centros[0][1])
    imagensDivididas = separaImagensEmTres(imagem, (stats[0, cv2.CC_STAT_LEFT], stats[0, cv2.CC_STAT_TOP], stats[0, cv2.CC_STAT_WIDTH], stats[0, cv2.CC_STAT_HEIGHT]))

    for i in range(1, numRotulos):
        #Pega os boundingBoxes
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centros[i]

        #Adiciona os pontos extraidos em uma lista
        listaPontos.append((i, x, y, w, h, area, cX, cY))

        #Divide os pontos encontrados pelas 3 regiões das respostas (imagem dividida em 3 horizontalmente)
        if x <= fundo[0] + fundo[2] / 3:
            pontosImagensDivididas[0].append((i, x, y, w, h, area, cX, cY))
        elif x <= fundo[0] + (fundo[2] / 3) * 2:
            pontosImagensDivididas[1].append((i, x, y, w, h, area, cX, cY))
        else:
            pontosImagensDivididas[2].append((i, x, y, w, h, area, cX, cY))
    return (fundo, listaPontos, imagensDivididas, pontosImagensDivididas, rotulos)


def desenhaPontos(imagem, listaPontos, fundo, rotulos):
    copiaImagem = imagem.copy()
    for i in range(0, len(listaPontos)):
        (index, x, y, w, h, area, cX, cY) = listaPontos[i]
        cv2.rectangle(copiaImagem, (x, y), (x + w, y + h), (50 + i, 40 + i, 255 - i), 2)
        cv2.circle(copiaImagem, (int(cX), int(cY)), 1, (0, 255, 1), -1)

        componentMask = (rotulos == index).astype("uint8") * 255


def iniciaProcessamento(image):

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
                    help="path to input image", default=image)
    ap.add_argument("-c", "--connectivity", type=int, default=8,
                    help="connectivity for connected component analysis")
    args = vars(ap.parse_args())

    image = args["image"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


    (fundo, listaPontos, imagensDivididas, pontosImagensDivididas, rotulos) = getPontos(image, thresh, args)

    (questoesMarcadas, questoesDivididas) = separaQuestoes(pontosImagensDivididas, fundo, rotulos, image)

    corrigirQuestoes(questoesMarcadas, questoesDivididas, image)

