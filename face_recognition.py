#%%
# importando bibliotecas
# necessário instalar o OpenCV - pip install opencv-python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
# %%
'''Para detectar olhos e bocas vamos utilizar 2 imagens, uma que contem apenas uma pessoa e a outra contém várias pessoas'''

image1 = cv2.imread("C:/Users/T-GAMER/projetos-data-science/face-recognition/data/image1.jpg",0)
image2 = cv2.imread("C:/Users/T-GAMER/projetos-data-science/face-recognition/data/image2.jpg",0)
print(image1.shape) #pra saber se a imagem carregou certo
print(image2.shape) #pra saber se a imagem carregou certo
# %%
# abrindo a imagem 1 em escala de cinza
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# %%
'''Como vamos utilizar o algoritmo Viola-Jones para detecção de face, olhos e boca, precisamos passar o path do algoritmo para o metodo CascadeClassifier(), por isso utilizaremos o comando abaixo para encontrar o path do arquivo XML que contem o algoritmo'''
cv2.data.haarcascades
# %%
'''Para detecção de face inicialmente utilizaremos o arquivo "haarcascade_frontalface_default.xml" '''
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# %%
'''Em seguida, vamos definir um metódo que aceite a imagem. Para detectar um rosto dentro dessa imagem,
    precisamos chamar o metódo detectMultiscale() do objeto face detector que inicializamos acima. 
    Uma vez que o rosto seja detectado, precisamos criar um retângulo ao redor do rosto. 
    Para isso, precisaremos dos componentes x e y da área do rosto além da largura e altura do rosto. 
    Com essas informações, poderemos criar um retângulo chamando o método rectangle do objeto OpenCV. 
    Finalmente, a imagem com um retângulo ao redor do rosto detectado será retornada pela função. 
    Para executar essas tarefas criaremos abaixo o metódo detect_face'''

def detect_face(image):
    face_image = image.copy()
    face_rectangle = face_detector.detectMultiScale(face_image, 
                                                    scaleFactor=1.5,
                                                    minNeighbors=2, 
                                                    minSize=(35, 35), 
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
    for(x, y, width, height) in face_rectangle:
        cv2.rectangle(face_image,(x, y),
                      (x + width, y + height),
                      (255,0,0),
                      7)
    return face_image
# %%
'''Para detecar o rosto, passaremos a imagem do rosto no metódo detect_face() criado acima.'''
detection_result_image1 = detect_face(image1)

'''Para verificar a imagem com detecção de rosto, passaremos o detection_result para o metódo imshow()'''
plt.imshow(cv2.cvtColor(detection_result_image1, cv2.COLOR_BGR2RGB))  # Convertendo BGR para RGB
plt.axis('off')  # Remove os eixos
plt.show()

# %%
'''Vamos tentar detectar as faces na imagem 2'''
detection_result_image2 = detect_face(image2)
plt.imshow(cv2.cvtColor(detection_result_image2, cv2.COLOR_BGR2RGB))  # Convertendo BGR para RGB
plt.axis('off')  # Remove os eixos
plt.show()

#Podemos ver que somente 5 rostos foram detectados, vamos tentar outro classificador
# %%
face_detector2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

def detect_face2(image):
    face_image = image.copy()
    face_rectangle = face_detector2.detectMultiScale(face_image, 
                                                    scaleFactor=1.1, #ajustando de 1.5 para 1.1 foi possível identificar todos os rostos
                                                    minNeighbors=2,
                                                    minSize=(35, 35), 
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
    
    for(x, y, width, height) in face_rectangle:
        cv2.rectangle(face_image,(x, y),
                      (x + width, y + height),
                      (255,0,0),
                      3)
    return face_image

detection_result2 = detect_face2(image2)

plt.imshow(cv2.cvtColor(detection_result2, cv2.COLOR_BGR2RGB))  # Convertendo BGR para RGB
plt.axis('off')  # Remove os eixos
plt.show()
# %%
