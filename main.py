import numpy as np
import gzip
import MLP
import cv2

def load_data_test(): #meload data test untuk soal 2 berisi 35 gambar 22 gambar fasion dan 13 gambar non-fashion
    data = np.zeros(0)
    for i in range(1, 36):
        img = cv2.imread('img/%s.jpg' % i) #membaca gambar pada folder img
        img = cv2.resize(img, (28, 28)) # mengubah gambar menjadi 28 x 28 px
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # mengubah gambar menjadi grayscale
        img = img.flatten() #menggabungkan array menjadi satu baris 
        data = np.append(data, img) #memasukkan data ke list
    return data.reshape(35, 28*28) #mereturn data menjadi 35 gambar yang berukuran 28x28


test_images = load_data_test()

#model = MLP.CreateModel(images, label, lr=0.01) #mentrain image fungsi ini dijalankan untuk mencari hasil train dari MLP. hasil train terdapat pada file "mlp_weight.h5"
model = MLP.LoadModel() #meload hasil tarin yang telah didaptkan sebelumnya dari fungsi di atas (line 19)
predict = MLP.Predict(test_images, model)

output = np.zeros(predict.shape[0])
correct = 0
for j in range(len(predict)):
    idx = 0
    p = predict[j]
    for i in range(len(p)):
        if(p[idx] < p[i]):
            idx = i
    if(p[idx] < 0.5):
        idx = -1
    output[j] = idx
    if(idx == 0):
        print(j,'T-shirt/top')
    elif idx == 1:
        print(j,'Trouser')
    elif idx == 2:
        print(j,'Pullover')
    elif idx == 3:
        print(j,'Dress')
    elif idx == 4:
        print(j,'Coat')
    elif idx == 5:
        print(j,'Sandal')
    elif idx == 6:
        print(j,'Shirt')
    elif idx == 7:
        print(j,'Sneaker')
    elif idx == 8:
        print(j,'Bag')
    elif idx == 9:
        print(j,'Ankle boot')
    elif idx == -1:
        print(j,'Other')

