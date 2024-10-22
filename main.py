from PIL import Image
import pickle
from img2vec_pytorch import Img2Vec

img2vec = Img2Vec()
img = Image.open("img79.jpg")
vec = img2vec.get_vec(img)
with open("animalclf2.p",'rb') as f:
    model = pickle.load(f)

print("Given image is of a",model.predict([vec])[0])
f.close()