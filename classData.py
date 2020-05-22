import numpy as np 
import pickle as pk
import random
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

class ParserDeMerda:
    def __init__(self, pickle):
        self.curves = pk.load(open(pickle, 'rb'))
        self.torchify = False
    def __len__(self):
        return len(self.curves)
    def __getitem__(self, index):
        if self.torchify:
            return torch.tensor(self.curves[index])
        return self.curves[index]

class Parser:
    def __init__(self, csvFilename, csvMetadataFilename):
        file1 = pd.read_csv(csvFilename)
        estrelles = set(file1['object_id'])
        corves = {x: [] for x in estrelles}
        array = np.array(file1)
        for line in array[:]:
            corves[line[0]].append((line[1], line[3])) #(#timestamp, #flux)
        self.corves = {x: [y[1] for y in sorted(corves[x], key = lambda x: x[0])]  for x in corves}
        file2 = np.array(pd.read_csv(csvMetadataFilename))[:]
        self.categories = {x[0]: int(x[-1]) for x in file2 }
        self.estrelles = list(estrelles)
        self.tamany = max([len(self.corves[x]) for x in corves])
        self.numClases = len(set(self.categories.values()))
        print(self.numClases)
        self.lut = {x: num for num, x in enumerate(set(self.categories.values()))}
        self.maximum = max([max(x) for x in self.corves.values()])
        del file1, corves, array, file2

    def __len__(self):
        return len(self.categories)
    
    def __getitem__(self, index):
        est = self.estrelles[index]
        x = np.zeros(self.tamany)
        x[:len(self.corves[est])] = np.array(self.corves[est])

        #x = (x - x.min())/(x.max() - x.min()) #0 , 1

        y = np.zeros(self.numClases)
        y[self.lut[self.categories[est]]] = 1

        y = y.reshape((1, -1))
        x = x.reshape((1, 1, -1))

        return x, y
try:
    obj = pk.load(open('datasetRick.p', 'rb'))
except:
    obj = Parser('training_set.csv', 'training_set_metadata.csv')
    pk.dump(obj, open('datasetRick.p', 'wb'))

#### Model Definition ####
#Best: primer DO = 0.35, Segon DO = 0.3. Primera Conv: 7KS
model = tf.keras.models.Sequential([
    #tf.keras.layers.Embedding(int(obj.maximum + 1), 128),
    tf.keras.layers.Conv1D(32, 7, activation = 'relu', padding = 'same'),\
    tf.keras.layers.MaxPooling1D(5, padding = 'same'), \
    tf.keras.layers.Conv1D(32, 7, activation = 'relu', padding = 'same'), \
    tf.keras.layers.GlobalMaxPooling1D(),\
    tf.keras.layers.Dense(128, activation = 'relu'),\
    tf.keras.layers.Dense(56, activation = 'relu'),\
    tf.keras.layers.Dense(26, activation = 'relu'),\
    tf.keras.layers.Dropout(0.2),\
    tf.keras.layers.Dense(obj.numClases, activation = 'softmax') # tf.keras.layers.Dense(int(obj.numClases * 1.2), activation = 'relu'), \
])



LR = 0.001

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  LR,
  decay_steps=int(len(obj)*0.80),
  decay_rate=1,
  staircase=False)

lossF = tf.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr_schedule)

@tf.function
def trainStep(x, y, lossFunction, model, optimizer):
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = lossFunction(y, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = 0
    if tf.math.equal(tf.math.argmax(prediction[0]), tf.math.argmax(y[0])):
        acc = 1

    return loss , acc 

def test_step(x, y, lossFunction, model):

    prediction = model(x)
    if tf.math.equal(tf.math.argmax(prediction[0]), tf.math.argmax(y[0])):
        return 1, lossFunction(y, prediction)
    return 0, lossFunction(y, prediction)


def main(model, data, epoches, optimizer, lossF):
    global LR
    printPeriod = 500
    lossTrain = []
    lossTest = []
    lossTrainIWillPlot = []
    startTest = int(len(data) * 0.80) #canvi 95% ---> 80%
    for i in range(epoches):
        prec = 0
        precTrain = 0
        for x in range(len(data)):
            inp, out = data[x]
            if x < startTest:
                loss, accuracy = trainStep(inp, out, lossF, model, optimizer)
                lossTrain.append(loss)
                precTrain += accuracy
            else:
                acc, loss = test_step(inp, out, lossF, model)
                prec += acc
                lossTest.append(loss)
            if x%printPeriod == 0:
                model.save_weights('./weights/')
                if x < startTest:
                    lossTrainIWillPlot.append(sum(lossTrain)/len(lossTrain))
                    print("Entrenant... sample numero {} amb loss {} i accuracy {}".format(x, sum(lossTrain)/len(lossTrain), precTrain/(x + 1)))
                    plt.plot(lossTrainIWillPlot)
                    plt.savefig('train.png')
                    plt.clf()
                    lossTrain = []
                else:
                    print("Testant... sample numero {} amb accuracy {} i loss {}".format(x, prec/(x - startTest + 1), lossTest[-1]))
                    plt.plot(lossTest)
                    plt.savefig('test.png')
                    plt.clf()
                

        
main(model, obj, 50, optimizer, lossF)
cat = list(obj.categories.values())
print(dict([(categoria, round(cat.count(categoria)/len(cat) * 100, 5)) for categoria in set(obj.categories.values())]))