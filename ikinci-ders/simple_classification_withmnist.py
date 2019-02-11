import numpy as np
# numpy'ın random fonksiyonlarının herkeste aynı sonuçları döndürmesi için seed olarak 42 kullanıyoruz
np.random.seed(42)

# Bu Python script'i sadece numpy kullanarak basit yapıda bir yapay sinir eğitmektedir.
# Buradaki kodlar çok basit şekliyle örneklendiyse de bir yapay sinir ağının işleyişini ve ne şekilde öğrendiğini görmek için faydalı olabilir.

class Layer:
    def __init__(self):
        """Burada diğer sınıflarımızın extend edeceği bir sınıf olarak bir layer (katman) tanımlıyoruz. Bu katman kendi başına hiçbir şey yapmıyor, ancak genel bir katmanın yapısını görmemiz açısından önemli."""
        self.weights = np.zeros(shape=(input.shape[1], 10))
        bias = np.zeros(shape=(10,))
        
    def forward(self, input):
        """
        (batch sayısı, girdi sayısı) boyutlarında bir dizi alıyor ve (çıktı hücreleri sayısı, sınıflandırma yapılan kategori sayısı) şeklinde bir dizi döndürüyor. mnist evri setinde 10 adet kategori olduğu için sınıflandırma yapılan kategori sayısı 10'a eşit.
        Bir katman en basit şekilde öğrendiği ağırlıkları (weights) verilen girdi ile çarpıp yine öğrendiği önyargı (biases) ile toplar.
        """
        output = np.matmul(input, self.weights) + bias
        return output

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        # learning_rate, kayıp fonksiyonumuzu (loss function) minimize etmeye çalışırken her adımda katmanın öğrenilebilir parametrelerini ne kadar azalttığımızı belirler
        self.learning_rate = learning_rate
        
        # Ağırlıkları rastgele sayılar ile oluşturuyoruz. Burada normal dağılım kullanıyoruz, ancak daha karmaşık modelleri için ağırlıkları başlatma yöntemi optimize edilmesi gereken bir hiperparametredir.
        self.weights = np.random.randn(input_units, output_units)*0.01
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        return np.matmul(input, self.weights) + self.biases
      
    def backward(self,input,grad_output):
        grad_input = np.dot(grad_output,np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        grad_weights = np.transpose(np.dot(np.transpose(grad_output),input))
        grad_biases = np.sum(grad_output, axis = 0)
        
        # Burada stochastic gradient descent hesaplanıyor. Sonraki modellerimizde daha gelişmişleriyle değiştireceğiz.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input

class ReLU(Layer):
    def __init__(self):
        """
        ReLU, Rectified Linear Unit'in kısaltmasıdır ve bir aktivasyon fonksiyonu olarak giren sinyal 0'dan küçük ise 0, 0'a eşit ya da daha büyük ise sinyali olduğu gibi döndürür. Bu, derin öğrenme ile çözmeğe çalıştığımız problemlerin uygulandığı verisetleri için gerekli olduğu üzere linear akışı bozmamıza yardımcı olur.
        Constructor fonksiyonunundametodunda  hangi bir işlem yapmadığımız için pass diyerek bu metoda boş bir gövde tanımlıyoruz
        """
        pass
    
    def forward(self, input):
        """
        (batch sayısı, girdi hücre sayısı) boyutundaki matrise eleman düzeyinde ReLU aktivasyon fonksiyonunu uyguluyoruz.
        """
        return np.maximum(0,input)

    def backward(self, input, grad_output):
        """Kayıp fonksiyonunun girdimize göre eğimini hesaplıyoruz"""
        relu_grad = input > 0
        return grad_output*relu_grad 

def softmax_crossentropy_with_logits(logits,reference_answers):
    """Bu fonksiyon softmax aktivasyon fonksiyonu ile modelin tahminleri ile olması gereken değerler arasındaki cross entropy'yi hesaplıyor. logits, modelin tahminlerini, reference_answers ise gerçek olması gereken değerleri ifade ediyor"""
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    
    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    """logits ve reference_answers'dan cross entropy'nin eğimini hesaplıyor"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]
  
# Bu fonksiyon, keras kütüphanesini kullanarak örnek verisetimizi yükleyecek.
# keras, eğitimimizin geri kalanında da kullanacağımız, derin öğrenme araştırma ve geliştirmeleri için kullanımı oldukça kolay bir python paketidir.
# Önce keras'ı import etmemiz gerekiyor.
import keras
def load_dataset(dataset='mnist', flatten=False):
    """Keras, derin öğrenme ve makine öğrenmesi çalışmalarında sıkça kullanılan bazı verisetlerini kendi içinde barındırmaktadır.
    Bu verisetlerini ilk yüklemeye çalıştığınızda keras internetten otomatik olarak indirecek ve önbelleğe alaaktır.
    Bu yüzden kodun çalışması için ilk çalıştırmanızda internete bağlı olduğunuzdan emin olun.
    keras'ın sağladığı verisetlerinden biri mnist diye bilinir.
    mnist veriseti, el ile yazılmış 0'dan 9'a kadar olan rakamların 28x8 boyutunda siyah-beyaz görüntülerinden oluşur.
    Daha gelişmiş bir veriseti ise mnist'e göre daha karmaşık ancak aynı boyutlarda olan ve 10 kategoride kıyafetin 28x28 boyutlarında siyah-beyaz görüntülerinden oluşan fashion mnist verisetidir.
    Bu fonksiyon default olarak mnist verisetini yükleyecektir.
    Fashion mnist verisetini yüklemek için fonksiyonun 'dataset' isimli ilk parametresine 'fashion' verebilirsiniz"""
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # görselleri içeren x_train ve x_test dizilerini 0-1 aralığında normalize etmek için 255 ile bölüyoruz
    X_train = X_train.astype(float) / 255.0
    X_test = X_test.astype(float) / 255.0

    # Son 10k örneği geçerleme yapmak için ayırıyoruz
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # Bu basit örneğimizde convolution işlemi gerçekleştirmediğimiz için görselleri 28x28 boyutunda bir matris yerine 784 boyutunda bir vektör olarak yeniden boyutlandırıyoruz
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test
  
  # Verisetimizi ve grafiklerimizi çizmek için bu amaç için geliştirilmiş matplotlib paketini kullanacağız.
  # matplotlib de Anaconda ile kurulu gelecektir.
import matplotlib.pyplot as plt
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

plt.figure(figsize=[6,6])
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title("Label: %i"%y_train[i])
    plt.imshow(X_train[i].reshape([28,28]),cmap='gray');

   # Katmanlarımızı danımlayıp network isimli listemize ekleyerek yapay sinir ağımızı ileri beslemeli şekilde oluşturuyoruz.
   # Bu eğitimizde daima ileri beslemeli yapıyı kullanacağız 
network = []
network.append(Dense(X_train.shape[1],100))
network.append(ReLU())
network.append(Dense(100,200))
network.append(ReLU())
network.append(Dense(200,10))

def forward(network, X):
    """
    Sinir ağımızın tüm katmanlarını sırasıyla çalıştırarak aktivasyon fonksiyonlarını hsaplıyoruzReturn a list of activations for each layer. .
    Tüm aktivasyon fonksiyonlarının sonuçlarından oluşan listenin son elemanı, modelimizin tahminlerine karşılık geliyor.
    """
    activations = []
    input = X
    for i in range(len(network)):
        activations.append(network[i].forward(X))
        X = network[i].forward(X)
        
    return activations

def predict(network,X):
    """
    Bu fonksiyon, verilen girdiyi modelden geçirerek modelin tahminlerini hesaplıyor.
    """
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)

def train(network,X,y):
    """
    Verisetimizden batch sayısı kadar X ve y değerleri verildiğinde bunları modelimizden geçirerek eğitimi gerçekleştiriyoruz.
    İlk olarak tüm katmanlar için forward() metodunu çalıştırarak aktivasyonları hesaplıyoruz.
Daha sonra her katmanın backward() metodunu çağırarak son katmandan geriye doğru gidiyoruz.    
Her bir katman için backward() metodunu çağırdığımızda modelin gradyanı hesaplanmış ve uygulanmış oluyor.    
    """
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations
    logits = layer_activations[-1]
    
    # Kayıp fonksiyonunu ve gradyanı hesaplıyoruz
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
    
    for i in range(1, len(network)):
        loss_grad = network[len(network) - i].backward(layer_activations[len(network) - i - 1], loss_grad)
    
    return np.mean(loss)
  
  
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """Bu fonksiyon her çağırıldığında içinde çalıştırdığı döngünün her dönüşünde batch_size kadar örneği verisetinden alarak modele besliyor ve bu örnekler üzerinde eğitimin gerçekleşmesini sağlıyor"""
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        print('{}-{} arası örnekler modele besleniyor'.format(start_idx, start_idx + batchsize))
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
train_log = []
val_log = []
for epoch in range(25):
    print('{} eğitim turundan {}. tur başlıyor'.format(25, epoch + 1))

    for x_batch, y_batch in iterate_minibatches(X_train,y_train,batchsize=32,shuffle=True):
        train(network,x_batch, y_batch)
    
    train_log.append(np.mean(predict(network,X_train)==y_train))
    val_log.append(np.mean(predict(network,X_val)==y_val))

    print('{}. tur eğitimin sonunda:'.format(epoch + 1))
    print("Eğitim doğruluğu:",train_log[-1])
    print("Geçerleme doğruluğu:",val_log[-1])
    plt.plot(train_log,label='train accuracy')
    plt.plot(val_log,label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

