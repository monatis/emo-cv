# Beni Oku

Bu dersimizde çiçekleri tanıyan basit yapılı bir CNN modeli eğiteceğiz.

Derste kullanacağımız verisetini [buraya tıklayarak indirebilirsiniz](http://download.tensorflow.org/example_images/flower_photos.tgz).

Dosyaları arşivden çıkarmak için ücretsiz ve açık kaynak kodlu [7Zip programını](https://www.7-zip.org/) kullanabilirsiniz.

# Keras'ın Conda sanal ortamına kurulması

Aşağıdaki komut satırı kodları, versiyonlarıyla birlikte kendi makinemde test ettiğim ortamı kurmanızı sağlayacaktır.

`conda create -n ailabs python=3.6.5`

`activate ailabs`

`python -m pip install numpy==1.12.0`
`python -m pip install tensorflow==1.10.0`
`python -m pip install keras pillow opencv-contrib-python matplotlib scikit-learn`

Örnek kodları bu ortam içerisinde koşturunuz.

# Nasıl çalıştırılır?

`split_dataset.py`, klasörlere ayrılmış verisetini training ve validation şeklinde ikiye ayırmak için kullandığımız kodu barındırıyor.

`python split_dataset.py -d D:\datasets\flower_photos -o D:\datasets\flower_photos -p 85`

`transfer_learning_mobilenet.py`, Keras'ın bize sağladığı kolaylıklardan yararlanarak MobileNetV2 mimarisi üzerinde transfer learning ile eğitim örneğini içermektedir.

`python transfer_learning_mobilenet.py -t D:\datasets\flower_photos\training -v D:\datasets\flower_photos\validation -c 5`