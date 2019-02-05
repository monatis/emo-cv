Derin Öğrenme ile Görüntü Tanıma eğitimimizin ilk gününde temel Python programlama, Numpy ile n boyutlu matris işlemleri ve OpenCV ile temel dijital görüntü işlemlerini ele alacağız. Bunun için bilgisayarınızda bazı yazılım ve kütüphanelerin kurulu olması gerekmektedir. Aşağıda, kurulumlar için takip etmeniz gereken adımlar yer almaktadır. 

## Anaconda kurulumu

Anaconda, birçok Python kütüphanesini içinde barındıran, çalışma ortamlarını farklı projeler için izole etmek ve paket gereksinimlerini yönetmek için gelişmiş araçlar barındıran bir Python dağıtımıdır.

 * Kurulumu yapmak için [buraya tıklayın](https://www.anaconda.com/distribution/).
 * İşletim sisteminize göre Windows, MacOS ya da Linux sekmesini seçin.
 * "Python 3.7 version" altında işlemci mimarinize göre 64-bit ya da 32-bit kurulum dosyasını bilgisayarınıza indirerek çalıştırın.
 * Kurulum yazılımında talimatları takip ederken "Add Python to system path" seçeneği çıkar ise bu seçeneği işaretlediğinizden emin olun.

## OpenCV kurulumu

OpenCV, bilgisayarlı görü yazılımları geliştirmek için gelişmiş algoritmalar ve görüntü işleme için kullanışlı veri tipleri sağlayan ve endüstri standardı haline gelmiş bir kütüphanedir. Biz de kodlarımızı geliştirirken OpenCV'den sıkça yararlanacağız. Yukarıda tarif edilen Anaconda kurulumunu yapmış olmanız halinde OpenCV'yi kurmak için:

 * Windows'ta komut satırını yönetici yetkileriyle, MacOS ve Linux'ta ise terminali açın.
 * şu komutu çalıştırın: `pip install opencv-contrib-python`


## Editör kurulumu

 * Python'da programlar, .py uzantılı ve kod komutlarını içeren metin formatındaki dosyaların `python` komut satırı aracına argüman olarak verilmesiyle çalıştırılır ve yorumlanan bir dil olduğundan derleme gerektirmez. Dolayısıyla Metin dosyalarını düzenleyebildiğiniz her editör ile Python programı geliştirmek mümkündür. Halıhazırda kod geliştirmeye aşina olduğunuz bir kod editörü (Eclipse, VC Code vb.) var ise bunu kullanmaya devam etmeniz üretkenliğinizi sürdürmenizi sağlayacaktır. Python'a yönelik  daha gelişmiş editör desteği almak için halıhazırda kullandığınız kod editörünün sonuna ' python' ifadesini ekleyerek Google araması yaptığınızda editörünüz için geliştirilmiş Python eklentisini ya da uzantısını bulmanız mümkündür.

İlk kez bir kod editörü seçeceklere ise her platformda çalışabilen ve açık kaynak kodlu bir kod editörü olan Visual Studio Code'u öneriyoruz.

 * VS Code'u kurmak için [buraya tıklayın](https://code.visualstudio.com/).
 * "Download for XXX Stable Build" butonuna tıklayarak kurulum dosyasını bilgisayarınıza indirin ve çalıştırarak kurulumu tamamlayan (XXX işletim sisteminizin adıdır ve otomatik olarak tanınacaktır).
