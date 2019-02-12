import cv2
import numpy as np
import random as rd

# Bu çalışmada Spirograph adı verilen çizim aracını uygulamasını yaptım.
# Bir çoğunuzun bu 'oyuncağı' hatırladığınızı düşünüyorum.
# Hypotrochoid adı verilen bu poligonuna ait formüle http://mathworld.wolfram.com/Hypotrochoid.html adresinden ulaşabilirsiniz.

width,height = 500 , 500 # Kanvas boyutları
canvas = np.zeros((width,height,3),dtype='uint8') # boş bir kanvas oluşturuldu.

# shifter fonksiyonu openCV de sol üst köşede olan (0,0) noktasının ortasına almamızı sağlıyor.
shifter = lambda x : (x[0]+width//2,-x[1]+height//2)  # Vektörel bir toplama işlemi
class Graph:
    """
    Args:
        outter_r (int) : Dış çemberin yarıçapı
        inner_r  (int) : İç çemberin yarıçapı
        h (int) : Kalem ucu noktasının iç çemberin merkezine olan uzaklığı
    
    """
    def __init__(self,outter_r,inner_r,h):
        self.outter_r =outter_r
        self.inner_r=inner_r
        self.h = h

    def graphPoint(self,t):
        """Kalem ucu noktasının t anında bulunduğu kordinatı döner"""
        x = (self.outter_r-self.inner_r)*np.cos(t) + self.h *np.cos((self.outter_r-self.inner_r)/self.inner_r*t)
        y = (self.outter_r-self.inner_r)*np.sin(t) + self.h *np.sin((self.outter_r-self.inner_r)/self.inner_r*t)
        return (int(x),int(y))
        
    

    def graphIt(self,color=(50,50,50),linewidth=1,step=0.1):
        """
            Color : Kalem ucu noktasının rengi
            linewidth : Kalem ucu noktasının kalınlığı
            step : Çizgi noktaları arası örneklem aralığı

        """
        t1 = step

        while True:      # Klavyeden Q tuşuna basılarak çıkılan bir döngü oluşturuyoruz.

            #Canvas matrisine class objesine ait graphPoint fonksiyonunu kullanarak iki nokta veriyoruz
            # Burada dikkat edilmesi gereken iki nokta arasında step farkla iki örnek oluşturmamız
            # Ayrıca shifter fonksiyonunu kullanarak (0,0) noktasını (250,250) noktasına map ediyoruz.    
            cv2.line(canvas,shifter(self.graphPoint(t1)),shifter(self.graphPoint(t1+step)),color,linewidth)
            
            
            t1 += step
            cv2.imshow('Spinograf',canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

#Main 
if __name__ == '__main__':
    for i in range(0,20): # 20 adet
        randomColor= (rd.randint(0,255),rd.randint(0,255),rd.randint(0,255)) # BGR renk
        randomOutter = rd.randint(100,300) 
        randomInner = rd.randint(50,100)
        randomH = rd.randint(1,50)

        Graph(randomOutter,randomInner,randomH).graphIt(randomColor) # Graph objesi üzerinden graphIt fonksiyonun çalıştırıyoruz.
    
