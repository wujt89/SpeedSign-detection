# SpeedSign-detection
Raport z wykonanego zadania chiałbym przedstawić jako opis wszystkich funkcji oraz chronologię z jaką są one odpowiednio wywoływane. Ograniczę się do bardziej ogólnych opisów funkcji (nie będę wchodził w szczegóły co robi każda zmienna z osobna itp.), aby większą uwage skupić na procesie działania algorytmu
1. checkCircle():
    Funkcja służy do usuwania kółek, które znalezione zostały przez algorytm poszukiwania okręgów w sposób nieprawidłowy. Nie chodzi tutaj o błędne zaznaczenie         samego okręgu, natomiast o pozycje tego okręgu. Funkcja sprawdza, czy dany okrąg nie znajduję się wewnątrz innego okręgu. Jeżeli ten warunek zostaje spełniony       okrąg jest wyrzucany ze zbioru okręgów znelezionych.
2. loadAndCirclePhoto():
    Jako argument przyjomwany jest obraz na którym maja być znalezione okręgi. Wykorzystywana jest do tego funkcja z biblioteki "cv2" HoughCircles. Korzytsam z         metody cv2.HOUGH_GRADIENT_ALT, ponieważ uzyskałem dzięki niej zdecydowanie lepsze wyniki. Wszystkie znalezione okręgi (również te błędne o którym była mowa w       poprzednim punkcie) są nastepnie przekazywane do funkcji checkAndDrawRedCircles().
3. CheckAndDrawRedCircles():
    Pierwszą rzeczą jaką robi ta funkcja to przefiltrowanie kółek za pomocą funkcji z punktu pierwszego. Następnie z przefiltorwane kólka zamieniane są na kwadraty,     a wartości początkowe oraz końcowe tych kwadratów zapisywane są w zmiennej boxes która jest wartością zwracaną.
4. load():
    Jak sama nazwa wskazuję służy ona do załadowania wszystkich interesujących nas informacji na temat obiektu z pliku xml, który jest podany jako argument. Oprócz tego w przypadku gdy obraz został zaznaczony jako zły, losowane jest inne dowolne miejsce na obrazie, aby algorytm nauczył się również, że drzewa czy niebo to też nie to czego szukamy.
5. learn():
    Funkcja uczenia się. Uczy się na podstawie wycinków wyznaczonych przez wartości boxów w pliku xml. Oprócz tego w przypadkuznalezienia znaku który jest dobry, jest on losowo przycinany tak, aby zachowane było przynajmniej 50% jego oryginalnej powierzchni. Dzięki temu spełniona zostanie zasada IOU.
6. extractSinglePhotoClassify():
    Na podtsawie tej funkcji tłumaczony jest słownik na deskryptor na podstawie podanego zdjęcia dla pojedynczego obrazu.
7. extractDetect():
    Funkcja robi to samo co powyższa jednakże dla całego zbioru zdjęć podawanych dla 'detect'.
8. train():
    Funckja służy do nauczania modelu.
9. predictSinglePhotoClassify():
    Funkcja służaca do testowania modelu dla pojedynczego zdjęcia w opcji 'classify'. Zwracany jest przewidywany label, na podstawie deksryptora danego boxa.
10. predictDetect():
    Funkcja robi to samo co powyższa jednakże dla całego zbioru w opcji 'detect'.
11. evaluate():
    służy do zwracania wyniku klasywikacji, jeżeli 'label_pred' został wyznaczony jako '1', wtedy na konsole wypisywany jest 'speedlimit' jeżeli '0' to 'others'.
12. evaluateDetect():
    służy do zwracania wyniku detekcji. Na konsole wypiuswyane są podane w instrukcji dane w zależności od wyniku detekcji.
13. extract():
    Funkcja robi to samo co extractSingleClassify jednakże dla całego zbioru treningowego.
    
#1 załadowanie danych treningowych oraz testowych

#2 Uczenie modelu na podstawie danych treningowych

#3 Wyciąganie danych ze słownika dla zbioru treningowego

#4 Trenowanie modelu

#5 Na konsoli pokazuję sie informacja, ≥że trening i uczenie przebiegło w sposób prawidłowy, i oczekiwane jest wprowadzenie przez użytkownika wartości 'classify' lub 'detect'.

#6 W przypadku wprowadzenia 'classify' przebiega proces klasyfikacji oczekujemy na wprowadzenie przez użytkownika ilości plików jakie chcemy sprawdzić, następnie nazwy pierwszego pliku, następnie ilości pól jakie chcemy sprawdzić na danym obrazie, następnie współrzędne pierwszego pola. I tak w pętli.

#7 w przypoadku wpisania 'detect' przebiega proces detekcji na podstawie obrazów w podanym w instukcji folderze.
