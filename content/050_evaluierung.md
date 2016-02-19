# Evaluierung \label{head:evaluation}

## Trainingsumgebung

Das Training wird auf dem, von der Hochschule Für Technik und Wirtschaft Berlin (HTW Berlin) zur Verfügung gestellten Server, *deepgreen02*, ausgeführt.

---------------     ----------------------------------------------------
CPU Model           Intel(R) Xenon(R) CPU E5-2609 v2 @ 2.5Ghz
CPU Kerne           4 Kerne, x86_64, Intel
CPU Anzahl          8 CPUs
Arbeitsspeicher     66 GB
Grafikkarte         Nvidia Tesla K20m
GPU Name            GK110
GPU MHz             706
GPU Speicher        5120 MB
--------------      ----------------------------------------------------
  : Trainingsserver Hardwareinfo. \label{table:deepgreen02}

## Vorverarbeitung

### Subbilder Generierung \label{head:sliding_window}

Im Gesamten müssen für die Trainingsdaten laut der Berechnung \ref{lst:patches_calc} 28460160 Subbilder, wie im Kapitel \ref{head:vor-nachverarbeitung} beschrieben, generiert werden. Diese Subbilder haben je nach Größe der gewählten Überlappung eine unterschiedliche Bildgröße. Bei einer Überlappung von 2 Pixel, besitzen sie die Bildgröße (5x5) und bestehen somit aus 25 Pixeln.


```
{caption="Kalkulation der zu generierenden Unterbilder" frame=single label=lst:patches_calc .java}
48 Bilder * 540p * 258p + 96 Bilder * 540p * 420p = 28460160 Unterbilder
28560160 Bilder * 25 Pixel = 714004000 Pixel
714004000 * (float32 -> 4 Bytes) -> 2856016000 Bytes -> 2.856 GB
```

In einem Benchmark wurden die drei implementierten Generierungsverfahren einander gegenübergestellt und die Resultate in der Tabelle \ref{table:sliding-window} aufgelistet. Hierbei ist ersichtlich, dass die nicht durchmischte Generierung anhand der *numpy*-Matrix-Index-Operationen mit 62 Sekunden am schnellsten ist.

| Verfahren          | Sekunden | Minuten | Maschine  |
|:-------------------|:--------:|:-------:|----------:|
| Schleifen          | 189      | 3,15    |deepgreen02|
| Schleifen, Zufall  | 319      | 5,2     |deepgreen02|
| Matrix-Op.         | 62       | 1       |deepgreen02|
| Matrix-Op, Zufall  | 173      | 2,8     |deepgreen02|
| Kompletter Zufall  | 100      | 1,6     |deepgreen02|

  : Vergleich der Sliding Window Algorithmen. \label{table:sliding-window}

Wenn nun die Zufälligkeit mit einberechnet wird, werden die Vorteile der *Totalen-Zufall* Methode sichtbar. Diese benötigt mit 100 Sekunden erheblich länger als die sortierte *numpy* Optimierte variante, dafür müssen die Daten jedoch nicht mehr durchmischt werden.

Es hat sich ergeben, dass das Durchmischen mehr Zeit in Anspruch nimmt als die Generierung der Subbilder. Dadurch ist die *Totalen-Zufall* Methode die schnellste Zufallsvariante.

Zu beachte ist, dass nicht garantiert ist, dass jedes Subbild berücksichtigt wird. Wiederum können andere Subbilder doppelt vorkommen.

![Vergleich: Batchweise oder Komplette Zufälligkeit \label{fig:batch_vs_fully}](images/fully_vs_batch_random.png)

In der Abblidung \ref{fig:batch_vs_fully} ist zu sehen, dass die komplette Zufallsvariante zwar bei den Trainingskosten nicht besser abschneidet als die Batchweise zufälligkeit, bei den Validierungskosten erreicht sie jedoch ein besseres Resultat. Ebenfalls im Verlauf Trainingskosten ist eine verbesserte Regelmäßigkeit abzulesen.

Für weitere Trainingsdruchläufe, wird die Komplette Zufälligkeit verwendet.

## Einfaches kNN

### Größe des Minibatch

Die Größe des *Minibatch* definiert wie viele Datensätze parallel trainiert werden. Diese Größe hat zwei wesentliche Einflüsse auf das Trainingsverhalten.

Als Vorteil benötigt das Trainieren einer Epoche bei einem größeren *Minibatch* weniger Zeit, da durch Vektorberechnungen parallel mehrere Datensätze direkt auf der *GPU* berechnet werden. Zusätzlich werden durch größere Gruppen weniger Kopiervorgänge zwischen dem CPU und GPU Arbeitsspeicher getätigt, was die Performance zusätzlich steigert.

Als Nachteil kann ein zu großen *Minibatch* den Trainingsfortschritt pro Epoche verringern, da nur am Ende jedes *Minibatch* Durchlauf die Gewichte angepasst werden. Dadurch werden die Gewichte bei einem kleineren *Minibatch* öfter angepasst.

![Trainingsverlauf  \label{fig:mb-vergleich}](images/training_minibatch_vergleich_2.png)

In der Abblidung \ref{fig:mb-vergleich} ist zu sehen, wie der Trainingsfortschritt pro Epoche sich verringert, um so größer der *Mini-Batch* gewählt wird. Gleichzeitig verringert sich aber auch die Lerndauer. Um möglichst viele Trainingsvariationen prüfen zu können, wird eine *Mini-Batch-Größe* von $500$ gewählt. Für das beste Resultat wird für die Kaggle Wettbewerb mit einer kleineren von $100$ Trainiert.

### Fixe gegenüber dynamischer Lernrate

![Unterschiedliche Trainingsverhalten bei fixer oder dynamischer Lernrate \label{fig:fix_dyn_eta}](images/lernrate_fix_vs_decrease.png)

In der Abbildung \ref{fig:fix_dyn_eta} ist der Trainingsablauf einer Fixen Lernrate mit einer dynamisch abnehmender Lernrate gegenübergestellt. Dabei ist vor allem bei den Validierungskosten den Unterschied sichtbar.

Anfangs ist bei beiden Validierungskurven eine sprunghafte Verbesserung sichtbar. Ab der zweiten Epoche ist das Minimum bereits fast erreicht.
Nun werden größere Sprünge bei die Kurve mit dem fixen Lernrate sichtbar, wobei die Kurve mit der dynamische Lernrate konstant bleibt. Dies, da die dynamische Lernrate pro Epoche verkleinert wird, womit der Lernvorgang verlangsamt rsp. verfeinert wird.

\FloatBarrier

### RMSprop gegenüber Stochastik Gradient Descent und Momentum

In der Abbildung \ref{fig:rmsprop_vs_sgd} ist ersichtlich, dass die beiden Gradientenabstiegsverfahren *RMSprop* und *Stochastik Gradient Descent*, *SGD*, sich in zwei Merkmalen unterscheiden. Zum Einen ist zu sehen, dass der *RMSprop* bereits am Anfang der ersten Epoche bessere Werte erzielt als der *SGD* am Ender der dritten. Die zweite Unterschied zeigt sich bei den Validierungskosten. Wo der *RMSprop* immer wieder kleine Ausschläge aufzeigt, nimmt der *SGD* kontinuierlich ab, wenn auch minimal.

![Unterschiedliche Trainingsverhalten der beiden Algorithmen RMSprop und SGD+Momentum \label{fig:rmsprop_vs_sgd}](images/rmsprop_vs_sgd.png)

Da der *RMSprop* Algorithmus offensichtlich schneller lernt, basieren weitere Tests ausschließlich auf dem *RMSprop* Algorithmus. Hiermit konnte das Resultat der Abbildung \ref{rmsprop-compair} bestätigt werden.

\FloatBarrier

### L2-Regularisation und Dropout

Die kleinste zum Trainieren verwendete Datenmenge mit jeweils 20 Trainings- und Validierungsbilder, ergeben 3486240 Subbilder zum Trainieren als auch zum Validieren. Damit besteht eine genügend große Datenmenge zur Verfügung mit welcher *Overfitting* durch genügend Trainingsdaten entgegengewirkt wird. Wird die gesamte Trainingsmenge von 144 Bilder verwendet, wir dies noch verstärkt, da die Trainingsmenge um das 7-Fache ansteigt.

![Bei Verwendung von Dropout wird das Resultat schlechter \label{fig:comp_dropout](images/comp_dropout.png)

Bestätigt wird diese Annahme durch Testläufte. In der Abbildung \ref{fig:comp_dropout} ist zu sehen, dass die Resultate sich mit dem verwenden von Regularisierungs-Methoden verschlechtern.

### Einfluss der unsichtbaren Schicht

Während der heuristischen Hyperparametersuche ist aufgefallen, dass die unsichtbare Schicht, solange sie größer ist als die Eingangsschicht, keine signifikante Änderungen mit sich bringt. Dies ist ebenfalls in der Abbildung \ref{fig:comp_hidden} an einem Beispiel sichtbar. Hier werden einmal von 21 bis 500 Neuronen in der unsichtbaren Schicht verwendet. Es ist sichtbar dass erst ab weniger als 100 Neuronen das Resultat negativ beeinflusst wird. Dies jedoch fast nicht spürbar. Erst wenn die Schicht zu klein wird, wird das Lernverhalten negativ beeinflusst.

![Auswirkung der unsichbaren Schicht anhand der Anzahl Neuronen \label{fig:comp_hidden}](images/hidden_units.png)

\FloatBarrier

### Denoising Autoencoder


### Bestes Resultat


**Kaggle Resultat**

## Stacked Denoising Autoencoder

Bei Stacked Deonising Autoencoder, handelt es sich um kNN mit mehreren aufeinander folgenden *AutoencoderLayer*. Es wird nun überprüft wie sich die Präzision pro neue Schicht verändert, wenn es voraustrainiert wird und wenn nicht.

### Zweischichtig

Wird das kNN um eine Schicht erweitert, wird das Resultat signifikant verbessert. Auch kann in der Abbildung \ref{fig:dA-vs-sdA} abgelesen werden, dass voraustrainierte Autoencoder, bessere Resultate erzeugen als ein nicht voraustrainiertes tiefes kNN.

![Eischichtiger Denoising Autoencoder gegenüber einem Zweischichtigen Stacked Denoising Autoencoder, mit und ohne Voraustraining \label{fig:dA-vs-sdA}](images/dA_vs_sdA.png)

\FloatBarrier

### Dreischichtig

\FloatBarrier
### Vierschichtig

