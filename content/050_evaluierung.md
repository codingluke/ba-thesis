# Evaluierung \label{head:evaluierung}

In diesem Kapitel wird versucht mit den zuvor Implementierten Modulen eine Konfiguration zu finden, welche im *Kaggle-Wettbewerb* so gut wie möglich abschließt. Dafür werden verschiedene Architekturen einander gegenübergestellt und ebenfalls die Eigenschaften der einzelnen Hyperparameter erforscht und dokumentiert.

## Trainingsumgebung

Das Training wird auf dem, von der *Hochschule Für Technik und Wirtschaft Berlin (HTW Berlin)* zur Verfügung gestellten Server, *deepgreen02*, ausgeführt.

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

Die Trainingsumgebung ist wie in Abbildung \ref{fig:training-kontext} in Kapitel \ref{head:aufzeichung-training-prozess} aufgebaut. Die Analyse und Visualisierung der Trainingsläufe sind in Form von *ipython notebooks* auf dem lokalen Laptop umgesetzt [@ipython-notebook]. Diese befinden sich im Ordner */notebooks/*. Die Trainingsläufe selbst werden in *Python*-Dateien im Ordner */trainings/* aufgesetzt. Diese Dateien werden auf dem Server *deepgreen02* gestartet.

Der Server *deepgreen02* verfügt über zwei GPUs. *Theano* unterstützt grundsätzlich das Ausführen auf mehreren GPUs, dies geschieht jedoch nicht vollautomatisch. Das Synchronisieren muss selbst implementiert werden, weshalb nur eine *GPU* verwendet wird.

## Subbilder Generierung Benchmark \label{head:sliding_window}

Im Gesamten müssen für die Trainingsdaten laut der Berechnung \ref{eq:patches-calc} 28460160 Subbilder, wie im Kapitel \ref{head:vor-nachverarbeitung} beschrieben, generiert werden. Diese Subbilder haben je nach Größe der gewählten Überlappung eine unterschiedliche Bildgröße. Bei einer Überlappung von 2 Pixel, besitzen sie die Bildgröße (5x5) und bestehen somit aus 25 Pixeln.

\begin{equation} \label{eq:patches-calc}
  48 Bilder * 540p * 258p + 96 Bilder * 540p * 420p = 28460160 Subbilder
\end{equation}

In einem Benchmark wurden die drei implementierten Verfahren zur Subbildgenerierung einander gegenüber gestellt und die Resultate in der Tabelle \ref{table:sliding-window} aufgelistet. Hierbei ist ersichtlich, dass die nicht durchmischte Generierung anhand der *numpy*-Matrix-Index-Operationen, *Matrix-Op*, mit 62 Sekunden am schnellsten ist. Die auf doppelter Schleife basierte Variante *Schleifen*, geschweige denn deren Zufallsvariante, ist mit Abstand geschlagen.

| Verfahren          | Sekunden | Minuten | Maschine  |
|:-------------------|:--------:|:-------:|----------:|
| Schleifen          | 189      | 3,15    |deepgreen02|
| Schleifen, Zufall  | 319      | 5,2     |deepgreen02|
| Matrix-Op.         | 62       | 1       |deepgreen02|
| Matrix-Op, Zufall  | 173      | 2,8     |deepgreen02|
| Totaler-Zufall  | 100      | 1,6     |deepgreen02|

  : Vergleich der Verfahren zur Subbilderstellung. \label{table:sliding-window}

Wenn nun die Zufälligkeit mit einberechnet wird, wird die Stärke der Methode *Totalen-Zufall* sichtbar. Diese benötigt mit 100 Sekunden erheblich länger als die sortierte *Matrix-Op*, dafür müssen die Daten nicht zusätzlich durchmischt werden. Während den Tests wurde sichtbar, dass das Durchmischen mehr Zeit in Anspruch nimmt als die Generierung der Subbilder. Dadurch ist die Variante *Totaler-Zufall* die schnellste durchmischte Zufallsvariante.

![Vergleich: Batchweise oder Komplette Zufälligkeit \label{fig:batch_vs_fully}](images/fully_vs_batch_random.png)

Zu beachte ist: es wird in der Methode *Totaler-Zufall* nicht garantiert, dass alle vorhandenen Subbild berücksichtigt werden. Wiederum können andere Subbilder doppelt vorkommen. In Abbildung \ref{fig:batch_vs_fully} konnte aufgezeigt werden, das dies bei der vorhandenen Datengröße nicht ins Gewicht fällt, im Gegenteil, sogar einen regelmäßigeren Trainingsverlauf mit sich bringt, da die Durchmischung zufälliger über die gesamte Daten stattfindet.

Für weitere Trainingsdurchläufe, wird auf Grund dieser Gegenüberstellung, die Methode *Totalen-Zufall* verwendet.

## Einfaches kNN

### Größe des Minibatch

Die Größe des *Minibatch* definiert, wie viele Datensätze parallel trainiert werden. Diese Größe hat zwei wesentliche Einflüsse auf das Trainingsverhalten.

Als Vorteil benötigt das Trainieren einer Epoche bei einem größeren *Minibatch* weniger Zeit, da durch Vektorberechnungen parallel mehrere Datensätze direkt auf der *GPU* berechnet werden. Zusätzlich werden durch größere Gruppen weniger Kopiervorgänge zwischen dem *CPU* und *GPU* Arbeitsspeicher getätigt, was die Performance zusätzlich steigert.
Als Nachteil kann ein zu großen *Minibatch* den Trainingsfortschritt pro Epoche verringern, da nur am Ende jedes *Minibatch* Durchlauf die Gewichte angepasst werden. Dadurch werden die Gewichte bei einem kleineren *Minibatch* öfter angepasst.

![Trainingsverlauf  \label{fig:mb-vergleich}](images/training_minibatch_vergleich_2.png)

In Abblidung \ref{fig:mb-vergleich} ist zu sehen, wie sich der Trainingsfortschritt pro Epoche verringert, um so größer der *Minibatch* gewählt wird. Gleichzeitig verringert sich aber auch die Lerndauer. Um möglichst viele Trainingsvariationen prüfen zu können, wird eine *Minibatch*-Größe von $500$ gewählt. Das *kNN* mit dem besten Resultat im Kaggle-Wettbewerb, wird zusätzlich mit einer *Minibatch*-Größe von $50$ trainiert.

### Fixe gegenüber dynamischer Lernrate

In Abbildung \ref{fig:fix_dyn_eta} ist der Trainingsablauf einer fixen Lernrate mit einer dynamisch, abnehmender Lernrate gegenübergestellt. Dabei wird bei den Validierungskosten der Unterschied sichtbar.

Anfangs ist bei beiden Validierungskurven eine sprunghafte Verbesserung sichtbar. Ab der zweiten Epoche ist das Minimum bereits fast erreicht. Nun werden größere Sprünge bei die Kurve mit dem fixen Lernrate sichtbar, wobei die Kurve mit der dynamische Lernrate konstant, leicht abnimmt. Dies geschieht, da die dynamische Lernrate pro Epoche verkleinert wird, womit der Lernvorgang verfeinert wird.

![Unterschiedliche Trainingsverhalten bei fixer oder dynamischer Lernrate \label{fig:fix_dyn_eta}](images/lernrate_fix_vs_decrease.png)

Die Kostenfunktion der Validierungskosten ist in Abbildung \ref{fig:fix_dyn_eta} nicht der *RMSE*. Bis zu diesem Zeitpunkt der Arbeit, war diese leider falsch. Die Aussage der Kurve stimmt jedoch trotzdem.

\FloatBarrier

### RMSprop gegenüber Stochastik-Gradient-Descent und Momentum

In Abbildung \ref{fig:rmsprop_vs_sgd} ist ersichtlich, dass die beiden Gradientenabstiegsverfahren *RMSprop* und *Stochastik Gradient Descent*, *SGD*, sich in zwei Merkmalen unterscheiden. Zum Einen ist zu sehen, dass der *RMSprop* bereits am Anfang der ersten Epoche bessere Werte erzielt als der *SGD* am Ender der dritten. Die zweite Unterschied zeigt sich bei den Validierungskosten. Der *RMSprop* zeigt immer wieder kleine Ausschläge an, wobei der *SGD* kontinuierlich ab nimmt.

![Unterschiedliche Trainingsverhalten der beiden Algorithmen RMSprop und SGD+Momentum \label{fig:rmsprop_vs_sgd}](images/rmsprop_vs_sgd.png)

Da der *RMSprop* Algorithmus offensichtlich schneller lernt, basieren weitere Tests ausschließlich auf dem *RMSprop* Algorithmus. Hiermit konnte das Resultat der Abbildung \ref{rmsprop-compair} bestätigt werden.

\FloatBarrier

### Hyperparametersuche mit Spearmint

Mit Hilfe der *Python*-Bibliothek *Spearmint* wurde eine Hyperparametersuche auf folgende Parameter und deren Wertebereich ausgeführt. Beim Training wurde der kleine Datensatz verwendet um so viele Kombinationen wie mögliche testen zu können. Es wird angenommen, dass die Tendenzen der Hyperparameter ebenfalls auf der kleinen Datenbasis erkennbar sind, wenn auch das Resultat mit der größeren Datenbasis unterschiedlich ausfallen kann. Die besten Werte werden danach verwendet um weitere Tests auf der größeren Datenbasis durchzuführen und schlussendlich ein Modell mit allen Trainingsdaten für den Kaggle-Wettbewerb zu trainieren.

Parameter   Wertebereich        Bester Wert
--------    ---------------     -----------
dropout     [0.0 .. 0.01]       **0.0**
l2          [0.0 .. 0.02]       **0.0**
eta_start   [0.01 .. 0.5]       **0.045**
eta_end     [0.001 .. 0.01]     **0.01**
hidden      [10 .. 500]         **199**

  : Ergebnisse der Hyperparametersuche mit einem einschichtigen kNN \label{table:hyper-kNN}

### L2-Regularisation und Dropout

Die kleinste zum Trainieren verwendete Datenmenge mit jeweils 20 Trainings- und Validierungsbilder, ergeben $3486240$ Subbilder zum Trainieren als auch zum Validieren. Damit besteht eine genügend große Datenmenge zur Verfügung mit welcher *Overfitting* durch genügend Trainingsdaten entgegengewirkt werden soll. Wird die gesamte Trainingsmenge von 144 Bilder verwendet, wir dies noch verstärkt, da die Trainingsmenge um das 7-Fache ansteigt.

![Bei Verwendung von Dropout wird das Resultat schlechter \label{fig:comp_dropout}](images/comp_dropout.png)

Bestätigt wird diese Annahme durch Tests. In der Abbildung \ref{fig:comp_dropout} ist zu sehen, dass die Resultate sich mit dem Verwenden von Regularisierungs-Methoden verschlechtern.

![Bei Verwendung von L2-Regularisation wird das Resultat nicht besser \label{fig:comp-l2}](images/l2-comair.png)

Eine weitere Erklärung, wieso Regularisierung in diesem Fall zu einer Verschlechterung führt ist, dass die Trainings- und Validierungsdaten sich zu ähnlich sind. Dies wird bereits in Kapitel \ref{head:explorative-datenanalyse} erwähnt. In diesem Fall führt das *Overfitting* zu einem besseren Validierungsresultat.

Unterstrichen, wird dies ebenfalls von der Tatsache, dass nur bekannte Schriften bereinigt werden sollen. Dies führt automatisch zu einem *Overfitting* der existierende Schriften. Wo möglich würde *Dropout* und *L2-Regularisation* bei Validierungsdaten mit komplett verschiedener Hintergründe und Schriftarten, die Validierungskosten positiver beeinflussen.

### Einfluss der unsichtbaren Schicht

Während der heuristischen Hyperparametersuche ist aufgefallen, dass die unsichtbare Schicht, solange sie größer ist als die Eingangsschicht, keine signifikante Änderungen bewirkt. Dies ist in Abbildung \ref{fig:comp_hidden} an mehreren Beispielen dargestellt. Darin werden zwischen 21 bis 500 Neuronen in der unsichtbaren Schicht verwendet. Es ist auffällig, dass erst ab weniger als 100 Neuronen das Resultat negativ beeinflusst wird. Dies jedoch fast nicht spürbar. Erst wenn die unsichtbare Schicht kleiner als die Eingangsschicht (25 Neuronen) wird, wird das Lernverhalten spürbar negativ beeinflusst.

![Auswirkung der unsichbaren Schicht anhand der Anzahl Neuronen \label{fig:comp_hidden}](images/hidden_units.png)

\FloatBarrier

### Denoising-Autoencoder

Für den *Denoising-Autoencoder*, *dA*, wurde ebenfalls eine Hyperparametersuche mit *Spearmint* durchgeführt. Dabei wurde nach dem besten Verunreinigungswert und der Lernrate gesucht. Als beste Werte für den *Denoising-Autoencoder* stellten sich die Werte *0.15* für die Verunreinigung und *0.025* für die Lernrate heraus. Die Werte der Parameter dropout, L2 und Anzahl Neuronen in der unsichtbaren Schicht wurde vom normalen einschichtigen *kNN* übernommen.

![Eischichtiger dA gegenüber einschichtigem, normalen kNN (kleiner Datensatz) \label{fig:dA-vs-knn}](images/1-layer-comair.png)

Es hat sich herausgestellt, dass der *Denoising-Autoencoder* mit dem kleinen, sowie dem größeren Trainingsdatensatz besser als das normale *kNN* abgeschlossen hat, wobei der Unterschied im größeren Trainingsdatensatz erheblich größer ausfällt.

![Eischichtiger dA gegenüber einschichtigem normalem kNN (großer Datensatz) \label{fig:dA-vs-knn}](images/1-layer-comair-medium.png)

## Stacked-Denoising-Autoencoder, SdA

Bei *Stacked-Deonising-Autoencoder*, *SdA*, handelt es sich um ein *kNN* bestehend aus mehreren aufeinander folgenden Schichten der Klasse *AutoencoderLayer*. Es wird nun überprüft wie sich die Präzision pro neue Schicht verändert, wenn es im Voraus trainiert wird und wenn nicht.

Ab der dritten Schicht wird zusätzlich mit der Aktivierungsfunkion *Rectified-Linear-Unit*, *ReLU*, trainiert. Es wird angenommen, dass diese zu besseren Resultaten führen kann, da sie wie in Kapitel \ref{head:relu_act} erläutert wird, bei tiefen Netzen viele Vorteile mit sich bringt.

### Zweischichtig

Die Größe der zweiten Schicht, wurde abermals mit Hilfe von *Spearmint* gesucht. Dabei hat sich *81* als bestes Resultat ergeben. Das Netz besteht somit aus der Konfiguration *dA(25,199)-dA(199,81)-fc(81,1)*. Dabei werden die beiden *dA*s mit einer Verrauschung von $0.14$ im voraus trainiert. Alle anderen Hyperparameter werden von den vorhergehenden Konfigurationen übernommen.

Wird das *kNN* um eine Schicht erweitert, wird das Resultat signifikant verbessert. Auch kann in Abbildung \ref{fig:dA-vs-sdA} abgelesen werden, dass im Voraus trainierter *SdA* bessere Resultate erzeugen als ein nicht im Voraus trainiertes *MLP*.

![Eischichtiger dA gegenüber einem zweischichtigen SdA mit und ohne Voraustraining (kleiner Datensatz) \label{fig:dA-vs-sdA}](images/2-layer-compair.png)

Der Trainingsverlauf durch die größere Trainingsmenge erzielt eins sehr ähnliches Bild (siehe Abbildung \ref{fig:dA-vs-sdA-medium}). Auch hier ist der im voraus Trainierte *SdA* dem normalen *MLP* überlegen. Weit abgeschlagen folgt der Einschichtige *dA*.

![Eischichtiger dA gegenüber einem zweischichtigen SdA mit und ohne Voraustraining (große Trainingsmenge) \label{fig:dA-vs-sdA-medium}](images/2-layer-compair-medium.png)

\FloatBarrier

### Dreischichtig

Für die dritte Schicht wurde abermals mit Hilfe von *Spearmint* die beste Kombination gesucht. Daraus erfolgte ein Netz mit der Konfiguration *da(25,199)-dA(199,81)-dA(81,70)-fc(81,1)*. Der Verrauschungs-Koeffizient wurde auch hier bei *0.14* belassen.

Der dreischichtige *SdA*, trainiert mit der Aktivierungsfunktion *Sigmoid*, erreicht auf der kleinen Datenbasis mit einem, *Root-Mean-Squared-Error*, *RMSE*, von $0.01369$ ein leicht besseres Resultat als der Zweischichtige. Die Verbesserung ist jedoch weit entfernt vom Sprung, welcher beim Schritt vom einschichtigen zum zweischichtigen *SdA* statt gefunden hat.

![Dreischichtige MLP mit verschiedenen Aktivierungsfunktionen und Voraustraining (kleine Trainingsmenge) \label{fig:3-layer-compair}](images/3-layer-compair.png)

Des Weiteren wurde erforscht, wie sich der dreischichtige *SdA* gegenüber einem normalen, nicht im voraus trainierten, dreischichtigen Netzwerk schlägt. Hier ist ablesbar, dass der *SdA* ab der vierten Epoche bessere Validierungswerte erzielt als das normale *MLP*. Noch deutlicher sichtbar, kann es an den Trainingskosten abgelesen werden.

Werden die beiden Tests mit der Aktivierungsfunktion *ReLU* wiederholt, werden keine bessere Werte erzielt. Im Gegenteil, sowohl der *SdA*, als auch das normale *MLP* schließen mit *ReLU*-Neuronen schlechter ab als mit *Simgmoid*-Neuronen. Interessant ist ebenfalls, dass der *ReLU-SdA* zu schlechteren Ergebnissen führt als das normal *ReLU-MLP*. Das normale *ReLU-MLP* ist sogar fast so gut wie der *Simgoid-SdA*. Des Weiteren ist abzulesen, dass der Validierungsverlauf mit *ReLU*-Neuronen viel Sprunghafter ist, als mit *Sigmoid*-Neuronen.

![Dreischichtiges MLP mit verschiedenen Aktivierungsfunktionen und Voraustraining (große Trainingsmenge) \label{fig:3-layer-compair-medium}](images/3-layer-comair-medium.png)

Werden die Netze mit der großen Datenmenge trainiert kommt es zu Überraschungen. Hier erzielen die mit *ReLU*-Neuronen besetzten *MLP* besser ab. Allen voran das normale, nicht im Voraus Trainierte *ReLU-MLP*. Bei den Trainingskosten zeigt der *Sigmoid-SdA* abermals seine Stärken. Das normale *Simgoid-MLP* ist sichtbar das schwächste.

\FloatBarrier

### Vierschichtig

Beim vierschichtigen *SdA* ist bei der kleinen Datengröße die Streuung der Konfigurationen sehr klein. Mit einem *RMSE* von $0.0141$ hat sich das beste Resultat vom normalen *ReLU-MLP*, gegenüber dem dreischichtigen *SdA* verschlechtert. Eine regelmäßigkeit lässt sich in den Trainingskosten erkennen. Hier schliessen der *Sigmoid-SdA* und das normale *ReLU-MLP* abermals am besten ab.

![4-Schichtige MLP mit verschiedenen Aktivierungsfunktionen und Voraustraining (kleine Trainingsmenge) \label{fig:4-layer-compair}](images/4-layer-compair.png)

Auch bestätigt sich, dass mit der großen Datenmenge die Unterschiede größer ausfallen und womöglich repräsentativer sind. Interessanterweise schließt bei der großen Datenmenge das normale *Sigmoid-MLP* am besten ab. Klarer Verlierer hingegen ist der *ReLU-SdA*. Mit *0.0122* gegenüber dem Bestwert von *0.0105* der normalen dreischichtigen *ReLU-MLP* ist das vierschichtige *MLP* dem dreischichtigen auch in der großen Datenmenge unterlegen.

![ 4-Schichtige MLP mit verschiedenen Aktivierungsfunktionen und Voraustraining (große Trainingsmenge) \label{fig:4-layer-compair-medium}](images/4-layer-comair-medium.png)

\FloatBarrier

## MLP mit der Aktivierungsfunkion ReLU \label{head:relu_act}

Auf das in der Abbildungen \ref{fig:3-layer-compair} auftegretene Phänomen, dass das nicht im voraus trainierte, normale *MLP* mit *ReLU*-Neuronen, fast gleich gut abschneidet, als die im voraus trainierten *SdA* mit *Sigmoid*-Neuronen, werden von der Arbeit "On rectified linear units for speech processing" bestätigt. Darin steht zu Lesen:

>"...and the units that compose the HDNN “rectified linear units” (ReLUs) [7]. This small change brings several advantages. First, in our experience it eliminates the necessity to have a “pretraining” phase using unsupervised learning [8].  We demonstrate empirically that we can easily and successfully train extremely deep networks even from random initialization.  Second, the convergence of HDNN is faster than in a regular logistic neural net with the same topology.  Third, HDNN is very simple to  optimize.   Even  vanilla  stochastic  gradient  descent  with constant  learning  rate  yields  very  good  accuracy.    Fourth, HDNN generalizes better than its logistic counterpart.  And finally,  rectified  linear  units  are  faster  to  compute  because they do not require exponentiation and division, with an over- all speed up of 25% on the 4 hidden layer neural network we tested on." [@Zeiler_onrectified, Seite 1]

Dies erklärt jedoch nicht das gute Resultat des normalen *Simgoid-MLP* in Abbildung \ref{fig:4-layer-compair-medium}. Eine mögliche Erklärung dafür ist, dass sich das Phänomen vom *Gradientenschwund* erst ab der vierten Schicht bemerkbar macht. Leider wurde aber genau mit vier Schichten keine signifikant besseren Werte durch *ReLU*-Neuronen erreicht.

Das schlechte Abschneiden des *SdA* mit *ReLU*-Neuronen, wird unteranderem darin vermutet, da das vorausgehende Training mit *Sigmoid*-Neuronen stattfindet. Somit ist nicht Auszuschließen, dass die trainierten Initialwerte nicht optimal für *ReLU*-Neuronen sind.

## Kaggle Resultate

Um die *kNN*-Konfigurationen auf Kaggle zu bewerten, wurden die Netze in gleicher Konfiguration wie in den vorhergehenden Kapitel mit allen verfügbaren Trainingsdaten für 15 Epochen Trainiert. Das vorausgehende Training der *SdA*s wurde mit 10 Epochen durchgeführt.

Auf Kaggle hat überraschenderweise das normale dreischichtige *MLP* mit *Sigmoid*-Neuronen, welches nicht durch *AutoencoderLayer* im voraus trainiert wurde, mit dem 11. Platz und einem *RMSE* von **0.01894** am Besten abgeschnitten. Bei allen anderen Schichten, hat sich, wie erwartet, der *SdA* gegenüber dem normalen *MLP* durchgesetzt.

Auf den eigenen Validierungsdaten, hat die Gewinner-Konfiguration nicht all zu gut Abgeschnitten. Dies lässt darauf schließen, dass die eigenen Validierungsdaten den Trainingsdaten zu ähnlich sind, und somit *Overfitting* als gut bewerten. Sehr überraschend ist ebenfalls das schlechte Abschneiden der *ReLU*-Neuronen, da diese in eigenen Tests häufig sehr gut abgeschlossen haben und, wie in Kapitel \ref{head:relu_act} beschrieben, auch in anderen Arbeiten zu Spitzenresultate führten.

Was bei den Lokalen- sowie den Kaggle-Tests gleich ausfällt, sind:

- Der *SdA* mit *ReLU*-Neuronen schließt immer sehr schlecht ab
- 3 Schichten sind besser als 4 Schichten
- Der signifikant Sprung liegt zwischen dem einschichtigen und zweischichtigen *MLP*

------------------------------------------------------------------
Architektur                 RMSE          Platz   Trainierdauer
-------------------------   -----------   -----   ----------------
3 Schichtig Sigmoid         **0.01894**   11      267 Minuten

3 Schichtig Sigmoid SdA     0.01934       12      314 Minuten

4 Schichtig Sigmoid SdA     0.01943       12      399 Minuten

4 Schichtig Sigmoid         0.01999       13      314 Minuten

3 Schichtig ReLU            0.02007       13      **258 Minuten**

2 Schichtig Sigmoid SdA     0.02038       14      490 Minuten

2 Schichtig Sigmoid         0.02113       17      484 Minuten

4 Schichtig ReLU            0.02115       17      335 Minuten

3 Schichtig ReLU SdA        0.02218       17      319 Minuten

4 Schichtig ReLU SdA        0.02567       24      398 Minuten

1 Schicht Sigmoid dA        0.02890       51      460 Minuten

1 Schicht Sigmoid           0.02973       52      300 Minuten
-----------------------------------------------------------------

  : Kaggle Resultate

### Minibatch-Größe

Das beste Modell, das dreischichtige, normale *MLP* mit *Sigmoid*-Neuronen, wurde nachträglich mit einer *Minibatch*-Größe von 50 anstatt den 500 Trainiert. Das Training dauerte dadurch 15.5 Stunden anstatt 4.5 Stunden, also mehr als drei mal so lange. Ausgewertet auf Kaggle erreichte dieses *MLP* einen *RMSE*-Score von **0.01885**. Es schnitt somit minimal besser ab. Die Trainingslänge ist jedoch erheblich höher. Damit konnte bestätigt werden, dass bei der Hyperparametersuche einen möglichst großen *Minibatch* verwendet werden sollte, um schneller Resultate zu erzielen.

### Trainingszeit

Da der Server *deepgreen02* von mehreren Benutzer gleichzeitig benutzt werden kann, sind massive Performance-Schwankungen festgestellt worden. So wurden die Netze für das zweischichtige und einschichtige *kNN* an einem Tag durchgeführt an dem ein anderer *Python*-Prozess die *CPU* fast 100% ausgelastet hat. Dadurch kommen die enormen Trainingszeiten zu Stande. In vorhergehenden Testläufe, waren das einschichtige und zweischichtige *MLP* einiges schneller.

### Trainings und Validationdaten Überdenken

Das inkonsistente Abschneiden der Netze bezüglich der Kaggle-Testdaten gegenüber den eigenen Validierungsdaten, lässt darauf schließen, dass die Validierungsdaten zum Einen zu spezifisch und zum Anderen zu ähnlich der Trainingsdaten sind. Es wurde beim Aufteilen der von Kaggle zur Verfügung gestellten Trainingsdaten in eigene Trainings- und Validationsdaten bewusst darauf geachtet, dass sich die Hintergründe und Texte unterscheiden. Die einzigen Merkmale welche in den Trainings- sowie den Validationsdaten gleich sind, sind die Schriftarten. Es könnte nun ein Validierungsdatensatz erstellt werden, welcher komplett andere Schriften enthält. Dies würde ein generelleres Validieren der Schriften ermöglichen.

Zusätzlich sollten neue, diverse Hintergründe den Trainings- sowie den Validierungsdaten hinzugefügt werden. Die vorhandenen Hintergründe sind sich tatsächlich sehr ähnlich. Hier könnte in Zukunft eine Methode für automatische Testdatengenerierung durch zufälliges generieren verunreinigten Bilder beihilfe schaffen.

### Trainingsmenge

Die Trainingsmenge hat ebenfalls Einfluss auf das Training. Nur schon die Tatsache, dass wenn die gleiche Anzahl Epochen trainiert wird, eine Epoche bei allen vorhandenen Trainingsdaten doppelt so viele Iterationen, wie bei der eigenen Trainingsdatengröße anfallen. Ebenfalls hat die Trainingsmenge Einfluss auf die Komplexität eines *kNN*. Je mehr Daten zur Verfügung stehen, desto komplexer kann ein *kNN* ohne dass *Overfitting* auftritt.

Interessanterweise, haben die Tests mit der kleinen Trainings- und Validierungsmenge gegenüber der Größeren zu ähnlichen Resultaten geführt, die Resultate der gesamten Trainingsmenge auf die Kaggle Testdaten, jedoch zu überraschend anderen.

### Größte Schwierigkeiten

Die größten Schwierigkeiten bereiten Flecken, welche den Kontrast zur Schrift extrem verringern. In den Kaggle Testdaten befindet sich ein Hintergrund, welcher exakt dies erfüllt (siehe Abbildung \ref{fig:schicht-vergleich-01}. In den Trainingsdaten befindet sich jedoch nur einen Hintergrund, welche dem nahe kommt. Dieser besitzt dennoch über einen höheren Kontrast. Somit fehlen ebenfalls Trainingsdaten um diesen Fall besser abzudecken.

### Detailanalyse ausgewählter Bilder

Für diese Analyse wurde in den Testdaten nach exemplarischen Ausschnitten gesucht, an welchen der Unterschied der verschiedenenschichtigen *MLP* aufgezeigt werden kann. Hier werden immer die normalen *Sigmoid-MLP* welche für den Kaggle-Wettbewerb mit allen Trainingsdaten trainiert wurden verwendet. Die Schichten sind in aufsteigender Reihenfolge von links nach rechts dargestellt.

![01.png: Bereinigt durch 1-4 schichtiges kNN von links nach rechts aufsteigend \label{fig:schicht-vergleich-01}](images/bild1-schicht-img-compair.png)

In Abbildung \ref{fig:schicht-vergleich-01} ist ein Bild mit dem kompliziertesten Hintergrund sichtbar. Das Wort *important* kann selbst von Auge nicht mehr gut entziffert werden. Hier wird sichtbar, dass selbst das einschichtige *kNN* bereits ein gutes Resultat erzielt. Das zweischichtige *MLP* kann vor allem die Flecken noch sichtbar besser entfernen. Das dreischichtige bereinigte noch mehr der Freifläche, besonders sichtbar ist das Verschwinden vom Randbereich des Flecks. Beim vierschichtigen *MLP* ist dann wieder eine Verschlechterung sichtbar.

![19.png: Bereinigt druch 1-4 schichtiges kNN von links nach rechts aufsteigend \label{fig:schicht-vergleich-19}](images/bild19-schicht-img-compair.png)

Abbildung \ref{fig:schicht-vergleich-19} zeigt das Bild *19.png*. Es besitzt einen rauen, gefleckten Hintergrund. Hier ist vor allem sichtbar, wie die Kontur der Schrift sich bis zur dritten Schicht verbessert. Auch die Freiflächen werden ab der zweiten Schicht annähernd perfekt bereinigt.

\FloatBarrier

Im Bild \ref{fig:schicht-vergleich-22} besteht der Hintergrund aus einem zerknitterten Papier. Die Schrift, ist offensichtlich nicht verzerrt, somit wurde der Hintergrund im Nachhinein hinzugefügt. An diesem Hintergrund ist sind vor allem an den Stellen der Wörter *corpus*, *gh Spanish* und *widesp* kompliziert. Hier ist sichtbar wie die Bereinigung an den Leerstellen sichtbar bis zur dritten Schicht verbessert wird. Was jedoch nicht funktioniert, ist das füllen von Lücken in den Texten. Dies ist beim Wort *corpus* sehr schön sichtbar.

![22.png: Bereinigt durch 1-4 schichtiges kNN von links nach rechts aufsteigend \label{fig:schicht-vergleich-22}](images/bild22-schicht-img-compair.png)

Das Bild *88.png* besitzt den einfachsten Hintergund. Dieser wurde bereits vom einschichtigen *kNN* sehr gut bereinigt. Einzig an der Stelle des Buchstabens *i* im Wort *important* kann eine Verbesserung erreicht werden. Ebenfalls ist eine leichte Verbesserung der Konturen ab der zweiten Schicht sichtbar.

![88.png: Bereinigt durch 1-4 schichtiges kNN von links nach rechts aufsteigend \label{fig:schicht-vergleich-88}](images/bild88-schicht-img-compair.eps)


\FloatBarrier

### 11. Platz auf Kaggle

Trotz der inkonsistenten Resultate, ist der 11 Platz auf Kaggle mit einem von Grund auf selbst geschrieben *kNN* ein beachtliches Resultat. Andere Mitstreiter, welche noch bessere Resultate erreichten, taten dies unter anderem durch bewusstes *Overfitting* der Kaggle-Testdaten. Dies kann durch das Extrahieren und miteinbeziehen der Hintergründe im Training erreicht werden. Auch verwenden viele Vorvearbeitungsschritte wie automatisierter Kontrastausgleich. Diese Informationen können im Kaggle-Forum zum Wettbewerb [@kaggleDDD] entnommen werden.

