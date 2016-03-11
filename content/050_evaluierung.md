# Evaluierung \label{head:evaluierung}

In diesem Kapitel wird versucht mit den zuvor implementierten Modulen eine Konfiguration zu finden, welche im Wettbewerb [@kaggleDDD] so gut wie möglich abschließt. Dafür werden verschiedene Architekturen einander gegenübergestellt und ebenfalls die Eigenschaften der einzelnen Hyperparameter erforscht und dokumentiert.

## Trainingsumgebung

Das Training wird auf dem von der *Hochschule Für Technik und Wirtschaft Berlin (HTW Berlin)* zur Verfügung gestellten Server, *deepgreen02*, ausgeführt.

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

Der Server *deepgreen02* verfügt über zwei GPUs. *Theano* unterstützt grundsätzlich das Ausführen auf mehreren GPUs, dies geschieht allerdings nicht vollautomatisch. Das Synchronisieren muss selbst implementiert werden, weshalb nur eine *GPU* verwendet wird.

## Subbilder Generierung Benchmark \label{head:sliding_window}

Im Gesamten müssen für die Trainingsdaten laut der Berechnung \ref{eq:patches-calc} 28460160 Subbilder, wie im Kapitel \ref{head:bereinigungsprozess} beschrieben, generiert werden. Diese Subbilder haben je nach Größe der gewählten Überlappung eine unterschiedliche Bildgröße. Bei einer Überlappung von 2 Pixel, besitzen sie die Bildgröße (5x5) und bestehen somit aus 25 Pixeln.

\begin{equation} \label{eq:patches-calc}
  48 Bilder * 540p * 258p + 96 Bilder * 540p * 420p = 28460160 Subbilder
\end{equation}

In einem Benchmark wurden die drei implementierten Verfahren zur Subbildgenerierung einander gegenüber gestellt und die Resultate in der Tabelle \ref{table:sliding-window} aufgelistet. Hierbei ist ersichtlich, dass die nicht durchmischte Generierung anhand der *numpy*-Matrix-Index-Operationen, *Matrix-Op*, mit 62 Sekunden am schnellsten ist. Die auf doppelter Schleife basierte Variante *Schleifen*, geschweige denn deren Zufallsvariante, ist mit großem Abstand geschlagen.

| Verfahren          | Sekunden | Minuten | Maschine  |
|:-------------------|:--------:|:-------:|----------:|
| Schleifen          | 189      | 3,15    |deepgreen02|
| Schleifen, Zufall  | 319      | 5,2     |deepgreen02|
| Matrix-Op.         | 62       | 1       |deepgreen02|
| Matrix-Op, Zufall  | 173      | 2,8     |deepgreen02|
| Totaler-Zufall  | 100      | 1,6     |deepgreen02|

  : Vergleich der Verfahren zur Subbilderstellung. \label{table:sliding-window}

Wenn nun die Zufälligkeit mit einberechnet wird, wird die Stärke der Methode *Totaler-Zufall* sichtbar. Diese benötigt mit 100 Sekunden erheblich länger als die sortierte *Matrix-Op*, dafür müssen die Daten nicht zusätzlich durchmischt werden. Während den Tests wurde sichtbar, dass das Durchmischen mehr Zeit in Anspruch nimmt als die Generierung der Subbilder. Dadurch ist die Variante *Totaler-Zufall* die schnellste Zufallsvariante.

![Vergleich: Batchweise Durchmischung gegen Totaler Zufall (Kleine-Datenbasis) [@hodel] \label{fig:batch_vs_fully}](images/fully_vs_batch_random.png)

Zu beachten ist: es wird in der Methode *Totaler-Zufall* nicht garantiert, dass alle vorhandenen Subbilder berücksichtigt werden. Wiederum können andere Subbilder doppelt vorkommen. In Abbildung \ref{fig:batch_vs_fully} konnte aufgezeigt werden, dass dies bei der vorhandenen Datengröße nicht ins Gewicht fällt, im Gegenteil, sogar einen regelmäßigeren Trainingsverlauf mit sich bringt, da die Durchmischung zufälliger innerhalb der gesamten Daten stattfindet. Für weitere Trainingsdurchläufe wird auf Grund dieser Gegenüberstellung die Methode *Totaler-Zufall* verwendet.

## Einfaches kNN

### Größe des Minibatch

Die Größe des *Minibatch* definiert, wie viele Datensätze parallel trainiert werden. Diese Größe hat zwei wesentliche Einflüsse auf das Trainingsverhalten.

Als Vorteil benötigt das Trainieren einer Epoche bei einem größeren *Minibatch* weniger Zeit, da durch Vektorberechnungen parallel mehrere Datensätze direkt auf der *GPU* berechnet werden. Zusätzlich werden durch größere Gruppen weniger Kopiervorgänge zwischen dem *CPU* und *GPU* Arbeitsspeicher getätigt, was die Performance zusätzlich steigert.

Als Nachteil kann ein zu großer *Minibatch* den Trainingsfortschritt pro Epoche verringern, da nur am Ende jedes *Minibatch*-Durchlaufs die Gewichte angepasst werden. Dadurch werden die Gewichte bei einem kleineren *Minibatch* öfter angepasst.

![Trainingsverlauf mit verschiedenen Minibatchgrößen (Kleine-Datenbasis) [@hodel] \label{fig:mb-vergleich}](images/training_minibatch_vergleich_2.png)

In Abbildung \ref{fig:mb-vergleich} ist zu sehen, wie sich der Trainingsfortschritt pro Epoche verringert, um so größer der *Minibatch* gewählt wird. Gleichzeitig verringert sich aber auch die Lerndauer. Um möglichst viele Trainingsvariationen prüfen zu können, wird eine *Minibatchgröße* von $500$ gewählt. Das *kNN* mit dem besten Resultat im Wettbewerb wird zusätzlich mit einer *Minibatchgröße* von $50$ trainiert.

### Fixe gegenüber dynamischer Lernrate

In Abbildung \ref{fig:fix_dyn_eta} ist der Trainingsablauf einer fixen Lernrate, einer dynamisch, abnehmenden Lernrate gegenübergestellt. Dabei wird bei den Validierungskosten der Unterschied sichtbar.

Anfangs ist bei beiden Validierungskurven eine sprunghafte Verbesserung sichtbar. Ab der zweiten Epoche ist das Minimum bereits fast erreicht. Nun werden größere Sprünge bei der Kurve mit der fixen Lernrate sichtbar, wobei die Kurve mit der dynamischen Lernrate konstant, leicht abnimmt. Dies geschieht, da die dynamische Lernrate pro Epoche verkleinert wird, womit wiederum der Lernvorgang verfeinert wird.

![Unterschiedliche Trainingsverhalten bei fixer oder dynamischer Lernrate (Kleine-Datenbasis) [@hodel] \label{fig:fix_dyn_eta}](images/lernrate_fix_vs_decrease.png)

Die Kostenfunktion der Validierungskosten ist in Abbildung \ref{fig:fix_dyn_eta} nicht der *RMSE*. Bis zu diesem Zeitpunkt der Arbeit war diese leider falsch. Die Aussage der Kurve stimmt dennoch weiterhin.

\FloatBarrier

### RMSprop gegenüber Stochastik-Gradient-Descent und Momentum

In Abbildung \ref{fig:rmsprop_vs_sgd} ist ersichtlich, dass die beiden Gradientenabstiegsverfahren *RMSprop* und *Stochastik-Gradient-Descent*, *SGD*, sich in zwei Merkmalen unterscheiden. Als Erstes ist zu sehen, dass der *RMSprop* bereits am Anfang der ersten Epoche bessere Werte erzielt als der *SGD* am Ende der dritten. Der zweite Unterschied zeigt sich bei den Validierungskosten. Der *RMSprop* zeigt immer wieder kleine Ausschläge an, wobei der *SGD* kontinuierlich abnimmt.

![Unterschiedliche Trainingsverhalten der beiden Algorithmen RMSprop und SGD+Momentum (Kleine-Datenbasis) [@hodel] \label{fig:rmsprop_vs_sgd}](images/rmsprop_vs_sgd.png)

Da der *RMSprop* Algorithmus offensichtlich schneller ler-t, basieren weitere Tests ausschließlich auf dem *RMSprop* Algorithmus. Hiermit konnte das Resultat der Abbildung \ref{rmsprop-compair} bestätigt werden.

\FloatBarrier

### Hyperparametersuche mit Spearmint

Mit Hilfe der *Python*-Bibliothek *Spearmint* wurde eine Hyperparametersuche auf den in Tabelle \ref{table:hyper-kNN} angegebenen Parametern und deren Wertebereichen ausgeführt. Beim Training wurde die *Kleine-Datenbasis* verwendet um so viele Kombinationen wie mögliche testen zu können. Es wird angenommen, dass die Tendenzen der Hyperparameter auch auf der kleinen Datenbasis erkennbar sind, wenn auch das Resultat mit der größeren Datenbasis unterschiedlich ausfallen kann. Die besten Werte werden danach verwendet um weitere Tests mit der größeren Datenbasis durchzuführen und abschließend ein Modell mit allen Trainingsdaten für den Wettbewerb zu trainieren.

Parameter   Wertebereich        Bester Wert
--------    ---------------     -----------
dropout     [0.0 .. 0.01]       **0.0**
l2          [0.0 .. 0.02]       **0.0**
eta_start   [0.01 .. 0.5]       **0.045**
eta_end     [0.001 .. 0.01]     **0.01**
hidden      [10 .. 500]         **199**

  : Ergebnisse der Hyperparametersuche mit einem einschichtigen kNN \label{table:hyper-kNN}

### L2-Regularisation und Dropout

Die *Kleine-Datenbasis*, mit jeweils 20 Trainings- und Validierungsbildern, ergibt immer noch $3486240$ Subbilder zum Trainieren als auch zum Validieren. Damit steht eine ausreichend große Datenmenge zur Verfügung, mit welcher *Overfitting* durch genügend Trainingsdaten entgegengewirkt werden soll. Wird die *Große-Datenbasis* mit 72 Bilder verwendet, wird dies noch verstärkt, da die Trainingsmenge um mehr als das das Dreifache ansteigt.

![Bei Verwendung von Dropout wird das Resultat schlechter (Kleine-Datenbasis) [@hodel]) \label{fig:comp-dropout}](images/comp_dropout.png)

Bestätigt wird diese Annahme durch in den Abbildungen \ref{fig:comp-dropout} und \ref{fig:comp-l2} dargestellten Tests. In der Abbildung \ref{fig:comp-dropout} ist zu sehen, dass die Resultate sich mit dem Verwenden von *Dropout* verschlechtern. Auch die *L2-Regularisation* in Abbildung \ref{fig:comp-l2} führt nicht zu der besagten Verbesserung.

![Bei Verwendung von L2-Regularisation wird das Resultat nicht besser (Kleine-Datenbasis) [@hodel] \label{fig:comp-l2}](images/l2-comair.png)

Eine Erklärung, wieso Regularisierung in diesem Fall zu einer Verschlechterung führt ist, dass die Trainings- und Validierungsdaten sich zu ähnlich sind. Dies wird bereits in Kapitel \ref{head:explorative-datenanalyse} erwähnt. In diesem Fall führt das *Overfitting* zu einem besseren Validierungsresultat.

Unterstrichen wird dies ebenfalls von der Tatsache, dass nur bekannte Schriften bereinigt werden sollen. Dies führt automatisch zu einem *Overfitting* der existierenden Schriften. Möglicherweise würden *Dropout* und *L2-Regularisation* bei Validierungsdaten mit komplett verschiedenen Hintergründen und Schriftarten, die Validierungskosten positiver beeinflussen.

### Einfluss der unsichtbaren Schicht

Während der heuristischen Hyperparametersuche ist aufgefallen, dass die unsichtbare Schicht, solange sie größer als die Eingangsschicht ist, keine signifikante Änderungen bewirkt. Dies ist in Abbildung \ref{fig:comp_hidden} an mehreren Beispielen dargestellt. Darin werden zwischen 21 bis 500 Neuronen in der unsichtbaren Schicht verwendet. Es ist auffällig, dass erst ab weniger als 100 Neuronen das Resultat negativ beeinflusst wird. Dieser ist allerdings fast nicht spürbar. Erst wenn die unsichtbare Schicht kleiner als die Eingangsschicht (25 Neuronen) wird, wird das Lernverhalten spürbar negativ beeinflusst.

![Auswirkung der unsichbaren Schicht anhand der Anzahl Neuronen (Kleine-Datenbasis) [@hodel] \label{fig:comp_hidden}](images/hidden_units.png)

\FloatBarrier

### Denoising-Autoencoder

Für den *Denoising-Autoencoder*, *dA*, wurde ebenfalls eine Hyperparametersuche mit *Spearmint* durchgeführt. Dabei wurde nach dem besten Verunreinigungswert und der Lernrate gesucht. Als beste Werte für den *Denoising-Autoencoder* stellten sich die Werte *0.15* für die Verunreinigung und *0.025* für die Lernrate heraus. Die Werte der Parameter dropout, L2 und Anzahl Neuronen in der unsichtbaren Schicht wurden vom normalen einschichtigen *kNN* übernommen.

![Eischichtiger dA gegenüber einschichtigem, normalen kNN (Kleine-Datenbasis) [@hodel] \label{fig:dA-vs-knn}](images/1-layer-comair.png)

Es hat sich herausgestellt, dass der *Denoising-Autoencoder* mit dem kleinen, sowie dem größeren Trainingsdatensatz besser als das normale *kNN* abgeschlossen hat, wobei der Unterschied im größeren Trainingsdatensatz erheblich größer ausfällt.

![Eischichtiger dA gegenüber einschichtigem normalem kNN (Große-Datenbasis) [@hodel] \label{fig:dA-vs-knn}](images/1-layer-comair-medium.png)

## Stacked-denoising-Autoencoder, SdA

Bei *Stacked-denoising-Autoencoder*, *SdA*, handelt es sich um ein *kNN* bestehend aus mehreren aufeinander folgenden Schichten der Klasse *AutoencoderLayer*. Es wird nun überprüft, wie sich die Präzision pro neuer Schicht verändert, wenn es im Voraus trainiert wird und wenn nicht.

Ab der dritten Schicht wird zusätzlich mit der Aktivierungsfunkion *Rectified-Linear-Unit*, *ReLU*, trainiert. Es wird angenommen, dass diese zu besseren Resultaten führen kann, da sie, wie in Kapitel \ref{head:relu_act} erläutert, bei tiefen Netzen viele Vorteile mit sich bringt.

### Zweischichtig

Die Größe der zweiten Schicht wurde abermals mit Hilfe von *Spearmint* ermittelt. Dabei hat sich *81* als bestes Resultat ergeben. Das Netz besteht somit aus der Konfiguration *dA(25,199)-dA(199,81)-fc(81,1)*. Dabei werden die beiden *dA*s mit einer Verrauschung von $0.14$ im Voraus trainiert. Alle anderen Hyperparameter werden von den vorhergehenden Konfigurationen übernommen.

Wird das *kNN* um eine Schicht erweitert, wird das Resultat signifikant verbessert. Auch kann in Abbildung \ref{fig:dA-vs-sdA} abgelesen werden, dass im Voraus trainierter *SdA* bessere Resultate erzeugen, als ein nicht im Voraus trainiertes *MLP*.

![Eischichtiger dA gegenüber einem zweischichtigen SdA mit und ohne Voraustraining (Kleine-Datenbasis) [@hodel] \label{fig:dA-vs-sdA}](images/2-layer-compair.png)

Der Trainingsverlauf durch die größere Trainingsmenge erzielt ein ähnliches Bild (siehe Abbildung \ref{fig:dA-vs-sdA-medium}). Auch hier ist der im Voraus trainierte *SdA* dem normalen *MLP* überlegen. Weit abgeschlagen folgt der einschichtige *dA*.

![Eischichtiger dA gegenüber einem zweischichtigen SdA mit und ohne Voraustraining (Große-Datenbasis) [@hodel] \label{fig:dA-vs-sdA-medium}](images/2-layer-compair-medium.png)

\FloatBarrier

### Dreischichtig

Für die dritte Schicht wurde abermals mit Hilfe von *Spearmint* die beste Kombination gesucht. Daraus erfolgte ein Netz mit der Konfiguration *da(25,199)-dA(199,81)-dA(81,70)-fc(70,1)*. Der Verrauschungs-Koeffizient wurde auch hier bei *0.14* belassen.

Der dreischichtige *SdA*, trainiert mit der Aktivierungsfunktion *Sigmoid*, erreicht auf der kleinen Datenbasis* mit einem *Root-Mean-Squared-Error*, *RMSE*, von $0.01369$ ein leicht besseres Resultat als der zweischichtige. Die Verbesserung ist dennoch weit entfernt vom Sprung, welcher beim Schritt vom einschichtigen zum zweischichtigen *SdA* stattgefunden hat.

![Dreischichtige MLP mit verschiedenen Aktivierungsfunktionen und Voraustraining (Kleine-Datenbasis) [@hodel] \label{fig:3-layer-compair}](images/3-layer-compair.png)

Des Weiteren wurde erforscht, wie sich der dreischichtige *SdA* gegenüber einem normalen, nicht im Voraus trainierten, dreischichtigen Netzwerk schlägt. Hier ist ablesbar, dass der *SdA* ab der vierten Epoche bessere Validierungswerte erzielt als das normale *MLP*. Deutlicher sichtbarer, kann ebenfalls an den Trainingskosten abgelesen werden, dass der *SdA* gegenüber dem *MLP* schneller trainiert.

Werden die beiden Tests mit der Aktivierungsfunktion *ReLU* wiederholt, werden keine besseren Werte erzielt. Im Gegenteil, sowohl der *SdA*, als auch das normale *MLP* schließen mit *ReLU*-Neuronen schlechter ab als mit *Simgmoid*-Neuronen. Interessant ist ebenfalls, dass der *ReLU-SdA* zu schlechteren Ergebnissen führt als das normal *ReLU-MLP*. Zusätzlich ist abzulesen, dass der Validierungsverlauf mit *ReLU*-Neuronen viel Sprunghafter ist als mit *Sigmoid*-Neuronen.

![Dreischichtiges MLP mit verschiedenen Aktivierungsfunktionen und Voraustraining (Große-Datenbasis) [@hodel] \label{fig:3-layer-compair-medium}](images/3-layer-comair-medium.png)

Werden die Netze mit der großen Datenbasis trainiert, kommt es zu Überraschungen. Hier erzielen die mit *ReLU*-Neuronen besetzten *MLP* bessere Werte. Allen voran das normale, nicht im Voraus trainierte *ReLU-MLP*. Bei den Trainingskosten zeigt der *Sigmoid-SdA* abermals seine Stärken. Das normale *Simgoid-MLP* ist sichtbar das schwächste.

\FloatBarrier

### Vierschichtig

Beim vierschichtigen *SdA* ist bei der kleinen Datenbasis die Streuung der Validierungskosten sehr klein. Mit einem *RMSE* von $0.0141$ hat sich das beste Resultat vom normalen *ReLU-MLP*, gegenüber dem dreischichtigen *SdA* verschlechtert. Eine Regelmäßigkeit lässt sich in den Trainingskosten erkennen. Hier schließen der *Sigmoid-SdA* und das normale *ReLU-MLP* abermals am besten ab.

![Vierschichtiges MLP mit verschiedenen Aktivierungsfunktionen und Voraustraining (Kleine-Datenbasis) [@hodel] \label{fig:4-layer-compair}](images/4-layer-compair.png)

Auch bestätigt sich, dass mit der großen Datenbasis die Unterschiede größer ausfallen und womöglich repräsentativer sind. Interessanterweise schließt bei der großen Datenbasis das normale *Sigmoid-MLP* am besten ab. Klarer Verlierer hingegen ist der *ReLU-SdA*. Mit *0.0122* gegenüber dem Bestwert von *0.0105* der normalen dreischichtigen *ReLU-MLP* ist das vierschichtige *MLP* dem dreischichtigen auch in der großen Datenbasis unterlegen.

![ 4-Schichtige MLP mit verschiedenen Aktivierungsfunktionen und Voraustraining (Große-Datenbasis) [@hodel] \label{fig:4-layer-compair-medium}](images/4-layer-comair-medium.png)

\FloatBarrier

## MLP mit der Aktivierungsfunkion ReLU \label{head:relu_act}

Das in Abbildungen \ref{fig:3-layer-compair} auftegretene Phänomen, dass das nicht im Voraus trainierte, normale *MLP* mit *ReLU*-Neuronen fast gleich gut abschneidet, wie die im Voraus trainierten *SdA* mit *Sigmoid*-Neuronen, wird von der Arbeit "On rectified linear units for speech processing" [@Zeiler_onrectified] in folgendem Absatz bestätigt:

>"...and the units that compose the HDNN “rectified linear units” (ReLUs) [7]. This small change brings several advantages. First, in our experience it eliminates the necessity to have a “pretraining” phase using unsupervised learning [8].  We demonstrate empirically that we can easily and successfully train extremely deep networks even from random initialization.  Second, the convergence of HDNN is faster than in a regular logistic neural net with the same topology.  Third, HDNN is very simple to  optimize.   Even  vanilla  stochastic  gradient  descent  with constant  learning  rate  yields  very  good  accuracy.    Fourth, HDNN generalizes better than its logistic counterpart.  And finally,  rectified  linear  units  are  faster  to  compute  because they do not require exponentiation and division, with an over- all speed up of 25% on the 4 hidden layer neural network we tested on." [@Zeiler_onrectified, Seite 1]

Dies erklärt hingegen nicht das gute Resultat des normalen *Simgoid-MLP* in Abbildung \ref{fig:4-layer-compair-medium}. Eine mögliche Erklärung dafür ist, dass sich das Phänomen vom *Gradientenschwund* erst ab der vierten Schicht bemerkbar macht. Leider wurde aber genau mit dem vierschichtigen *MLP* keine signifikant besseren Werte durch *ReLU*-Neuronen erreicht.

### Schlechtes Abschneiden des ReLU-SdA

Das schlechte Abschneiden des *SdA* mit *ReLU*-Neuronen wird im vorausgehenden Training mit *Sigmoid*-Neuronen vermutet. Es ist dadurch nicht auszuschließen, dass die trainierten Initialwerte suboptimal für *ReLU*-Neuronen sind.

## Wettbewerb-Resultate

Um die *kNN*-Konfigurationen im Wettbewerb bewerten zu lassen, wurden die Netze in gleicher Konfiguration wie in den vorhergehenden Kapitel mit allen verfügbaren Trainingsdaten für 15 Epochen trainiert. Das vorausgehende Training der *SdA*s wurde mit 10 Epochen durchgeführt.

Wie der Tabelle \ref{table:kaggle-results} entnommen werden kann, hat überraschenderweise das normale dreischichtige *MLP* mit *Sigmoid*-Neuronen, welches nicht durch *AutoencoderLayer* im Voraus trainiert wurde, mit dem 11. Platz und einem *RMSE* von **0.01894** am Besten abgeschnitten. Bei allen anderen Schichten hat sich, wie erwartet, der *SdA* gegenüber dem normalen *MLP* durchgesetzt.

Mit den eigenen Datensätze trainiert, hat die Gewinner-Konfiguration, gegenüber den Anderen, schlecht abgeschnitten. Dies lässt die Vermutung zu, dass die gewählten Validierungsdaten den Trainingsdaten zu ähnlich sind, worauf sich *Overfitting* positiv auswirkt. Sehr überraschend ist ebenfalls das schlechte Abschneiden der *ReLU*-Neuronen, da diese in eigenen Tests häufig sehr gut abgeschlossen haben und, wie in Kapitel \ref{head:relu_act} beschrieben, auch in anderen Arbeiten zu Spitzenresultaten führten.

Parallelen zwischen den eigenen sowie den Wettbewerb-Resultate sind:

- der *SdA* mit *ReLU*-Neuronen schließt durchgehend schlecht ab
- drei Schichten sind besser als vier
- der signifikante Sprung liegt zwischen dem einschichtigen *kNN* zu dem zweischichtigen *MLP*

------------------------------------------------------------------
Architektur                 RMSE          Platz   Trainingsdauer
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

  : Wettbewerb-Resultate \label{table:kaggle-results}

### Minibatchgröße

Das beste Modell, das dreischichtige, normale *MLP* mit *Sigmoid*-Neuronen, wurde nachträglich mit einer *Minibatchgröße* von 50 anstatt den 500 trainiert. Das Training dauerte dadurch 15.5 Stunden anstatt 4.5 Stunden, also mehr als dreimal so lange. Im Wettbewerb, erreichte dieses *MLP* einen *RMSE*-Score von **0.01885**. Es schnitt somit minimal besser ab. Die Trainingslänge ist jedoch erheblich höher. Damit konnte bestätigt werden, dass bei der Hyperparametersuche ein möglichst großer *Minibatch* verwendet werden sollte, um schneller Resultate zu erzielen.

### Trainingszeit

Da der Server *deepgreen02* von mehreren Benutzern gleichzeitig benutzt werden kann, sind massive Performance-Schwankungen festgestellt worden. So wurden die Trainingsgänge für das ein- und zweischichtige *kNN* an einem Tag durchgeführt, an dem ein anderer *Python*-Prozess die *CPU* zu fast 100% ausgelastet hat. Dadurch kommen die enormen Trainingszeiten zu Stande. In vorhergehenden Testläufen, waren das einschichtige und zweischichtige *MLP* um einiges schneller.

### Eigene Trainings- und Validationdaten überdenken

Das inkonsistente Abschneiden der Netze im Wettbewerb gegenüber den eigenen Datensätze lässt darauf schließen, dass die gewählten Validierungsdaten zum einen zu spezifisch und zum anderen zu ähnlich zu den Trainingsdaten sind. Es wurde beim Aufteilen der vom Wettbewerb zur Verfügung gestellten Trainingsdaten in eigene Trainings- und Validationsdaten bewusst darauf geachtet, dass sich die Hintergründe und Texte unterscheiden. Die einzigen Merkmale, welche in den Trainings- sowie den Validationsdaten gleich sind, sind die Schriftarten. Es könnte nun ein Validierungsdatensatz erstellt werden, welcher komplett andere Schriften enthält. Dies würde ein generelleres Validieren der Schriften ermöglichen.

Zusätzlich sollten neue, diverse Hintergründe dem Datensatz hinzugefügt werden. Die vorhandenen Hintergründe sind sich tatsächlich sehr ähnlich. Hier könnte in Zukunft eine Methode für automatische Testdatengenerierung durch zufälliges Generieren verunreinigter Bilder Abhilfe schaffen.

### Trainingsmenge

Die Trainingsmenge hat ebenfalls Einfluss auf das Training. Die Tatsache, dass bei gleicher Anzahl trainierter Epochen bei einer doppelt so großen Trainingsmenge auch doppelt so viele Lern-Iterationen anfallen, bestätigt dies. Ebenfalls hat die Trainingsmenge Einfluss auf die Komplexität eines *kNN*. Je mehr Daten zur Verfügung stehen, desto komplexer kann ein *kNN* sein, ohne dass *Overfitting* auftritt (siehe Kapitel \ref{head:overfitting}).

Interessanterweise haben die Tests mit der kleinen Datenbasis gegenüber dem großen zu ähnlichen Resultaten geführt; die Resultate im Wettbewerb, trainiert mit der *Wettbewerb-Datenbasis* jedoch zu überraschend anderen.

### Größte Schwierigkeiten

Die größten Schwierigkeiten bereiten Flecken, welche den Kontrast zur Schrift extrem verringern. In den Wettbewerbs-Testdaten befindet sich ein Hintergrund, welcher exakt dies erfüllt (siehe Abbildung \ref{fig:schicht-vergleich-01}). In den Trainingsdaten befindet sich hingegen nur ein Hintergrund, welcher dem nahe kommt. Dieser Hintergrund verfügt dennoch über einen höheren Kontrast, womit geeignete Trainingsdaten fehlen, um diesen Fall besser abzudecken.

### Detailanalyse ausgewählter Bilder

Für diese Analyse wurde in den Wettbewerb-Testdaten nach exemplarischen Ausschnitten gesucht, an welchen der Unterschied der verschiedenenschichtigen *MLP* aufgezeigt werden kann. Hier werden immer die normalen *Sigmoid-MLP*, welche für den Wettbewerb mit allen Trainingsdaten trainiert wurden, verwendet. Die Schichten sind in aufsteigender Reihenfolge von links nach rechts dargestellt.

![01.png: Bereinigt durch 1-4 schichtiges kNN von links nach rechts aufsteigend [@hodel] \label{fig:schicht-vergleich-01}](images/bild1-schicht-img-compair.png)

In Abbildung \ref{fig:schicht-vergleich-01} ist ein Bild mit dem anspruchsvollsten Hintergrund sichtbar. Das Wort *important* kann selbst vom menschlichen Auge, nur durch erhöhte Konzentration, entziffert werden. Es ist erkennbar, dass das einschichtige *kNN* bereits ein gutes Resultat erzielt. Das zweischichtige *MLP* schafft es die Schrift leserlicher zu machen und bereinigt die Zwischenräume sichtbar besser. Das dreischichtige ist dabei noch gründlicher, was besonders am Verschwinden vom Randbereich des Flecks ersichtlich ist. Beim vierschichtigen *MLP* macht sich wieder eine leichte Verschlechterung wahrnehmbar.

![19.png: Bereinigt druch 1-4 schichtiges kNN von links nach rechts aufsteigend [@hodel] \label{fig:schicht-vergleich-19}](images/bild19-schicht-img-compair.png)

Abbildung \ref{fig:schicht-vergleich-19} zeigt das Bild *19.png*. Es besitzt einen rauen, gefleckten Hintergrund. Hier ist vor allem hervorzuheben, wie die Kontur der Schrift sich bis zur dritten Schicht leicht, aber stetig verbessert. Auch die Freiflächen werden ab der zweiten Schicht annähernd perfekt bereinigt.

\FloatBarrier

Im Bild \ref{fig:schicht-vergleich-22} besteht der Hintergrund aus einem zerknitterten Papier. Die Schrift ist offensichtlich nicht verzerrt, somit wurde der Hintergrund im Nachhinein hinzugefügt. An diesem Hintergrund sind vor allem die Stellen der Wörter *corpus*, *gh Spanish* und *widesp* herausfordernd. Hier ist wahrnehmbar, wie die Bereinigung der Freistellen bis zur dritten Schicht verbessert wird. Was hingegen nur geringfügig funktioniert, ist das Füllen von weißen Ritzen innerhalb der Buchstaben. Dies ist beim Wort *corpus* sehr schön sichtbar.

![22.png: Bereinigt durch 1-4 schichtiges kNN von links nach rechts aufsteigend [@hodel] \label{fig:schicht-vergleich-22}](images/bild22-schicht-img-compair.png)

Das Bild *88.png* der Abbildung \ref{fig:schicht-vergleich-88} besitzt den unproblematischsten Hintergrund. Dieser wurde bereits vom einschichtigen *kNN* sehr gut bereinigt. Einzig an der Stelle des Buchstabens *i* im Wort *important* kann eine Verbesserung erreicht werden. Ebenfalls ist eine leichte Verbesserung der Buchstabenkonturen ab der zweiten Schicht sichtbar.

![88.png: Bereinigt durch 1-4 schichtiges kNN von links nach rechts aufsteigend [@hodel] \label{fig:schicht-vergleich-88}](images/bild88-schicht-img-compair.eps)

\FloatBarrier

### 11. Platz im Wettbewerb

Trotz der inkonsistenten Resultate, wurde der 11. Platz im Wettbewerb erreicht. Andere Mitstreiter, welche noch bessere Resultate erreichten, schafften dies unter anderem durch das Bewusste *Overfitting* der Testdaten. Dies kann durch das Extrahieren und Miteinbeziehen deren Hintergründe im Training erreicht werden. Auch verwenden viele zusätzliche Vorverarbeitungsschritte, wie z.B. automatisierter Kontrastausgleich oder Kantenfinder-Algorithmen. Dies kann aus dem Forum zum Wettbewerb entnommen werden.
