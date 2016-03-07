# Aufgabenstellung

## Anforderungensanalyse

### Funktionale Anforderungen

Kaggle Wettbewerb

  ~ : FA01
  ~ Es soll ein bestmögliches Resultat im Wettbewerb [@kaggleDDD] erzielt werden.

Bereinigen von Bildern verschiedener Größe

  ~ : FA02
  ~ Die vorgegebenen Bilder besitzen verschiedene Dimensionen.
  ~ Das *kNN* muss in der Lage sein alle Bilder zu bereinigen.

Bereinigen verschiedener Schriftarten

  ~ : FA03
  ~ Die Testbilder liegen in verschiedenen Schriftarten vor. Dabei sollen alle vorhandenen Schriftarten gleichermaßen behandelt werden.

Wiederverwendung des trainierten Modells

  ~ : FA04
  ~ Das trainierte Modell soll persistent abgelegt werden, damit es später wiederverwendet werden kann, z.B durch einen Python Web-Service.

Datenvorbearbeitung

  ~ : FA05
  ~ Die Bilder dürfen vor dem Bereinigen durch das *kNN* zugeschnitten werden.
  ~ Es dürfen keine zusätzlichen Filter oder Grafikalgorithmen wie Kantenfinder und Kontrastanpassungen vor der Bereinigung durch das *kNN* angewendet werden.

kNN

  ~ : FA06
  ~ Es soll ein einschichtiges *kNN* trainiert werden.
  ~ Es soll ein mehrschichtiges *kNN* durch die Methode der *Autoencoder* trainiert werden.

Trainingsalgorithmus

  ~ : FA07
  ~ Als Trainingsalgorithmus soll das *Stochastische Gradientenabstiegsverfahren* sowie der *RMSProp* Verwendung finden.
  ~ Regularisation soll durch *Dropout*, L2 erreicht werden
  ~ Momentum soll implementiert werden.
  ~ Trainieren  mit *Mini-Batch* soll möglich sein.
  ~ Die *early-stopping* Funktionalität soll eingebaut sein.

Trainingsverlauf

  ~ : FA08
  ~ Der Trainingsverlauf soll zur späteren Analyse aufgezeichnet werden.
  ~ Um möglichst effizient die besten Hyperparameter für die *kNN* zu finden, soll ein intelligenter *Gridsearch-Algorighmus* verwendet werden.

Evaluation

  ~ : FA09
  ~ Die Modelle werden durch die von Kaggle [@kaggleDDD] zur Verfügung gestellten Testdaten direkt auf kaggle.com bewertet.
  ~ Die verschiedenen Modelle sollen miteinander auf Trainingsdauer, Trainingsverlauf verglichen werden.
  ~ Die Auswirkung diverser Hyperparameter soll aufgezeigt werden.

### Nicht Funktionale Anforderungen

Trainieren mit großen Datenmengen

  ~ : NFA01
  ~ Das Trainieren des *kNN* soll mit einer Menge von Trainingsdaten möglich sein, welche die Größe des Arbeitsspeichers überschreitet.

Trainieren auf einer GPU

  ~ : NFA02
  ~ Um möglichst viele Konfigurationen von *kNN* zu vergleichen, soll das *kNN* so implementiert werden, dass das Trainieren auf einer GPU möglich ist.

Software-Unit-Tests

  ~ : NFA03
  ~ Eingene Klassen und Methoden sollen mit Software-Unit-Tests getestet werden.

Programmiersprache

  ~ : NFA04
  ~ Als Programmiersprache soll *Python* zum Einsatz kommen.

Python Bibliotheken

  ~ : NFA05
  ~ Es darf keine umfassende *kNN* Bibliothek wie *Keras* verwendet werden. Die verwendeten Algorithmen sollen selbst geschrieben werden.
  ~ *Theano* wird zur Optimierung der Algorithmen und deren Portierung auf GPU code verwendet.
  ~ *Spearmint* wird für die Hyperparametersuche eingesetzt.
  ~ *Pandas* und *matplotlib* werden für die Visualisierung und Analyse der Lerndaten eingesetzt.

### Abgrenzungen

Bereinigen von Farbbilder

  ~ : AB01
  ~ Das Bereinigen von Farbbilder wird im Rahmen dieser Bachelorarbeit nicht bearbeitet.
  ~ Es werden ausschließlich Bilder in Graustufe mit schwarzer Schrift auf weißem Hintergrund berücksichtigt.

Weitere Deep-Learning Techniken

  ~ : AB02
  ~ Auf weitere Deep-Learning Methoden wie die *Bolzmann Maschienen* und *Deep-Belief-Netze* wird nicht eingegangen.
  ~ Ebenfalls werden nur *fast-forward* *kNN* untersucht. Auf weitere Methoden wie die *recurrent kNN* wird nicht eingegangen.

Schriftbilder

  ~ : AB03
  ~ Die Bilder dürfen keine Handschrift beinhalten.
  ~ Es werden nur die Schriftarten und Stiele, welche in den Trainingsdaten vorkommen berücksichtigt.

## Explorative Datenanalyse \label{head:explorative-datenanalyse}

Die Trainings- und Testdaten werden direkt vom Kaggle Wettbewerb *Denoising Dirty Documnets* [@kaggleDDD] zur Verfügung gestellt.
In der explorativen Datenanalyse wurden folgende Eigenschaften ausfindig gemacht:

Trainingsdaten

  ~ 144 Bilder insgesamt; 48 Bilder in der Größe (540 x 258); 96 Bilder in der Größe (540 x 420); alle Bilder sind in verrauschter und bereinigter Form vorhanden; 8 verschiedene Hintergründe; 2 verschiedene Texte; 3 verschiedene Schriftarten; 2 verschiedenen Schriftgrößen; alle Schriftarten sind kursiv und normal vorhanden; alle Kombinationen von Hintergrund, Text, Schriftart und Stil sind vorhanden; die größeren Bilder teilen sich die Hintergründe mit den kleineren Bilder und erweitern diese auf ihre Größe; Bildformat *PNG*.

Testdaten

  ~ 72 Bilder insgesamt; 24 Bilder in der Größe (540 x 258); 48 Bilder in der Größe (540 x 420); alle Bilder sind nur in verrauschter Form vorhanden; 4 verschiedene Hintergründe, welche sich von den Hintergründen der Trainingsdaten unterscheiden; 5 verschiedene Schriftarten identisch zu den Trainingsschriftarten; 3 verschiedene Schriftgrößen; 1 Text der sich von den 2 Trainingstexten unterscheidet; kursiv und normal; nicht alle Kombinationen von Hintergrund, Text, Schriftart und Stil vorhanden; die größeren Bilder teilen sich die Hintergründe mit den kleineren Bildern und erweitern diese auf ihre Größe; Bildformat *PNG*.

**Zusammenfassung**

Das wesentlichste Merkmal ist, dass sich in den Trainings- sowie in den Testdaten exakt die selben Schriftarten befinden. Der Unterschied liegt im neuen Text sowie in den Hintergrundbildern. Das trainierte *kNN* muss im Kaggle-Wettbewerb somit keine neuen Schriftarten erkennen können. Vielmehr ist ein Modell im Vorteil, welches extremes *Overfitting* auf die vorhandenen Schriften ausübt und nicht im Generellen Schriften bereinigen kann.

Ebenfalls ist auffällig, dass die Hintergrundbilder der Testdaten zwar anders ausfallen, jedoch sehr ähnlicher Struktur sind. Es scheint als ob die Trainings- und Testbilder aus gleichen Quellen stammen. Daher wird vermutet, dass Regularisierungstechniken wie die *L2-Regularisation* sowie *Dropout* nur bedingt Verbesserungen mit sich bringen.

## Alternativen

### Schwellenwert und Kontrast

Eine simple Möglichkeit automatisiert einen gewissen Grad an Bereinigung verrauschter Schriftbilder zu erlangen bieten Schwellenwert- und Kontrast-Algorithmen. Bei Graustufenbildern repräsentiert der Wert 0 eines Pixel Schwarz und der Wert 1 Weiß.

Beim Schwellenwert wird für jedes Pixel des Bildes überprüft, ob dieser eine Schwelle an Grauwert überschreitet. Wenn er dies tut, wird der Wert gelassen, wenn nicht wird der Wert auf Weiß gesetzt. Dadurch entsteht automatisch ein größerer Kontrast.

Mit dieser Methode, kann feines Hintergrundrauschen und leichter Graustich entfernt werden. Flecken, welche über die gleiche Intensität wie die des Textes verfügen, können damit nicht entfernt werden, da diese einfache Funktion das "Wesen" von Text nicht kennt.

#### Resultat

Das unter dem Wettbewerb öffentlich hochgeladene Skript *clean-by-thresholding* führt eine soeben beschriebe Schwellenwertfunktion auf das Bild aus wobei ein Pixel ab dem Wert 0.2 auf 1, Weiß, gesetzt wird. In der Mitte der Abbildung \ref{fig:threshold} ist die Bereinigung eines Bildausschnittes zu sehen, links und rechts davon sind die verrauschte und die optimal bereinigte Variante angegeben. Hier ist ersichtlich, dass die feinen Konturen der Schrift nicht gut beibehalten werden. Für diese Lösung ist kein *Kaggle*-Resultat vorhanden. Optisch ist jedoch ersichtlich, dass das Resultat weit hinter der *kNN*s des Kaptitels \ref{head:evaluierung}, liegen dürfte.

![Bereinigung durch eine Schwellenwertfunktion \label{fig:threshold}](images/threshold.png)

Ein anderes veröffentlichtes Skript erweiterte dieses Verfahren durch eine *Fourier-Transformation* zu einem Hochpassfilter. Diese Lösung erreichte ein *RMSE* von $0.09568$ und somit den 95 Platz auf Kaggle.

### Logistic Regression und Random Forest

Eine verbesserte Möglichkeit Bilder zu bereinigen bieten klassische probabilistische Modelle des maschinellen Lernens. Dabei kann eine logistische Regression oder auch ein Entscheidungsbaum wie der *Random Forest* zum Einsatz kommen.

Bei der Bildanalyse schneiden diese Modelle häufig schlecht ab, da es sich bei Bildern meistens um nicht-lineare Daten handelt. Es gibt zwar ebenfalls probabilistische Modelle, welche in der Lage sind nicht-lineare Daten zu verarbeiten, wie die *Support-Vector-Machine* mit dem entsprechenden Kernel, diese sprengen jedoch den Rahmen dieser Arbeit.

#### Resultat

Ein im *Kaggle*-Forum geteiltes *Random-Forest*-Modell erzielt mit einem *RMSE* von $0.02811$ den 49. Platz und überbietet damit das in Kapitel \ref{head:evaluierung} trainierte einschichtige *kNN*. Es ist nicht bekannt, ob beim Training die Hintergründe der Testdaten für vorsätzliches *Overfitting* verwendet wurden.

### Andere Programmiersprachen

Der in diese Bachelorarbeit verwendete Programmcode ist in *Python* geschrieben. Es wäre auch möglich *Java*, *Lua*, *C++* oder *R* zu verwenden. Alle diese Sprachen verfügen über gute Bibliotheken. So besitzt *Java* unter anderem die *Frameworks* *deeplearning4j* und *h2o*, wobei letzteres über eine *API* auch von *R* verwendet werden kann. In *LUA* wurde *Torch* geschrieben und *C++* kennt neben *Caffe* auch *TensorFlow* von Google sowie viele weitere.

## Begründung der Wahl

### Künstliche neuronale Netze

Die künstlichen neuronalen Netze werden gewählt, da diese Technologie in den letzten Jahren in fast jedem Wettbewerb wie bei der in Hand geschriebenen Zahlenerkennung [@mnist], die klassischen probabilistischen Modelle übertroffen haben. Vorallem in der Bildbearbeitung machen *kNN* stetig Vortschritte was über den *Image-Net-Wettbewert* zusätzlich gefördert wird [@imagenet].

Es soll mit dieser Bachelorarbeit bewiesen werden, dass *kNN* ebenfalls für das Bereinigen verrauschter Schriftbilder geeignet sind.

### Die Programmiersprache Python

Die Programmiersprache *Python* wurde gewählt, da es dafür die besten Ressourcen und Tutorien gibt. *Python* genießt in der Welt der Wissenschaft eine große Beliebtheit, so gibt es im Internet ausführliche Beschreibungen der Techniken, welche auch in dieser Arbeit verwendet werden. Zusätzlich ist *Python* mit geringem Aufwand installiert, sehr portabel und hat eine komfortable Syntax.

Mit dem Server *deepgreen02* steht ein für *Deep-Learning* mit *Python* optimierter Server der *Hochschule für Technik und Wirtschaft* zur Verfügung. Auch die Bibliothek *Theano* trägt viel zur Wahl von *Python* bei. Mit ihr ist es so einfach wie noch nie in einer komfortablen, dynamischen Programmiersprache die Algorithmen zu definieren, um diese später, hoch performant, auf einer *GPU* auszuführen.

*Python* verfügt ebenfalls über viele Bibliotheken in anderen Bereichen wie der Web-Programmierung. So können in *Python* geschriebene Modelle einfach als Web-Service Plattform unabhängig zur Verfügung gestellt werden. Auch bieten immer mehr *Big-Data* Plattformen wie *Apache Hadoop* und *Apache Spark* Schnittstellen für *Python* an. [@spark]
