# Aufgabenstellung

## Anforderungensanalyse

### Funktionale Anforderungen

Kaggle Wettbewerb

  ~ : FA01
  ~ Es soll ein bestmögliches Resultat im Wettbewerb [@kaggleDDD] erzielt werden.

Bereinigen von Bilder verschiedener Größe

  ~ : FA02
  ~ Die vorgegebenen Bilder besitzen verschiedene Dimensionen.
  ~ Das kNN muss in der Lage sein alle Bilder zu bereinigen.

Bereinigen verschiedener Schriftarten

  ~ : FA03
  ~ Die Testbilder liegen in verschiedenen Schriftarten vor. Dabei sollen alle vorhandenen Schriftarten gleichermaßen behandelt werden.

Wiederverwendung des Trainierten Modells

  ~ : FA04
  ~ Das Trainierte Modell soll persistent abgelegt werden, damit es später wiederverwendet werden kann, z.B durch einen Python Web-Service.

Datenvorbearbeitung

  ~ : FA05
  ~ Die Bilder dürfen vor dem Bereinigen durch das kNN zugeschnitten werden.
  ~ Es dürfen keine zusätzlichen Filter oder Grafikalgorithmen wie Kantenfinder und Kontrastanpassungen vor der Bereinigung durch das kNN angewendet werden.

kNN

  ~ : FA06
  ~ Es soll ein einschichtiges kNN Trainiert werden.
  ~ Es soll ein mehrschichtiges kNN durch die Methode der Autoencoder Trainiert werden.

Trainingsalgorithmus

  ~ : FA07
  ~ Als Trainingsalgorithmus soll das *Stochastische Gradientenabstiegsverfahren* sowie der *RMSProp* Verwendung finden.
  ~ Regularisation soll durch Dropout, L1, L2 und dem künstlichen Erweitern der Trainingsdaten erreicht werden.
  ~ Momentum soll implementiert werden.
  ~ Trainieren  mit *Mini-Batch* soll möglich sein.
  ~ Die *early-stopping* Funktionalität soll eingebaut sein.

Trainingsverlauf

  ~ : FA08
  ~ Der Trainingsverlauf soll zur späteren Analyse aufgezeichnet werden.
  ~ Um möglichst effizient die besten Hyperparameter für die kNN zu finden, soll ein intelligenter *Gridsearch-Algorighmus* implementiert werden.

Evaluation

  ~ : FA09
  ~ Die Modelle werden durch die von Kaggle [@kaggleDDD] zur Verfügung gestellten Testdaten direkt auf kaggle.com bewertet.
  ~ Die verschiedenen Modelle sollen miteinander auf Trainingsdauer, Trainingsverlauf Verglichen werden.

### Nicht Funktionale Anforderungen

Trainieren mit großen Datenmengen

  ~ : NFA01
  ~ Das Trainieren des kNN soll mit einer Menge von Trainingsdaten möglich sein, welche die Größe des Arbeitsspeicher überschreitet.

Trainieren auf einer GPU

  ~ : NFA02
  ~ Um möglichst viele Konfigurationen von kNN zu vergleichen, soll das kNN so implementiert werden, dass das Trainieren auf einer GPU möglich ist.

Software-Unit-Tests

  ~ : NFA03
  ~ Eingene Klassen und Methoden sollen mit Software-Unit-Tests getestet werden.

Programmiersprache

  ~ : NFA04
  ~ Als Programmiersprache soll *Python* zum Einsatz kommen

Python Bibliotheken

  ~ : NFA05
  ~ Es darf keine umfassende kNN Bibliothek wie *Keras* verwendet werden. Die verwendeten Algorithmen sollen selbst geschrieben werden.
  ~ *Theano* wird zur Optimierung der Algorithmen und deren Portierung auf GPU code verwendet.
  ~ *Spearmint* wird für die Hyperparametersuche eingesetzt
  ~ *Pandas* und *matplotlib* werden für die Visualisierung und Analyse der Lerndaten eingesetzt.

### Abgrenzungen

Bereinigen von Farbbilder

  ~ : AB01
  ~ Das Bereinigen von Farbbilder wird im Rahmen dieser Bachelorarbeit nicht bearbeitet.
  ~ Es werden ausschließlich auf Bilder in Graustufe mit schwarzer Schrift auf weißem Hintergrund berücksichtigt.

Weitere Deep-Learning Techniken

  ~ : AB02
  ~ Auf weitere Deep-Learning Methoden wie die *Bolzmann Maschienen* und *Deep-Belief-Netze* wird nicht eingegangen.
  ~ Ebenfalls werden nur *fast-forward* kNN untersucht. Auf weitere Methoden wie die *recurrent kNN* wird nicht eingegangen.

Handschrift

  ~ : AB03
  ~ Die Bilder dürfen keine Handschrift beinhalten.

## Datenanalyse

Die Trainings und Testdaten werden direkt vom Kaggle Wettbewerb *Denoising Dirty Documnets* zur Verfügung gestellt.
Dabei wurden folgende Eigenschaften in der explorativen Datenanalyse ausfindig gemacht.

Trainingsdaten

  ~ :
  ~ 144 Bilder insgesamt
  ~ 48 Bilder in der Größe (540 x 258)
  ~ 96 Bilder in der Größe (540 x 420)
  ~ Alle Bilder sind in verrauschter und bereinigter Form vorhanden
  ~ 8 verschiedene Hintergründe
  ~ 2 verschiedene Texte
  ~ 3 verschiedenen Schriftarten
  ~ 2 verschiedenen Schriftgrößen
  ~ Kursiv und normal
  ~ Alle Kombinationen von Hintergrund, Text, Schriftart und Stiele vorhanden
  ~ Die größeren Bilder teilen sich die Hintergründe mit den kleineren Bilder und erweitern diese auf ihre Größe
  ~ Bildformat *PNG*

Testdaten

  ~ :
  ~ 72 Bilder insgesamt
  ~ 24 Bilder in der Größe (540 x 258)
  ~ 48 Bilder in der Größe (540 x 420)
  ~ Alle Bilder sind nur in verrauschter Form vorhanden
  ~ 4 verschiedene Hintergründe unterschiedlich zu den Hintergründe der Trainingsdaten
  ~ 5 verschiedene Schriftarten wobei 3 gleich sind zu den Trainingsschriftarten
  ~ 3 verschiedene Schriftgrößen
  ~ 1 Text unterschiedlich zu den 2 Trainingstexte
  ~ Kursiv und normal
  ~ Nicht alle Kombinationen von Hintergrund, Text, Schriftart und Stiele vorhanden
  ~ Die größeren Bilder teilen sich die Hintergründe mit den kleineren Bilder und erweitern diese auf ihre Größe
  ~ Bildformat *PNG*

## Alternativen

### Schwellwert und Kontrast

Die einfachste Möglichkeit automatisiert einen gewissen Grad an Bereinigung von verrauschter Schriftbilder zu erlangen bieten einfache Schwellenwert und Kontrast Algorithmen. Bei Graustufenbilder repräsentiert der Wert 0 eines Pixel Schwarz und der Wert 1 Weiß.

Beim Schwellenwert wird für jeden Pixel des Bildes überprüft, ob dieser eine Schwelle an Grauwert überschreitet. Wenn er dies tut, wird der Wert gelassen, wenn nicht wird der Wert auf Weiß gesetzt. Dadurch entsteht automatisch ein größeren Kontrast. Diese Methode wird auch Tiefpassfilter genannt.

Mit dieser Methode, kann feines Hintergrundrauschen und leichter Graustich entfernt werden. Flecken, welche über die gleiche Intensität wie der Text verfügen, können damit nicht entfernt werden, da diese einfache Funktion das "Wesen" von Text, nicht kennt.

#### Resultat

### Logistic Regression und Random Forest

Eine verbesserte Möglichkeit Bilder zu bereinigen bieten klassische probabilistische Modelle des maschinellen Lernens. Dabei kann eine logistische Regression oder auch einen Entscheidungsbäume wie der *Random Forest* zum Einsatz kommen.

Bei der Bildanalyse schneiden diese Modelle häufig schlecht ab, da es sich bei Bildern meistens um nicht-lineare Daten handelt. Es gibt zwar ebenfalls probabilistische Modelle, welche in der Lage sind nicht-lineare Daten zu verarbeiten, wie z.B. die *Support Vector Machine* mit dem entsprechenden Kernel, diese sprengen jedoch den Rahmen dieser Bachelorarbeit.

#### Resultat logistische Regression

#### Resultat Random Forest

### Andere Programmiersprachen

Der Programmcode zu diese Bachelorarbeit wird in *Python* geschrieben. Es wäre auch möglich *Java*, *Lua*, *C++* oder *R* zu verwenden. Alle vier anderen Sprachen verfügen über gute Bibliotheken.

Java

  ~ deeplearning4j

LUA

  ~ Torch

C++

  ~ Caffe und viele mehr

## Begründung der Wahl

### Künstliche neuronale Netze

Die künstlichen neuronalen Netze werden gewählt, da diese Technologie in den letzten Jahren in fast jedem Wettbewerb die klassischen probabilistischen Modelle übertroffen haben.

Es soll mit dieser Bachelorarbeit bewiesen werden, dass kNN ebenfalls beim Bereinigen von verrauschter Schriftbilder geeignet sind.

### Programmiersprache Python

Die Programmiersprache *Python* wurde gewählt, da es dafür die besten Ressourcen und Tutorien gibt. *Python* genießt in der Welt der Wissenschaft eine große Beliebtheit, so gibt es im Internet ausführliche Beschreibungen der Techniken, welche in dieser Bachelorarbeit verwendet werden. Zusätzlich ist *Python* mit geringem Aufwand installiert, sehr portabel und hat eine komfortable Syntax.

Mit dem Server *deepgreen02* steht ein für *Deep-Learning* mit *Python* optimierter Server der Hochschule für Technik und Wirtschaft zur Verfügung. Auch die Bibliothek *Theano* trägt viel zur Wahl von *Python* bei. Mit ihr ist es so einfach wie noch nie in einer komfortablen, dynamischen Skriptsprache die Algorithmen zu Definieren um diese später, hoch Performant, auf einer GPU auszuführen.

*Python* verfügt ebenfalls über viele Bibliotheken in anderen Bereichen wie z.B. der Web-Programmierung. So können in *Python* geschriebene Modelle einfach als Web-Service Plattform unabhängig zur Verfügung gestellt werden. Auch bieten immer mehr Big-Data Plattformen wie *Apache Hadoop* und *Apache Spark* Schnittstellen für *Python* an.

