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

Software-Unit-Tests

  ~ : AB04
  ~ Die kNN werden nicht durch Unit-Tests geprüft. Die Evaluation auf Kaggle soll für die Zuverlässigkeit garantieren.

## Datenanalyse


## Alternativen

### Schwellwert und Kontrast

### Logistic Regression und Random Forest

### TensorFlow

### Andere Programmiersprachen

## Begründung der Wahl
