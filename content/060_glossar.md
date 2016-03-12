# Abkürzungsverzeichnis {.unnumbered}

CUDA

  ~ Compute-Unified-Device-Architecture

dA

  ~ Denoising-Autoencoder

GPU

  ~ Graphical-Processing-Unit

kNN

  ~ künstliches neuronales Netzwerk

MLP

  ~ Multi-Layer-Perceptorn

ReLU

  ~ Rectified-Linear-Unit

RMSE

  ~ Root-Mean-Squared-Error

RMSprop

  ~ Root-Mean-Squared-Propagation

SdA

  ~ Stacked-denoising-Autoencoder

SGD

  ~ Stochastik-Gradient-Descent
VPN

  ~ Virtuelles-Privates-Netzwerk

# Glossar {.unnumbered}

Array

  ~ In der Informatik verwendeter Datentyp. Je nach Anwendungsgebiet kann die Bedeutung wechseln. Hier wird Array mit einem Vektor gleichgesetzt.

Backpropagation

  ~ Ein Algorithums zum Trainieren von *kNN*, siehe Kapitel \ref{head:backprop}.

Batch

  ~ *Batch* ist ein englischer Begriff und bedeutet Stapel. In dieser Arbeit wird *Batch* für die Gruppierung einzelner Datensätze zur parallelen Verarbeitung verwendet.

Blocks

  ~ Eine auf *Theano* basierte *Python*-Bibliothek zur Konfiguration und zum Trainieren von *kNN*.


Bias

  ~ Aus der Neurobiologie bzw. Psychologie stammender Begriff. Ein Wert, der die Wahrnehmungsverzerrung bzw. das Vorurteil ausdrückt.

Bibliothek

  ~ Unter Bibliothek wird in der Arbeit eine Programmcode-Bibliothek verstanden.

binary_crossentropy

  ~ Bezieht sich auf die *cross-entropy* und ist deren Implementation in der Bibliothek *Theano*.


Binominalverteilung

  ~ Die Binominalverteilung ist eine Wahrscheinlichkeitsverteilung, welche eine Gaußsche Glockenkurve ergibt.

Compute-Unified-Device-Architecture

  ~ Von *Nvidia* entwickelte Programmierumgebung, mit welcher Grafikkarten für generelle Berechunungen verwentet werden können. Bisher konnten Grafikkarten nur für 3D-Berechnungen verwendet werden

cPickle

  ~ *cPickle* ist die standardmäßige Bibliothek in *Python* um *Python*-Objekte zu *serialisieren*.

cross-entropy

  ~ Die *cross-entropy* ist eine Kostenfunktion, siehe Kapitel \ref{head:kostenfunktion}.

Denoising-Autoencoder

  ~ Eine von Architektur für künstliche neuronale Netze, siehe Kapitel \ref{head:dA}.

deepgreen02

  ~ Ein von der HTW-Berlin zur Verfügung gestellter Server. Auf ihm werden die Berechungen in Kapitel \ref{head:evaluierung} durchgeführt.

dict

  ~ Der *dict* ist ein Datentyp der Programmiersprache *Python*. Dabei handelt es sich um eine Schlüssel-Wertzuweisung. Das *dict* ist vergleichbar mit einem aus Java bekanntem *HashSet*.

Exception

  ~ *Exception* ist ein Konzept in der Informatik um unerwartete Ereignisse abzufangen.

Fule

  ~ Eine *Python*-Bibliothek zur Vorverarbeitung von Daten wärend der Laufzeit.

Framework

  ~ Eine Zusammenstellung verschiedener Bibliotheken, welche zusammen ein spezifisches Problem angehen. Die Begriffe Bibliothek (*Library*) und *Framework* können sich überschneiden. Es ist nicht klar definiert, ab wann eine Bibliothek zu einem *Framework* wird. Ein *Freamework* ist jedoch meistens umfassender.

Grid-Search-Algorithmus

  ~ Der *Grid-Search-Algorithmus* ist ein Algorithmus, welcher aus einer gegebenen Tabelle, alle Werte miteinander kombiniert und in einer Schleife zurückgibt.

Hadamard-Produkt

  ~ auch elementweises Produkt, ist das Produkt zweier Matrizen gleicher Größe, wobei elementweise, die Stellen mit demselben Index zusammen multipliziert werden.

Hyperparametersuche

  ~ Im maschinellen Lernen wird unter *Hyperparameter* ein Parameter für das trainieren von Modellen verstanden. Das Training wiederum, ist die Suche nach den besten Parametern des Modells. Die *Hyperparametersuche* ist somit die Suche der optimalen *Hyperparametern* mit welchen wiederum beim Trainieren die besten Modell-Parameter gesucht werden.

Identität

  ~ Eine Funktion, welche ihr Argument zurückgibt.

ipython notebook

  ~ Das *ipython notebook* ist ein Programm, mit welchem direkt im Browser Python-Programmcode bearbeitet und ausgeführt werden kann. [@ipython-notebook]

Integrationstest

  ~ Ein *Integraiontstest*, auch *black-box-test* genannt, ist ein Softwaretest, mit welchem zusammenhängende Funktionalitäten getestet werden. Es wird also die *Integration* eines Systems zur weiteren Verwendung getestet.

Kaggle

  ~ Kaggle ist eine Plattform, auf welcher Wettbewerbe (Competitions) rund um das Thema Datenanalyse und maschinelles Lernen, stattfinden. Kaggle wurde vor allem durch den auf eine Million dotierte *Netflix*-Wettbewerb aus dem Jahre 2009 bekannt.

Keras

  ~ Eine *Python*-Bibliothek zur Konfiguration und zum Trainieren von *kNN*.

Lernrate

  ~ Die *Lernrate*, auch *Proportionalitätskonstante* genannt, ist, im maschinellen Lernen, ein Wert der angibt, wie stark das Lernen in die eingeschlagene Richtung (Gradient) geschehen soll.

Logistische-Regression

  ~ Die *Logistische-Regression* ist ein linearer Klassifikator im Bereich des maschinellen Lernens.

matplotlib

  ~ matplotlib ist eine *Python*-Bibliothek zur Generierung von Diagrammen.

Minibatch

  ~ Unter *Minibatch* wird in dieser Arbeit ein kleinen Stapel (engl. *Batch*), welcher zusammen mit allen anderen *Minibatches* die gesamte Datenmenge darstellt, verstanden.

Modell

  ~ Unter *Modell* wird in dieser Arbeit ein mathematisches, durch maschinelles Lernen trainiertes, Modell zur vereinfachten Abbildung der Wirklichkeit verstanden.

MongoDB

  ~ *MongoDB* ist eine Schema-freie, dokumentbasierte, noSql Datenbank der Firma *MongoDB, Inc.*.

Multi-Layer-Perceptron

  ~ Die englische, in der Fachwelt geläufigere, Bezeichnung für ein mehrschichtiges künstliches neuronales Netz.

numpy

  ~ Eine in C und *Python* geschriebene Bibliothek für effiziente Matrizen-Operationen in *Python*.

Nvidia

  ~ *Nvidia* ist eine Firma, welche sich auf Grafikkarten spezialisiert hat und mit *CUDA* eine proprietäre Architektur zur Verfügung stellt, Grafikkarten für allgemeine Berechnungen zu verwenden.

open-source

  ~ Unter *open-source* wird in der Arbeit verstanden, dass der Programmcode für alle offen zugänglich ist und dass dieser frei verwendet werden darf. Es gibt auch *open-source*, wo zwar der Programmcode offen zugänglich, jedoch nicht zur freien Verwendung verfügbar ist.

Overfitting

  ~ Das *Overfitting*, Überanpassung, ist ein im maschinellen Lernen verwendeter Begriff, welcher aussagt, dass ein Modell die spezifischen Trainingsdaten "zu gut" gelernt hat und somit auf neuen, generellen Daten schlecht abschneidet, siehe Kapitel \ref{head:overfitting}.

pandas

  ~ Eine *Python*-Bibliothek zur einfachen Manipulation großer Datenmengen.

Python

  ~ In dieser Arbeit wird mit *Python* immer die Programmiersprache gemeint.

Rectified-Linear-Unit

  ~ Eine Aktivierungsfunktion für Neuronen innerhalb eines kNN, siehe Kapitel \ref{head:aktivierungsfunktion}.

Root-Mean-Square-Error

  ~ Der *Root-Mean-Square-Error* ist eine Kostenfunktion, siehe Kapitel \ref{head:kostenfunktion}.

Root-Mean-Squared-Propagation

  ~ Die *Root-Mean-Squared-Propagation* ist ein Algorithmus zur *Backpropagation*, siehe Kapitel \ref{head:rmsprop}.

Spearmint

  ~ *Spearmint* ist eine *Python*-Bibliothek, welche einen intelligenten *Grid-Search-Algorithmus* zur Verfügung stellt.

Sigmoid

  ~ Eine Aktivierungsfunktion für Neuronen innerhalb eines kNN, siehe Kapitel \ref{head:aktivierungsfunktion}.

Stochastik-Gradient-Descent

  ~ Stochastisches Gradientenabstiegsverfahren, siehe Kapitel \ref{head:sgd}.

Stacked-denoising-Autoencoder

  ~ Eine von Architektur für künstliche neuronale Netze, siehe Kapitel \ref{head:stacked-autoencoder}.

Support-Vector-Machine

  ~ Die *Support-Vector-Machine* ist ein Klassifikator im Bereich des maschinellen Lernens, welcher je nach verwendetem Kernel linear aber auch nichtlinear sein kann.

Serialisierung

  ~ Die *Serialisierung* ist in der Informatik, die sequentielle Abbildung bestehender Objekte. Damit kann der Status eines Objekts in einer Datei persistent gesichert werden.

Virtuelles-Privates-Netzwerk

  ~ *VPN* ermöglicht die sichere Kommunikation von lokal getrennten Netzwerken über das Internet

tuple

  ~ Unter *tuple* wird in dieser Arbeit ein Datentyp der Programmierspache *Python* verstanden. Dieser besteht aus mehreren Werten, welche durch ein Komma getrennt werden.

Unittest

  ~ Ein *Unittest* ist ein Softwaretest, mit welchem elementare Methoden von Klassen getestet werden. Optimalerweise wird versucht den Methodenaufruf durch Grenzwertparameter zu testen um möglichst viele Szenarien abzudecken.

Wettbewerb

  ~ In dieser Arbeit ist mit Wettbewerb immer der *Kaggle*-Wettbewerb "Denoising Dirty Documents" [@kaggleDDD] gemeint.
