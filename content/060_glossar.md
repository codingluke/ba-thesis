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

  ~ Mulit-Layer-Perceptorn

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

Root-Mean-Squared-Propagation

  ~ Die *Root-Mean-Squared-Propagation* ist ein Algorithmus zur *Backpropagation*, siehe Kapitel \ref{head:rmsprop}

Stochastik-Gradient-Descent

  ~ Stochastisches Gradientenabstiegsverfahren, siehe Kapitel \ref{head:sgd}

Virtuelles-Privates-Netzwerk

  ~ *VPN* ermöglicht die sichere Kommunikation von lokal getrennten Netzwerken über das Internet

Compute-Unified-Device-Architecture

  ~ Von *Nvidia* entwickelte Programmierumgebung, mit welcher Grafikkarten für generelle Berechunungen verwentet werden können. Bisher konnten Grafikkarten nur für 3D-Berechnungen verwendet werden

Stacked-denoising-Autoencoder

  ~ Eine von Architektur für künstliche neuronale Netze, siehe Kapitel \ref{head:stacked-autoencoder}

Hyperparametersuche

  ~ Ein Hyperparameter ist ein Parameter für Parameter, er könnte auch Metaparameter genannt werden. Die Hyperparametersuche ist somit die Suche von Parameter für weitere Parameter, welche wiederum beim Trainieren gesucht werden.

Rectified-Linear-Unit

  ~ Eine Aktivierungsfunktion für Neuronen innerhalb eines kNN, siehe Kapitel \ref{head:aktivierungsfunktion}.

Sigmoid

  ~ Eine Aktivierungsfunktion für Neuronen innerhalb eines kNN, siehe Kapitel \ref{head:aktivierungsfunktion}.

Keras

  ~ Eine *Python*-Bibliothek zur Konfiguration und zum Trainieren von *kNN*.

Bias

  ~ Aus der neurobiologie, psychologie stammender Begriff. Ein Wert der die Wahrnehmungsverzerrung bzw. das Vorurteil ausdrückt.

Batch

  ~ *Batch* ist ein englischer Begriff und bedeutet Stapel. In dieser Arbeit wird *Batch* für die Gruppierung einzelner Datensätze zur parallelen Verarbeitung verwendet.

Backpropagation

  ~ Ein Algorithums zum Trainieren von *kNN*, siehe Kapitel \ref{head:backprop}.

Minibatch

  ~ Unter *Minibatch* wird in dieser Arbeit einen kleinen Stapel (*Batch*), welcher zusammen mit allen anderen *Minibatches* die gesamte Datenmenge darstellt, verstanden.

Hadamard-Produkt

  ~ auch elementweises Produkt, ist das Produkt zweier Matrizen gleicher Größe, wobei elementweise, die Stellen mit demselben Index zusammen multipliziert werden.

Identität

  ~ Eine Funktion, welche ihr Argument zurückgibt.

Python

  ~ In dieser Arbeit wird mit *Python* immer die Programmiersprache gemeint.

numpy

  ~ Eine in C und *Python* geschriebene Bibliothek für effiziente Matrizen-Operationen in *Python*.

Theano

  ~ Eine in C und *Python* geschriebene Bibliothek für symbolische Mathematik, siehe Kapitel \ref{head:theano}.

Array

  ~ In der Informatik verwendeter Datentyp. Je nach Anwendungsgebiet kann die Bedeutung wechseln. Hier wird Array mit einem Vektor gleichgesetzt.

Multi-Layer-Perceptron

  ~ Die Englische, in der Fachwelt geläufigere, Bezeichnung für ein mehrschichtiges künstliches neuronales Netz.

open-source

  ~ Unter *open-source* wird in der Arbeit verstanden, dass der Programmcode für alle offen zugänglich ist und dass dieser frei verwendet werden darf. Es gibt auch *open-source*, wo zwar der Programmcode zwar offen zugänglich, jedoch nicht zur frei Verwendung verfügbar ist.

Wettbewerb

  ~ In dieser Arbeit wird mit Wettbewerb immer den *Kaggle*-Wettbewerb "Denoising Dirty Documents" [@kaggleDDD] gemeint.

Kaggle

  ~ Kaggle ist eine Plattform, auf welcher Wettbewerbe (Competitions) rund um das Thema Datenanalyse und maschinelles Lernen, stattfinden. Kaggle wurde vor allem durch den auf eine Million dotierte *Netflix*-Wettbewerb aus dem Jahre 2009 bekannt.

Grid-Search-Algorithmus

  ~ Der *Grid-Search-Algorithmus* ist ein Algorithmus, welcher aus einer gegebenen Tabelle, alle Werte miteinander Kombiniert und in einer Schleife zurückgibt.

Modell

  ~ Unter *Modell* wird in dieser Arbeit ein mathematisches, durch maschinelles Lernen trainiertes, Modell zur vereinfachten Abbildung der Wirklichkeit verstanden.

Bibliothek

  ~ Unter Bibliothek wird in der Arbeit eine Programmcode-Bibliothek verstanden.

matplotlib

  ~ matplotlib ist eine *Python*-Bibliothek zur Generierung von Diagrammen.

Support-Vector-Machine

  ~ Die *Support-Vector-Machine* ist ein Klassifikator im Bereich des maschinellen Lernens, welcher je nach verwendetem Kernel linear aber auch nichtlinear sein kann.

Logistische-Regression

  ~ Die *Logistische-Regression* ist ein linearer Klassifikator im Bereich des maschinellen Lernens.

Nvidia

  ~ *Nvidia* ist eine Firma, welche sich auf Grafikkarten spezialisiert hat und mit *CUDA* eine proprietäre Architektur zur Verfügung stellt, Grafikkarten für allgemeine Berechnungen zu verwenden.

tuple

  ~ Unter *tuple* wird in dieser Arbeit ein Datentyp der Programmierspache *Python* gemeint. Dieser besteht aus mehreren Werten, welche durch ein Komma getrennt werden.

cPickle

  ~ *cPickle* ist die standardmäßige Bibliothek in *Python* um *Python*-Objekte zu *Serialisieren*.

Serialisierung

  ~ Die *Serialisierung* ist in der Informatik, die sequentielle Abbildung bestehender Objekte. Damit kann der Status eines Objekts in einer Datei persistent gesichert werden.

Exception

  ~ *Exception* ist ein Konzept in der Informatik um unerwartete Ereignisse abzufangen.

Root-Mean-Square-Error

  ~ Der *Root-Mean-Square-Error* ist eine Kostenfunktion, siehe Kapitel \ref{haed:kostenfunktion}.

cross-entropy

  ~ Die *cross-entropy* ist eine Kostenfunktion, siehe Kapitel \ref{haed:kostenfunktion}.

binary_crossentropy

  ~ Bezieht sich auf die *cross-entropy* und ist deren Implementation in der Bibliothek *Theano*.

Denoising-Autoencoder

  ~ Eine von Architektur für künstliche neuronale Netze, siehe Kapitel \ref{head:dA}.

Binominalverteilung

  ~ Die Binominalverteilung ist eine Wahrscheinlichkeitsverteilung, welche eine Gaußsche Glockenkurve ergibt.

Spearmint

  ~ *Spearmint* ist eine *Python*-Bibliothek, welche einen intelligenten *Grid-Search-Algorithmus* zur Verfügung stellt.

ipython notebook

  ~ Das *ipython notebook* ist ein Programm, mit welchem direkt im Browser Python-Programmcode bearbeitet und ausgeführt werden kann. [@ipython-notebook]

MongoDB

  ~ *MongoDB* ist eine Schema-freie, dokumentbasierte, noSql Datenbank der Firma *MongoDB, Inc.*.

Unittest

  ~ Ein *Unittest* ist ein Softwaretest, mit welchem elementare Methoden von Klassen getestet werden. Optimalerweise wird versucht der Aufruf durch Grenzwertparameter zu testen um möglichst viele Szenarien abzudecken.

Integrationstest

  ~ Ein *Integraiontstest*, auch *black-box-test* genannt, ist ein Softwaretest, mit welchem zusammenhängende Funktionalitäten getestet werden. Es wird also die *Integration* eines Systems zur weiteren Verwendung getestet.

dict

  ~ Der *dict* ist ein Datentyp der Programmiersprache *Python*. Dabei handelt es sich um ein Schlüssel, Wertzuweisung. Das *dict* ist vergleichbar mit einem aus Java bekanntem *HashSet*.

deepgreen02

  ~ Ein von der HTW-Berlin zur Verfügung gestellter Server. Auf ihm werden die Berechungen in Kapitel \ref{head:evaluierung} durchgeführt.

Lernrate

  ~ Die *Lernrate*, auch *Proportionalitätskonstante* genannt, ist, im maschinellen Lernen, ein Wert der angibt wie stark das Lernen in die eingeschlagene Richtung (Gradient) geschehen soll.

pandas

  ~ Eine *Python*-Bibliothek zur einfachen Manipulation großer Datenmengen.

Fule

  ~ Eine *Python*-Bibliothek zur Vorverarbeitung von Daten wären der Laufzeit.

Blocks

  ~ Eine auf *Theano* basierte *Python*-Bibliothek zur Konfiguration und zum Trainieren von *kNN*.

Framework

  ~ Eine Zusammenstellung verschiedener Bibliotheken, welche zusammen in ein spezifisches Problem angehen. Die Begriffe Bibliothek (*Library*) und *Framework* können sich überschneiden. Es ist nicht klar definiert ab wann eine Bibliothek zu einem *Framework* wird. Ein *Freamework* ist jedoch meistens umfassender.

Overfitting

  ~ Das *Overfitting*, Überanpassung, ist ein im maschinellen Lernen verwendeter Begriff, welcher aussagt, dass ein Modell die spezifischen Trainingsdaten zu gut gelernt hat und somit auf neuen, generellen Daten schlecht abschneidet, siehe Kapitel \ref{head:overfitting}.
