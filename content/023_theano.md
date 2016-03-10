## Theano \label{head:theano}

Die folgende Beschreibung grundlegender Eigenschaften der *Python*-Bibliothek *Theano* basiert auf Informationen der offiziellen Projekt-Webseite [@deeplearning.net-2015].

*Theano* ist eine *Python*-Bibliothek, welche das effiziente Definieren, Optimieren und Evaluieren von mathematischen Ausdrücken ermöglicht. Die große Stärke zeigt sich in der performanten Verarbeitung mehrdimensionaler Arrays. Um dies zu erreichen arbeitet *Theano* Hand-in-Hand mit der *Python*-Bibliothek *numpy* und ergänzt diese mit einer *GPU*-Schnittstelle. So können kostspielige Berechnungen komfortabel auf der *GPU* ausgeführt werden, ohne das notwendige Wissen über die spezifischen *GPU*-Programmiersprachen, wie z.B. *CUDA*.

Möglich macht dies die von *Theano* verwendete Graphenstruktur zur Darstellung mathematischer Ausdrücke. In Abbildung \ref{fig:thano_graph} ist zu sehen, wie der Ausdruck, $Z=X+Y$, anhand der Graphenstruktur dargestellt wird, wobei alle Variablen vom Typ *theano.thensor.matrix* sind. Diese Graphenstruktur bringt mehrere Vorteile mit sich:

Optimierung:

  ~ Die mathematischen Ausdrücke können zur Laufzeit optimiert werden.

Symbolische Differenzierung:

  ~ Komplexe Transformationen, wie die im *Gradientenabstiegsverfahren* verwendete partielle Ableitung, können automatisiert werden.

Automatische Kompilierung in C oder GPU code:

  ~ Die Graphen können in verschiedene Zielsprachen kompiliert werden. Momentan werden *C* und *CUDA* unterstützt. Dadurch verbindet sich das Gute aus zwei Welten. Die dynamische Sprache *Python* zur komfortablen Definition und die schnellen, statischen Sprachen *C* und *CUDA* zur Ausführung.

![Visualisierung der Theano-Graphenstruktur am Beispiel, Z = X + Y, wobei alle Variablen vom Typ theano.tensor.matrix sind [@deeplearning.net-2015] \label{fig:thano_graph}](images/theano_graph.png)

### Alternativen

*TensorFlow* von *Google* ist eine *C++* Bibliothek, welche viele Konzepte mit *Theano* teilt. Für *TensorFlow* gibt es zusätzlich eine *Python* Schnittstelle und hat bereits viele Algorithmen zur einfachen Verwendung vorimplementiert. Auch ist es darauf ausgelegt, in verteilten Systemen komfortabel einsetzbar zu sein.

Da *Theano* ein von Firmen unabhängiges, reines *open-source*-Projekt ist und *TensorFlow* hingegen von *Google* stammt, wurde für diese Arbeit *Theano* vorgezogen.

