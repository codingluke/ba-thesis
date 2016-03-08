## Theano

### Definition

*Theano* ist eine *Python* Bibliothek, welche das Definieren, Optimieren und Evaluieren von mathematischen Ausdrücken effizient unterstützt. Die große Stärke zeigt sich in der Verarbeitung mehrdimensionaler Arrays.

### Eigenschaften

Theano arbeitet Hand-in-Hand mit der *Python* Bibliothek *numpy* und ergänzt diese mit einer transparenten *GPU*-Schnittstelle. So können kostspielige Berechnungen komfortabel auf der *GPU* ausgeführt werden, ohne das Wissen über die spezifischen *GPU*-Programmiersprachen, wie beispielsweise *CUDA*.

Möglich macht dies die von *Theano* verwendete Graphenstruktur zur Abbildung der mathematischen Ausdrücke. In der Abbildung \ref{fig:thano_graph} ist zu sehen, wie der Ausdruck $z=x+y$ abgebildet wird, wobei x, y und z vom Typ *theano.thensor.matrix* sind.

![Theano Graphenstruktur: Z = matrix(X) + matrix(Y) [@deeplearning.net-2015] \label{fig:thano_graph}](images/theano_graph.png)

Diese Darstellung bringt mehrere Vorteile mit sich:

**Optimierung**

Die mathematischen Ausdrücke können zur Laufzeit optimiert werden.

**Symbolische Differenzierung**

Komplexe Transformationen, wie die im *Gradientenabstiegsverfahren* verwendete partielle Ableitung, können automatisiert werden.

**Automatische Kompilierung in C oder GPU code**

Die Graphen können in verschiedene Zielsprachen kompiliert werden. Momentan wird *C*, *CUDA* unterstützt. Dadurch verbindet sich das Gute aus zwei Welten. Die dynamische Sprache *Python* zur komfortablen Definition und die schnellen, statischen Sprachen *C* oder *CUDA* zur Ausführung.

### Alternativen

*TensorFlow* von *Google* ist eine *C++* Bibliothek, welche viele Konzepte mit *Theano* teilt. Für *TensorFlow* gibt es auch eine *Python* Schnittstelle.

*TensorFlow* hat bereits viele Algorithmen für *kNN* zur Verwendung implementiert und ist darauf ausgelegt, in verteilten Systemen komfortabel einsetzbar zu sein. Ebenfalls die Dokumentation von *TensorFlow* ist der von *Theano* voraus, obwohl das Projekt jünger ist.

Da *Theano* ein reines *open-source*-Projekt ist und *TensorFlow* von Google stammt, wurde für diese Arbeit *Theano* vorgezogen.

