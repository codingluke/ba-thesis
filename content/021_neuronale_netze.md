## Künstliche neuronale Netze \label{head:kNN}

Die nachfolgende Erläuterung der Grundlagen künstlicher neuronaler Netze wurden vorwiegend aus dem Online-Buch "Neural Networks and Deep Learning" von Michael Nilson [@nielsen_2015] ins Deutsche übersetzt, sowie aus dem Kapitel "Künstliche neuronale Netze" des Buches "Künstliche Intelligenz - ein moderner Ansatz" von Peter Norvig [@ki-norvig, S.845-856], in eigener Sprache verfasst. Weitere Quellen werden an entsprechender Stelle im Fließtext angegeben.

### Ursprung

Künstliche neuronale Netze, kurz *kNN*, sind der Versuch die aus der Neurowissenschaft bekannten Konzepte der neuronalen Netze, künstlich, mittels Programmcode, nachzubauen. Erste logische Nachahmungen eines organischen Neuron und deren Verknüpfungen zu Netze wurde bereits 1943 von McCulloch und Pitts beschrieben [@mcculloch].

Weiterentwickelt wurde es durch Frank Rosenblatt. Dieser entwickelte vor allem Ende der 1950er Jahre das Konzept des Perzeptrons [@rosenblatt1958perceptron].

### Das Neuron \label{neuron_aufbau}

Das Neuron ist das Kernstück der *kNN* (siehe Abbildung \ref{perzeptron-ki-norvig}). Das einfachste erdenkliche *kNN* besteht aus genau einem Neuron (wobei es streng genommen kein Netz mehr darstellt). Ein Neuron besteht aus folgenden Bestandteilen:

![Aufbau eines Neurons [@ki-norvig, S.846] \label{perzeptron-ki-norvig}](images/perzeptron.jpg)

- Eingabeverknüpfungen; besitzen eigene Gewichte
- Eingabefunktion; besitzt einen Bias pro Neuron
- Aktivierungsfunktion; berechnet die Aktivierung anhand der Eingabefunktion
- Ausgabe; Resultat der Aktivierungsfunktion
- Ausgabeverknüpfungen

#### Eingabeverknüpfungen

Die Eingabeverknüpfungen (in Abbildung \ref{perzeptron-ki-norvig} als $a$ angegeben), werden häufig auch als Eingabevektor $X$ bezeichnet. Die Ziffer $a$ wurde hier gewählt, da der Eingabevektor sich in einem Netz aus Ausgabeverknüpfungen zusammenstellen kann.

Ein Neuron kann über mehrere Eingabeverknüpfungen verfügen, wobei $a_{0}$ immer mit 1 angegeben ist und zusammen mit dem Gewicht $W_{0,j}$ den Bias darstellt.

Jede Eingabeverknüpfung verfügt über eine individuelle Gewichtung. Diese Gewichtung wird mit dem jeweiligen Eingabewert verrechnet.

#### Eingabefunktion

Die Eingabefunktion bestimmt, wie die Eingabewerte mit den Gewichten verrechnet und zusammengefasst werden. Üblicherweise wird dies durch die Multiplikation der einzelnen Eingabewerte mit deren Gewichten und deren anschließender Aufsummierung getan (siehe Gleichung \ref{eq:eingabefunktion}).

\begin{equation} \label{eq:eingabefunktion}
  in_j = \displaystyle\sum_{i=0}^{n} w_{i,j}*a_i
\end{equation}

#### Aktivierungsfunktion \label{head:aktivierungsfunktion}

Die Aktivierungsfunktion definiert den Wertebereich, welcher von dem Neuron ausgegeben wird und beeinflusst dadurch maßgeblich dessen Eigenschaft. Dabei wird die Ausgabe $in_j$ (der Eingabefunktion \ref{eq:eingabefunktion}) als Eingabe der Aktivierungsfunktion $g$ (in Gleichung \ref{eq:aktivierungsfunktion}) verwendet.

\begin{equation} \label{eq:aktivierungsfunktion}
a_j = g(in_j)
\end{equation}

Bei dem einfachen Neuron (auch Perzeptron genannt) wird eine Schwellenwertfunktion (siehe Gleichung \ref{eq:schwellenwert}) verwendet, welche den Wert 0 oder 1 zurück gibt. Diese symbolisieren, ob das Neuron aktiv 1 oder inaktiv 0 ist. Die Schwellenwertfunktion hat sich in der Praxis als zu grob herausgestellt. Kleine Änderungen am Bias und den Gewichten können zu einer großen Veränderung des Resultats führen und sind dadurch unberechenbar. [vgl. @nielsen_2015, K.1]

\begin{equation} \label{eq:schwellenwert}
g(in_j) =
  \begin{cases}
    0  & \quad \text{if } in_j \leq \text{ Schwellenwert}\\
    1  & \quad \text{if } in_j > \text{ Schwellenwert}\\
  \end{cases}
\end{equation}

Wird als Aktivierungsfunktion die Sigmoid-Funktion (siehe Gleichung \ref{eq:sigmoid}) verwendet, ist das Resultat immer eine Gleitkommazahl zwischen $0$ und $1$. Somit ist ein Neuron nicht aktiv oder inaktiv, sondern eher aktiv oder eher inaktiv. Eine kleine Änderung an den Bias oder Gewichten führt ebenfalls zu einer kleinen Veränderung des Resultats. Das Sigmoid-Neuron ist daher berechenbarer und besser zu trainieren. [vgl. @nielsen_2015, K.1]

\begin{equation} \label{eq:sigmoid}
g(in_j) = \frac{1}{1 + e^{-in_j}}
\end{equation}

Eine weitere Aktivierungsfunktion ist die *Rectified-Linear-Unit*, *ReLU*, zu deutsch Gleichgerichtete Lineare Einheit, auch bekannt als "Rampenfunktion" (siehe Gleichung \ref{eq:relu}). Dabei handelt es sich um eine lineare Funktion, welche alle Negativwerte auf $0$ anhebt und die positiven Werte unverändert stehen lässt. Die Rampenfunktion hat sich laut [@GlorotBB11] gegenüber der Sigmoid-Funktion als natürlicher und plausibler erwiesen und erhält immer mehr Beachtung, vor allem im *Deep-Learning*.

\begin{equation} \label{eq:relu}
  g(in_j) = \max(0, in_j)
\end{equation}

#### Ausgabe und Ausgabeverknüpfungen

Die Ausgabe ist das Resultat der Aktivierungsfunktion ($a_j$). Dieser Wert kann als Ausgabeverknüpfung an beliebige weitere Neuronen weitergegeben werden. Aus der Perspektive der Empfängerneuronen entspricht eine Ausgabeverknüpfung einer Stelle ihres Eingabevektors, welche wiederum eine eigene Gewichtung besitzt.

### Das Neuron als logisches Bauteil

Ein Neuron, wie im Kapitel \ref{neuron_aufbau} beschrieben, kann durch die Veränderung der Gewichte der einzelnen Eingangsverknüpfungen und dem Bias jedes beliebige logische Bauteil repräsentieren. Ein *kNN* kann dadurch als eine virtuelle Schaltplatine angesehen werden, bei welcher die Bauteile dynamisch, durch Anpassung der Gewichte, verändert werden können. Somit kann ein *kNN* mit einer Anzahl von $N$ Neuronen theoretisch jede mögliche, logische Schaltung mit der gleichen Anzahl an Bauteilen darstellen.

![Der logische Addierer (links) dargestellt durch Neuronen (rechts). Die der Bias ist bei allen Neuronen 3 und die Gewichte -2 [@nielsen_2015, K.1] \label{addierer-perceptron}](images/schaltung-knn.png)

### Vom Neuron zum Netz {#kNN_aufbau}

Wie in Abbildung \ref{mlp-generell} dargestellt bestehen *kNN* aus Neuronen, welche untereinander zu einem Netz verbunden werden. Dabei wird zwischen den Eingangsneuronen, unsichtbaren Neuronen und Ausgangsneuronen, gekennzeichnet durch die jeweiligen Schichten, unterschieden [vgl. @ki-norvig, S.845].

Die Ausgangsverknüpfungen einer Schicht werden zu Eingangsverknüpfungen der nächsten Schicht. Dadurch ergibt sich ein gerichteten Graphen. *KNN* die diesem Modell folgen, werden als "feed-forward Netze" bezeichnet. Auf andere Modelle wird im Rahmen dieser Arbeit nicht eingegangen, *kNN* sind folglich in dieser Bachelorarbeit immer "feed-forward Netze".

![Darstellung eines mehrschichtigen feed-forward kNN [vgl. @nielsen_2015, K.1] \label{mlp-generell}](images/mlp-generell.png)

Ein *kNN* besitzt immer je eine Schicht von Eingangs- und Ausgangsneuronen. Dazwischen können sich keine, eine oder mehrere Schichten von unsichtbaren Neuronen befinden. Die mittleren Schichten werden "unsichtbar", oder auch "hidden" genannt, da auf diese von außen in der Regel nicht zugegriffen werden kann. Die einzelnen Schichten können eine beliebige Anzahl Neuronen enthalten, in Abbildung \ref{mlp-generell} wird nur eine Möglichkeit aufgezeigt.

Besitzt ein *kNN* keine unsichtbare Schicht handelt es sich um ein Kernel-Perzeptron. Auf dieses wird im Rahmen der vorliegender Bachelorarbeit nicht eingegangen.

Besitzt ein *kNN* eine unsichtbare Schicht, handelt es sich um ein einschichtiges *kNN*. Einschichtige *kNN* sind universal; das heißt sie sind theoretisch im Stande jede beliebige Funktion darzustellen.

Besitzt ein *kNN* mehrere unsichtbare Schichten, wird auch von einem mehrschichtigen *kNN* gesprochen. Im Englischen werden diese auch *Multi Layered Perceptron*, kurz *MLP*, genannt. Bei der Verwendung von *MLP* wird darauf Ableitend auch von *Deep-Learning* gesprochen. Deep, da diese eine gewisse Tiefe durch die unsichtbaren Schichten besitzen. [vgl. @ki-norvig, S.846-850]

### KNN als Funktion

Ein *kNN* kann als Funktion betrachtet werden, wobei der Eingangsvektor $X$ dem Funktionsparameter entspricht und der Ausgangsvektor $y$ dem Funktionswert (siehe Gleichung \ref{eq:knn-funktion}).

\begin{equation} \label{eq:knn-funktion}
  f(X) = y
\end{equation}

Je nach Konfiguration und Trainingsdaten kann das *kNN* einem Klassifikator, Regressor oder einer beliebigen anderen Funktion entsprechen.

### Trainieren eines kNN

Beim Trainieren eines *kNN* wird versucht, die Gewichte der einzelnen Eingangsverknüpfungen so zu modifizieren, dass bei einem bestimmten Eingangsvektor $X$ der entsprechende Ausgangsvektor $y$ resultiert.

Um das Resultat zu überprüfen wird eine *Kostenfunktion* verwendet. Die Modifikation der Gewichte wird mehrmals in kleinen Schritten ausgeführt. Dabei versucht man den Kostenfunktionswert zu minimieren.

Um zu wissen in welche Richtung die Gewichte angepasst werden müssen, wird das in Kapitel \ref{head:gradientenabstiegsverfahren} beschriebene *Gradientenabstiegsverfahren* angewandt.

#### Kostenfunktion

Die Kostenfunktion^[Die Kostenfunktion wird im Englischen als *cost*, oder auch *loss* bezeichnet] berechnet die Abweichung, bzw. den Fehler, vom aktuell berechneten Ausgangswert zum gewünschten Zielwert. Anhand dieser Abweichung werden für jedes Neuron im Netz das Gewicht und den Bias modifiziert.

Die klassische und auch von dem Kaggle Wettbewerb [@kaggleDDD] vorgeschriebene Kostenfunktion ist der *Root-Mean-Squared-Error*, *RMSE*. Der *RMSE*, dargestellt in Gleichung \ref{eq:rmse}, berechnet die Distanz von jeder Stelle des Zielvektors zu der entsprechenden Stelle des berechneten Ausgangsvektors. Die einzelnen Distanzen werden quadriert und miteinander aufsummiert. Die Distanzen werden deswegen quadriert, um dem Fehler ein höheres Gewicht zu geben. Wenn diese nicht quadriert würden, wäre der Lernfortschritt kleiner. Am Schluss wird davon wieder die Wurzel gezogen. [vgl. @nielsen_2015, K.2]

\begin{equation} \label{eq:rmse}
  RMSE(w, b) \equiv \sqrt{\frac{1}{2n} \displaystyle\sum_{X} \|y(X) - a \|^2}
\end{equation}

In Gleichung \ref{eq:rmse} steht $w$ für die Menge aller Gewichtsvektoren und $b$ für die Biasvektoren. Die Funktion $y(X)$ steht für den Zielvektor und $a$ für den Ausgangsvektor. Die Variable $X$ Steht für eine Menge von Eingangsvektoren, wobei die Variable $n$ dessen Größe beinhaltet. Es werden also die Gesamtkosten aller Trainingsdaten berechnet. [vgl. @nielsen_2015, K.1]

Eine andere Kostenfunktion, die *cross-entropy* Funktion, kommt in modernen *kNN* immer häufiger zum Einsatz (siehe Gleichung \ref{eq:cross-entropy}). Sie hat den Vorteil, dass der Lernprozess kontinuierlicher abläuft. Der *RMSE* hat hingegen die Eigenschaft am Anfang sehr große Fehler zu finden welche schnell abflachen. [vgl. @nielsen_2015, K.3]

\begin{equation} \label{eq:cross-entropy}
  CE(w, b) \equiv -\frac{1}{2n}
  \displaystyle\sum_{X} [y(X)ln(a) + (1 - y(X))ln(1-a)]
\end{equation}

#### Gradientenabstiegsverfahren \label{head:gradientenabstiegsverfahren}

Beim Gradientenabstiegsverfahren wird die Kostenfunktion mit deren Variablen als Tal angenommen. Abbildung \ref{gradientenabstieg} zeigt dies anhand der Funktion $C$ auf der *y-Achse*, welche zwei Variablen, $v_1$ auf der *x-Achse* und $v_2$ auf der *z-Achse*, voraussetzt.


![Gradientenabstiegsverfahren im 3D Raum [@nielsen_2015, K.2] \label{gradientenabstieg}](images/gradientenabstieg.png)

Das Ziel ist die Kosten zu Minimieren. Dies bedeutet den Punkt zu finden, welcher auf der *y-Achse* den geringsten Wert besitzt. Um diesen Wert zu erreichen, müssen die Variable $v_1$ und $v_2$ auf einen bestimmten Wert gesetzt werden. Um die Brücke zu *kNN* zu schlagen können die Variablen $v_1$ und $v_2$ als die Gewichte und Bias angesehen werden.

Wird ein zufälliger Startwert gewählt, z.B. der Punkt des grünen Balls in Abbildung \ref{gradientenabstieg}, kann damit der Gradient dieses Punktes berechnet werden. Der Gradient ist die partielle Ableitung der Funktion $C$, wobei die Variablen $v_1$ und $v_2$ nun einen festen Wert besitzen.

Anhand dem Gesetz der partiellen Ableitung zeigt der Gradient in die Richtung des Maximums. Wird der Gradient invertiert, zeigt der resultierende Vektor in die Richtung des Minimums. Dieser Vektor ist in Abbildung \ref{gradientenabstieg} als grüner Pfeil eingezeichnet.

So ist es möglich schrittweise die Variablen dem Optimum anzugleichen. Am Beispiel der Gewichte und Bias lauten die Aktualisierungsregeln pro Schritt folgendermaßen:

\begin{equation} \label{eq:update_gewichte}
  w \to w'  = w - n
  \frac{\partial C_{X}}{\partial w}
\end{equation}

\begin{equation} \label{eq:update-bias}
  b \to b'  = b - n
  \frac{\partial C_{X}}{\partial b}
\end{equation}

Um das Gradientenabstiegsverfahren auf kleinere Untergruppen der Trainingsdaten, auch *Batches* genannt, anzuwenden, müssen diese wie folgend aufsummiert werden.

\begin{equation} \label{eq:update_gewichte}
  w \to w'  = w -\frac{n}{m}
  \displaystyle\sum_{j} \frac{\partial C_{X_j}}{\partial w}
\end{equation}

\begin{equation} \label{eq:update-bias}
  b \to b'  = b -\frac{n}{m}
  \displaystyle\sum_{j} \frac{\partial C_{X_j}}{\partial b}
\end{equation}

#### Backpropagation-Algorithmus \label{head:backprop}

Der *Backpropagation-Algorithmus* wurde ursprünglich im Jahre 1974 von Paul Werbos an der Harvard Universität entwickelt [@backprop]. In der Praxis findet er aber erst seit 1986 durch die Arbeit "Beyond regression: new tools for prediction and analysis in the behavioral sciences" von David Rumelhart, Geoffrey Hilton und Ronald Williams [@RumelhartHintonWIlliams1986] Verwendung.

Er löste das Problem der effizienten Gewichtsfindung in den versteckten Schichten. Davor geschah dies auf extrem ineffiziente Weise, welche die anfängliche Euphorie über *kNN* bis in die 80er Jahre verstummen lies. Die folgende Beschreibung wurde aus der [@nielsen_2015, K.2] übersetzt.

Der Algorithmus besteht im Wesentlichen aus drei Schritten:

1. **Feed-forward**: Das Eingabemuster wird durch das *kNN* geführt.
2. **Ausgangsfehler**: Der Ausgabevektor wird mittels der Kostenfunktion mit dem Zielvektor verglichen und der Fehlervektor daraus abgeleitet.
3. **Rückführung des Fehlers (Backpropagate)**: Der Ausgangsfehler wird nun schichtweise zurückgeführt. Dadurch erhält jede Schicht einen eigenen Fehlerwert, der vom Ausgangsfehler beeinflusst wird.

Für diese Schritte werden vier wesentliche Gleichungen benötigt:

**1. Berechnung des Fehlers in der Ausgangsschicht**

\begin{eqnarray} \label{eq:backprop-1}
  \delta^L_j = \frac{\partial C}{\partial a^L_j} g'(in^L_j)
\end{eqnarray}

Die Gleichung \ref{eq:backprop-1} besteht aus zwei Termen. Der erste, linke Term beschreibt wie schnell sich der Fehler $\delta^L_j$, des $j$-ten Neuron der Ausgangsschicht $L$, anhand der Konstenfunktion, in Relation zu dessen Aktivierung $a^L_j$ ändert. Der zweite Term misst, wie schnell sich die Aktivierungsfunktion $g$ durch den Eingabefunktionswert $in^L_j$ ändert. Diese Berechnung muss für jedes Ausgangsneuron gemacht werden. Dafür gibt es die Vektordarstellung $\delta^L = \nabla_{a^L} C \odot \sigma'(in^L)$, wobei $\odot$ das *Hadamard Produkt* darstellt. $\delta^L$ steht somit für einen Vektor aller Fehler, $a^L$ für alle Aktivierungswerte und $in^L$ für alle Eingabefunktionswerte der Ausgangsschicht.

**2. Berechnung des Fehlervektors $\delta^l$ einer unsichtbaren Schicht $l$ anhand des Fehlervektors der darauffolgenden Schicht $\delta^{l+1}$**

\begin{eqnarray} \label{eq:backprop-2}
  \delta^l = ((w^{l+1})^T \delta^{l+1}) \odot g'(in^l)
\end{eqnarray}

Der rechte Term der Gleichung \ref{eq:backprop-2} schließt auf die Fehlerdifferenz der Schicht $l$ ausgehend vom Fehlervektor $\delta^{l+1}$ der Folgeschicht und deren aktuellem Gewichtsvektor $w^{l+1}$. Durch das *Hadamard Produkt* wird dieser Term der Änderungsrate der Aktivierungsfunktion $g$ der Schicht $l$ angerechnet und ergibt den angenommenen Fehlervektor $\delta^l$. So wird der Fehler schichtweise von der Ausgangsschicht auf beliebig viele vorhergehende, unsichtbare Schichten zurückgeführt.

**3. Berechnung des Gradienten der Kostenfunktion in Relation zu den Bias im Netzwerk**

\begin{eqnarray} \label{eq:backprop-3}
  \frac{\partial C}{\partial b^l_j} = \delta^l_j
\end{eqnarray}

Die Gleichung \ref{eq:backprop-3} zeigt, dass sich die partielle Ableitung, also Änderungsrate, der Kostenfunktion an der Position des $j$-ten Neurons der $l$-ten Schichtrespektive gleich verhält, wie der bereits berechnete Fehler $\delta^l_j$

**4. Berechnung des Gradienten der Kostenfunktion in Relation zu den Gewichten im Netzwerk**

\begin{eqnarray} \label{eq:backprop-4}
  \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j.
\end{eqnarray}

Die Gleichung \ref{eq:backprop-4} zeigt wie die Änderungsrate, der Gradient, der Kostenfunktion an der Stelle von jedem Neuron in jeder Schicht in Relation zu den jeweiligen Gewichten berechnet werden kann. Dafür muss der Aktivierungswert $a^{l-1}_k$ der vorhergehenden Schicht $l-1$ , welcher dem Wert der Eingabeverknüpfung entspricht, mit dem Fehler $\delta^l_j$ der zu berechnenden Schicht $l$ multipliziert werden.

#### Stochastisches Gradientenabstiegsverfahren

Aufbauend auf dem Kapitel \ref{head:backprop}, welches anhand dem *Backpropagation-Algorithmus* aufzeigt, wie die Gradienten der Kostenfunktion der einzelnen Neuronen berechnet werden können, ist das *stochastische Gradientenabstiegsverfahren*, ein Algorithmus, um die Gewichte in Anbetracht der berechneten Gradienten zu modifizieren.

Die Gewichte werden für jede Schicht der Gleichung $w^l \rightarrow w^l-\frac{\eta}{m} \sum_x \delta^{x,l} (a^{x,l-1})^T$ angepasst. Die Bias mit der Gleichung $b^l \rightarrow b^l-\frac{\eta}{m} \sum_x \delta^{x,l}$.

Werden diese zwei Gleichungen mit den im Kapitel \ref{gradientenabstiegsverfahren} beschriebenen Gleichungen \ref{eq:update_gewichte} und \ref{eq:update-bias} verglichen, wird ersichtlich, dass die darin zu berechnenden Gradienten nun durch die im *Backpropagation-Algorithmus* berechneten Gradienten ausgetauscht werden.

#### RMSProp / Root Mean Square Propagation

Die *RMSProp* wurde von Tijemen Tieleman [@Tieleman2012] vorgeschlagen und wird von Geoffrez Hinton im Kurs *COURSERA: Neural Networks for Machine Learning* vermittelt. Besagtes Verfahren zur Gewichtsmodifikation hat in der vorliegenden Bachelorarbeit zu sehr guten Ergebnissen geführt. Zur Zeit gibt es keine offizielle Veröffentlichung des Verfahrens.

Das Verfahren erweitert das *stochastische Gradientenabstiegsverfahren* insofern, dass der Gradient durch den *Root-Mean-Square* aller vorhergehenden Gradienten skaliert wird. Dadurch haben die Gradienten vorhergehender Lernintervallen Einfluss auf den aktuellen Gradienten.

\begin{eqnarray} \label{eq:rmsprop-1}
  MeanSquare(w^l_{jk}, t) = 0.9 * MeanSquare(w^l_{jk}, t-1) + 0.1 (G^{lt}_{jk})^2
\end{eqnarray}

Die Gleichung \ref{eq:rmsprop-1} zeigt, wie der durch *Backpropagation* angenommene Gradient $G^{lt}_{jk}$ der Kostenfunktion $C$ in Relation der Gewichte $w^l_{jk}$ rekursiv über die Trainingszeit $t$, im Quadrat gemittelt, mitgeführt wird. Dabei ist der mitgeführte, gemittelte Gradient mit $0.9$ stärker gewichtet als der Aktuelle.

\begin{eqnarray} \label{eq:rmsprop_2}
  G =  \frac{G^{lt}_{jk}}{\sqrt[2]{MeanSquare(w^l_{jk}, t)}}
\end{eqnarray}

Der für die Gewichtsanpassung analog der *stochastischen Gradientenabstiegsverfahren* zu verwendende Gradient $G$ wird berechnet, indem der durch *Backpropagation* angenommene Gradient $G^{lt}_{jk}$ durch die Wurzel des mitgeführten, gemittelten Gradienten geteilt wird.

![Vergleich verschiedener Backpropagation Methoden \label{rmsprop-compair} [@rnn, Min.18]](images/rmsprop_comairation.png)

In der Abbildung \ref{rmsprop-compair} ist sichtbar, dass der *RMSprop* (Schwarz) gegenüber dem *SGD* (Rot) viel schneller das Optimum (Stern) erreicht. Der durch *Momentum* erweiterte *SGD* (grün), welcher als nächstes erläutert wird, schneidet besser ab als der einfache *SGD*, legt hingegen einen weiteren Weg als der *RMSprop* zurück. Auf die drei anderen aufgeführten Methoden NAG, Adagrad und Adadelta wird in dieser Bachelorarbeit nicht eingegangen.

#### Momentum

Der Term *Momentum* bedeutet auf Deutsch Schwung, Eigendynamik, Moment oder auch Wucht. *Momentum* ist der Versuch, das Gradientenabstiegsverfahren mit dem aus der Physik bekannten Geschwindigkeitszuwachs beim Abstieg zu ergänzen.

\begin{eqnarray}
  v & \rightarrow  & v' = \mu v - \eta \nabla C \label{eq:momentum-1}\\
  w & \rightarrow & w' = w+v'.  \label{eq:momentum-2}
  \end{eqnarray}

In der Gleichung \ref{eq:momentum-1} steht die Variable $v$ für *velocity*, was auf deutsch Geschwindigkeit bedeutet. Die Variable $\mu$ steht für den *momentum co-effizient* und kann als eine Art Reibung interpretiert werden. Der Term $\eta \nabla C$ ist der bereits bekannte Gradient in Relation zum Ausgangsfehler. So wird auf Zeit eine Geschwindigkeit aufgebaut, welche vom Gradienten abhängig ist. Zeigt der Gradient mehrere Iterationen in die gleiche Richtung, erhöht sich die Geschwindigkeit. Diese wird stetig den Gewichten $w$ mit der Gelchung \ref{eq:momentum-2} hinzuaddiert. Der *momentum co-effizient* soll zwischen 0 und 1 liegen, wobei 1 keine Reibung und 0 hohe Reibung bedeutet. Die Reibung soll verhindern, dass die Geschwindigkeit so groß wird, dass der Wert beim Erreichen des Optimums nicht über das Ziel hinaus schießt. [@nielsen_2015, K.3]

*Momentum* kann auch in Kombination mit dem *RMSprop* Verfahren angewendet werden. Der Nutzen ist dabei noch Gegenstand der Forschung. Deswegen wird in der vorliegenden Bachelorarbeit *RMSprop* nicht mit *Momentum* kombiniert [@Tieleman2012].

#### Overfitting \label{head:overfitting}

*Overfitting* entsteht, wenn ein Modell die Trainingsdaten so gut gelernt hat, dass es für genau diesen Datensatz sehr gute Resultate liefert, für einen anderen hingegen wieder signifikant schlechte. Das Modell hat zu wenig generalisiert und zu viele spezielle Details gelernt. Um *Overfitting* zu vermeiden gibt es mehrere Strategien:

- Einen möglichst großen Trainingsdatensatz erstellen
- Ein *kNN* mit weniger Parametern (Neuronen) wählen. Dies sollte nur im Notfall in Betracht gezogen werden.
- Anhand eines vom Trainingsdatensatz ausgegliederten Validationsdatensatzes die Präzision regelmäßig überprüfen und das Training stoppen, wenn diese abnimmt (*early-stopping*)
- L2 Regularisation und Dropout, welche in den Folgekapiteln erläutert werden.

#### L2 Regularisation

Die *L2-Regularisation* versucht dem *Overfitting* entgegenzuwirken, indem es die Kostenfunktion mit einem zusätzlichen Term, den sogenannten *Regularisation-Term*, ergänzt.

\begin{eqnarray} \label{eq:l2}
  C = C_0 + \frac{\lambda}{2n}
  \sum_w w^2,
\end{eqnarray}

Die Gleichung \ref{eq:l2} zeigt diesen Term, der dem Kostenfunktionswert $C_0$ hinzuaddiert wird. Er entspricht der Summe der Quadrate aller Gewichte im Netzwerk. Dieser wird mit dem Faktor $\lambda / 2n$ skaliert. Dabei entspricht $\lambda$ dem *Regularisations-Parameter* und $n$ der Größe vom Trainingsdatensatz.

Eine Interpretation der *L2-Regularisation* ist das Vorziehen von kleinen vor großen Gewichten. Je größer der *Regularisations-Parameter* $\lambda$ gewählt wird, desto eher werden kleine Gewichte bevorzugt. Größere Gewichte werden gleich behandelt. Ist $\lambda$ klein, wird das Minimieren der ursprünglichen Kostenfunktion bevorzugt. Bei dem Wert 0 wird der gesamte *Regularisation-Term* eliminiert.

Eine mögliche Erklärung für das Funktionieren der *L2-Regularisation* lautet, dass kleinere Gewichte eine kleinere Komplexität besitzen und damit eine einfachere, generellere Beschreibung der Daten ermöglichen. Eine andere Annahme geht davon aus, dass durch das Gleichbehandeln von großen Mustern die kleinen oft vorkommenden Muster, welche eher dem generellen Modell entsprechen, bevorzugt werden. Das *kNN* lernt also ein einfacheres, generelleres Modell, welches auch mit komplett neuen Daten gut funktioniert. Beide Aussagen stützen sich vor allem auf empirische Studien und sind keine "sattelfeste" Erklärungen. Es gibt sehr wohl Beispiele, bei welchen ein komplexes Modell das einfachere überbietet. Deswegen darf die *L2-Regularisation* nicht blind angewandt werden. Ebenfalls wurde empirisch nachgewiesen, das die *L2-Regularisation* nicht nur *Overfitting* vorbeugt, sondern auch zu konstanteren Ergebnissen bei mehreren Trainingsgängen führen soll. [@nielsen_2015, K.3]

#### Dropout / Rauswerfen \label{head:dropout}

*Dropout* ist ein weiteres Verfahren dem *Overfitting* entgegenzuwirken. Dazu werden bei jeder Trainingsiteration zufallsbedingt eine definierte Anzahl Neuronen deaktiviert, wodurch sich der Aufbau des *kNN* stetig ändert (Abbildung \ref{dropout}). Dies führt zum Effekt, dass während einem Trainingsdurchlaufs mehrere neuronale Konstellationen vom gleichen *kNN* trainiert werden und ist mit der Idee vergleichbar, mehrere *kNN* gleichzeitig zu trainieren und deren Resultat anschließend zu mitteln.

![Dropout, temporäres deaktivieren zufälliger Neuronen [vgl. @nielsen_2015, K.3] \label{dropout}](images/Dropout.pdf)

Basierend auf der Annahme, dass verschiedene trainierte *kNN* sich alle auf ihre eigene Art überanpassen, werden durch *Dropout* die jeweiligen Resultate gemittelt womit sich die einzelnen Überanpassungen gegenseitig auflösen sollen [@nielsen_2015, K.3].

#### Gewichte und Bias initialisieren

Vor dem Trainieren eines *kNN* müssen die Gewichte und Bias auf einen Startwert gesetzt werden. Standardmäßig wird eine zufällige, unabhängige Gaussverteilung zwischen $0$ und $1$ gewählt. Zu einem besserem Resultat führt die normalisierte Gaussverteilung $W \sim N(0,1)$.

In den letzten Jahren haben mehrere Arbeiten darauf hingewiesen, das durch intelligentes Initialisieren der Gewichte das Lernen erheblich beeinflusst werden kann. Auch ist die Initialisierung von der Aktivierungsfunktion abhängig.

\begin{eqnarray} \label{eq:init-lecun}
  W \sim N(0,1 / \sqrt{n_{\rm in}})
\end{eqnarray}

Die im Jahre 1998 von LeCun et al in [@lecun_backprop] vorgeschlagene Verteilung, dargestellt in der Gleichung \ref{eq:init-lecun}, hat immer noch Bestand und wird oft eingesetzt.

Im Rahmen dieser Bachelorarbeit wird für die *Sigmoid* Aktivierungsfunktion die Methode von Glorot und Benigo aus dem Jahre 2010 verwendet. Für die *Sigmoid* Aktivierungsfunktion gilt folgende Verteilung \ref{eq:init-relu}, welche auch im offiziellen Theano Tutorial zu *kNN* verwendet wird:

\begin{eqnarray} \label{eq:init-relu}
  W \sim N(0,\sqrt{\frac{12}{n_{in} + n_{out}}})
\end{eqnarray}

Für die Aktivierungsfunktion $ReLU$ wird die Initialisierung nach \ref{eq:init_relu} vorgeschlagen. Diese ist identisch zu der der *Sigmoid* Funktion. Es werden aber auch Negativwerte zugelassen.

\begin{eqnarray} \label{eq:init_relu}
  W \sim N(-\sqrt{\frac{12}{n_{in} + n_{out}}},\sqrt{\frac{12}{n_{in} + n_{out}}})
\end{eqnarray}

