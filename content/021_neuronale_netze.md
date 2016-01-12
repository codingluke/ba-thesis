## Künstliche neuronale Netze {#kNN}

### Ursprung

Künstliche neuronale Netze, kurz kNN, sind der Versuch, die aus der Neurowissenschaft bekannten Konzepte der neuronalen Netze, künstlich, mittels Programmcode, nachzubauen. Erste logische Nachahmungen eines organischen Neuron und deren Verknüpfungen zu Netze wurde bereits 1943 von McCulloch und Pitts beschrieben [@mcculloch].

Weiterentwickelt wurde es durch Frank Rosenblatt. Dieser entwickelte vor allem Ende der 1950er Jahre das Konzept des Perzeptron [@rosenblatt1958perceptron].

### Das Perzeptron / Neuron \label{neuron_aufbau}

Das Neuron, auch Perzeptron genannt, wie in Abbildung \ref{perzeptron-ki-norvig} dargestellt, ist das Kernstück der kNN. Das einfachste erdenkliche kNN besteht aus genau einem Neuron (wobei es da streng genommen kein Netz mehr darstellt). Dieses sogenannte Perzeptron und auch die erweiterte, hier verwendete Form, das Sigmoid-Perzeptron, bestehen aus folgenden Bestandteilen:

![Aufbau eines Perzeptron/Neuron [@ki-norvig, S.846] \label{perzeptron-ki-norvig}](images/perzeptron.jpg)

- Eingabeverknüpfungen, besitzen eigene Gewichte
- Eingabefunktion, besitzt einen Bias pro Neuron
- Aktivierungsfunktion, Berechnet Aktivierung anhand der Eingabefunktion
- Ausgabe, Resultat der Aktivierungsfunktion
- Ausgabeverknüpfungen

#### Eingabeverknüpfungen

Die Eingabeverknüpfungen in der Abbildung \ref{perzeptron-ki-norvig} als $a$ angegeben, werden häufig auch als Eingabevektor $X$ bezeichnet. Die Ziffer $a$ wurde gewählt, da der Eingabevektor sich bei einem Netz aus Ausgabeverknüpfungen zusammenstellen kann.

Ein Perzeptron kann über mehrere Eingabeverknüpfungen verfügen, wobei $a_{0}$ immer mit 1 angegeben ist und zusammen mit dem Gewicht $W_{0,j}$ den Bias darstellt.

Jede Eingabeverknüpfung verfügt über eine individuelle Gewichtung. Diese Gewichtung wird mit dem jeweiligen Eingabewert verrechnet.

#### Eingabefunktion

Die Eingabefunktion bestimmt, wie die Eingabewerte mit den Gewichte verrechnet und zusammengefasst werden. Üblicherweise wird dies durch die Multiplikation der einzelnen Eingabewerte mit deren Gewichte und deren anschließenden Aufsummierung getan.

\begin{equation} \label{eq:eingabefunktion}
  in_j = \displaystyle\sum_{i=0}^{n} w_{i,j}*a_i
\end{equation}

#### Aktivierungsfunktion

Die Aktivierungsfunktion definiert den Wertebereich, welcher von dem Neuron ausgegeben wird und beinflusst daduch massgeblich dessen Eigenschaft. Dabei wird die Ausgabe ($in_j$) der Eingabefunktion \ref{eq:eingabefunktion} als Eingebe für die Aktivierungsfunktion verwendet.

\begin{equation} \label{eq:aktivierungsfunktion}
a_j = g(in_j)
\end{equation}

Bei dem einfachen Perzeptron wird eine Schwellenwertfunktion \ref{eq:schwellenwert} verwendet welche den Wert 0 oder 1 zurück gibt. Diese symbolisierten ob das Neuron entweder aktiv 1 oder inaktiv 0 ist. Die Schwellenwertfunktion hat sich in der Praxis als zu grob herausgestellt. Bei kleinen Änderungen am Bias und den Gewichten kann zu einem großen Unterschied des Resultats führen und ist dadurch unberechenbar. [vgl. @nielsen_2015, K.1]

\begin{equation} \label{eq:schwellenwert}
g(in_j) =
  \begin{cases}
    0  & \quad \text{if } in_j \leq \text{ Schwellenwert}\\
    1  & \quad \text{if } in_j > \text{ Schwellenwert}\\
  \end{cases}
\end{equation}

Bei dem Sigmoid-Perzeptron wird als Aktivierungsfunktion die Sigmoid-Funktion \ref{eq:sigmoid} verwendet, diese gibt immer eine Gleitkommazahl zwischen $0$ und $1$ zurück. Somit ist ein Neuron nicht aktiv oder inaktiv, sondern eher aktiv oder eher inaktiv. Eine kleine Änderung am Bias oder an den Gewichten führt auch zu einer kleinen Veränderung des Resultats. Das Sigmoid-Perzeptron ist daher berechenbarer und besser zu trainieren. [vgl. @nielsen_2015, K.1]

\begin{equation} \label{eq:sigmoid}
g(in_j) = \frac{1}{1 + e^{-in_j}}
\end{equation}

Der Tangens Hyperbolicus, hat ähnliche Eigenschaften wie die Sigmoid-Funktion. Der Wertebereich befindet sich jedoch zwischen $-1$ und $1$. Es sind also auch Negativwerte möglich. [vgl. @nielsen_2015, K.6]

Die Rectified Linear Unit (ReLU), zu deutsch Gleichgerichtete Lineare Einheit, auch bekannt als "Rampenfunktion", ist eine Lineare Funktion, welche alle Negativwerte auf $0$ anhebt und die positiven Werte unverändert stehen lässt. Die Rampenfunktion hat sich gegenüber der Sigmoid-Funktion als natürlicher und plausibler erwiesen und erhält immer mehr Beachtung, vor allem im Deep-Learning [@GlorotBB11].

\begin{equation} \label{eq:relu}
  g(in_j) = \max(0, in_j)
\end{equation}

#### Ausgabe und Ausgabeverknüpfungen

Die Ausgabe ist das Resultat der Aktivierungsfunktion ($a_j$). Dieser Wert kann nun als Ausgabeverknüpfung an beliebige weitere Neuronen weitergegeben werden. Aus der Sicht der Empfängerneuronen ist eine Ausgabeverknüpfung eine Stelle im ihrem Eingabevektor, welcher wiederum eine eigene Gewichtung besitzt.

### Das Neuron als logisches Bauteil

Ein Neuron, wie im Kapitel \ref{neuron_aufbau} beschrieben, kann durch die Veränderung der Gewichte der einzelnen Eingangsverknüpfungen und dem Bias jedes beliebiges logisches Bauteil repräsentieren. Ein kNN kann dadurch als eine virtuelle, Schaltplatine angesehen werden, bei welcher die Bauteile dynamisch, durch Anpassung der Gewichte, verändert werden können. Somit kann ein kNN mit einer Anzahl von $N$ Neuronen theoretisch jede mögliche Schaltung mit der gleichen Anzahl Bauteilen darstellen.

![Der logische Addierer (links) dargestellt durch Neuronen (rechts). Die der Bias ist bei allen Neuronen 3 und die Gewichte -2 [@nielsen_2015, K.1] \label{addierer-perceptron}](images/schaltung-knn.png)

### Vom Neuron zum Netz {#kNN_aufbau}

Wie dargestellt in der Abbildung \ref{mlp-generell} bestehen kNN aus Neuronen, welche untereinander zu einem Netz verbunden werden. Dabei wird zwischen den Eingangsneuronen, unsichtbaren Neuronen und Ausgangsneuronen, gekennzeichnet durch die jeweiligen Schichten, unterschieden [vgl. @ki-norvig, S.845].

Die Ausgangsverknüpfungen einer Schicht werden zu Eingangsverknüpfungen der nächsten Schicht. Dadurch ergibt sich ein gerichteten Graphen. KNN die diesem Modell folgen, werden als "feed-forward Netze" bezeichnet. Auf andere Modelle wird im Rahmen dieser Arbeit nicht eingegangen, kNN sind folglich in dieser Bachelorarbeit immer "feed-forward Netze".

![Darstellung eines mehrschichtigen feed-forward kNN [vgl. @nielsen_2015, K.1] \label{mlp-generell}](images/mlp-generell.png)

Ein kNN besitzt immer je eine Schicht von Eingangs- und Ausgangsneuronen. Dazwischen können sich keine, eine oder mehrere Schichten von unsichtbaren Neuronen befinden. Die mittleren Schichten werden "unsichtbar", oder auch "hidden", genannt, da auf diese von außen in der Regel nicht zugegriffen werden kann. Die einzelnen Schichten können eine beliebige Anzahl Neuronen enthalten, in Abbildung \ref{mlp-generell} wird nur eine Möglichkeit aufgezeigt.

Besitzt ein kNN keine unsichtbare Schicht handelt es sich um ein Kernel-Perzeptron. Auf diese wird im Rahmen dieser Bachelorarbeit nicht eingegangen.

Besitzt ein kNN eine unsichtbare Schicht handelt es sich um ein einschichtiges kNN oder auch Perzeptron-Netz genannt. Perzeptron-Netze sind universal, das heißt sie sind theoretisch im Stande jede beliebige Funktion darzustellen.

Besitzt ein kNN mehrere unsichtbare Schichten, spricht man auch von einem mehrschichtigen kNN, im Englischen auch MLP, Multi Layered Perzeptron, genannt. Bei der Verwendung von mehrschichtigen kNN wird häufig von Deep-Learning gesprochen. Deep, da diese eine gewisse Tiefe durch die unsichtbaren Schichten besitzen. [vgl. @ki-norvig, S.846-850]

### KNN als Funktion

Ein kNN kann als Funktion betrachtet werden, wobei der Eingangsvektor $X$ der Funktionsparameter entspricht und der Ausgangsvektor $y$ dem Funktionswert.

\begin{equation} \label{eq:knn-funktion}
  f(X) = y
\end{equation}

Je nach Konfiguration kann das kNN einem Klassifikator, Regressor oder einer beliebigen anderen Funktion entsprechen.

### Trainieren eines kNN

Beim Trainieren eines kNN, wird versucht die Gewichte der einzelnen Eingangsverknüpfungen so zu modifizieren, dass bei einem bestimmten Eingangsvektor $X$ der entsprechende Ausgangsvektor $y$ resultiert.

Um das Resultat zu überprüfen wird eine *Kostenfunktion* verwendet. Die Modifikation der Gewichte wird mehrmals in kleinen Schritten ausgeführt. Dabei versucht man den Kostenfunktionswert zu minimieren.

Um zu wissen in welche Richtung die Gewichte angepasst werden müssen wird das *Gradientenabstiegsverfahren* angewandt.

#### Kostenfunktion

Die Kostenfunktion^[Die Kostenfunktion wird im Englischen als *cost*, oder auch *loss* bezeichnet] berechnet die Abweichung, bzw. den Fehler, vom aktuell berechneten Ausgangswert zum gewünschten Zielwert. Anhand dieser Abweichung wird für jedes Gewicht und jeden Bias modifiziert.

Die klassische und auch von dem Kaggle Wettbewerb [@kaggleDDD] vorgeschriebene Kostenfunktion ist der *Mean Squared Error*, *MSE*. Der *MSE* \ref{eq:mse} berechnet die Distanz von jeder Stelle des Zielvektors mit der entsprechenden Stelle des berechneten Ausgangsvektors. Die einzelnen Distanzen werden quadriert und miteinander aufsummiert. Die Distanzen werden quadriert um dem Fehler ein höheres Gewicht zu geben. Wenn man diese nicht Quadriert, dauert das lernen länger.

\begin{equation} \label{eq:mse}
  MSE(w, b) \equiv \frac{1}{2n} \displaystyle\sum_{X} \|y(X) - a \|^2
\end{equation}

In der Funktion \ref{eq:mse} steht $w$ für die Menge aller Gewichtsvektoren und $b$ für die Biasvektoren. Die Funktion $y(X)$ steht für den Zielvektor und $a$ für den Ausgangsvektor. Die Variable $X$ Steht für eine Menge von Eingangsvektoren wobei die Variable $n$ dessen Größe beinhaltet. Es werden also die Gesamtkosten, aller Trainingsdaten berechnet. [vgl. @nielsen_2015, K.1]

Eine andere Kostenfunktion, die *cross-entropy* Funktion, kommt in modernen kNN immer häufiger zum Einsatz. Sie hat den Vorteil, dass der Lernprozess kontinuierlicher abläuft. Der *MSE* hat hingegen die Eigenschaft am Anfang sehr große Fehler zu finden welche schnell abflachen. [vgl. @nielsen_2015, K.3]

\begin{equation} \label{eq:mse}
  CE(w, b) \equiv -\frac{1}{2n}
  \displaystyle\sum_{X} [y(X)ln(a) + (1 - y(X))ln(1-a)]
\end{equation}

#### Gradientenabstiegsverfahren \label{gradientenabstiegsverfahren}

Beim Gradientenabstiegsverfahren wird die Kostenfunktion mit deren Variablen als Tal angenommen. Die Abbildung \ref{gradientenabstieg} zeigt dies anhand der Funktion $C$ auf der *y-Achse* welche zwei Variablen, $v_1$ auf der *x-Achse* und $v_2$ auf der *z-Achse*, voraussetzt.


![Gradientenabstiegsverfahren im 3D Raum \label{gradientenabstieg}](images/gradientenabstieg.png)

Das Ziel ist es die Kosten zu Minimieren. Dies bedeutet den Punkt zu finden, welcher auf der *y-Achse* den geringsten Wert besitzt. Um diesen Wert zu erreichen, müssen die Variable $v_1$ und $v_2$ auf einen bestimmten Wert gesetzt werden. Um die Brücke zu kNN zu schlagen können die Variablen $v_1$ und $v_2$ als die Gewichte und Biase angesehen werden.

Wird nun einen zufälligen Startwert gewählt, z.B. der Punkt des grünen Balls in der Abbildung \ref{gradientenabstieg}, kann damit der Gradient dieses Punkts berechnet werden. Der Gradient ist die partielle Ableitung der Funktion $C$, wobei die Variablen $v_1$ und $v_2$ nun einen festen Wert besitzen.

Anhand dem Gesetz der partiellen Ableitung, zeigt der Gradient in die Richtung des Maximums. Wird der Gradient invertiert, zeigt der Resultierende Vektor in die Richtung des Minimums. Dieser Vektor ist in der Abbildung \ref{gradientenabstieg} als grüner Pfeil eingezeichnet.

So ist es möglich Schrittweise die Variablen an das Optimum anzugleichen. Am Beispiel der Gewichte und Biase lautet diese Aktualisierungsregel pro Schritt folgendermaßen.

\begin{equation} \label{eq:update_gewichte}
  w \to w'  = w - n
  \frac{\partial C_{X}}{\partial w}
\end{equation}

\begin{equation} \label{eq:update_bias}
  b \to b'  = b - n
  \frac{\partial C_{X}}{\partial b}
\end{equation}

Um das Gradientenabstiegsverfahren auf kleinere Untergruppen der Trainingsdaten, auch *Batches* genannt, anzuwenden, müssen diese Aufsummiert werden.

\begin{equation} \label{eq:update_gewichte}
  w \to w'  = w -\frac{n}{m}
  \displaystyle\sum_{j} \frac{\partial C_{X_j}}{\partial w}
\end{equation}

\begin{equation} \label{eq:update_bias}
  b \to b'  = b -\frac{n}{m}
  \displaystyle\sum_{j} \frac{\partial C_{X_j}}{\partial b}
\end{equation}

#### Backpropagation Algorithmus \label{backprop}

Der Backpropagation Algorithmus wurde ursprünglich im Jahre 1974 von Paul Werbos an der Harvard Universität entwickelt [@backprop]. In der Praxis findet er aber erst seit 1986 durch die Arbeit "Beyond regression: new tools for prediction and analysis in the behavioral sciences" von David Rumelhart, Geoffrey Hilton und Ronald Williams [@RumelhartHintonWIlliams1986] verwendung.

Er löste das Problem der effizienten Gewichtsfindung in den versteckten Schichten. Davor geschah dies auf sehr ineffizienter Weise, welche die anfängliche Euphorie über kNN bis in die 80er Jahre verstummen lies.

Der Algorithmus besteht im wesentlichen aus drei Schritten:

1. **Feed-forward**: Das Eingabemuster wird durch das kNN gleitet.
2. **Ausgangsfehler**: Der Ausgabevektor wird mittels der Kostenfunktion mit dem Zielvektor verglichen und der Fehlervektor daraus abgeleitet.
3. **Rückführung des Fehlers (Backpropagate)**: Der Ausgangsfehler wird nun Schichtweise zurückgeführt. Dadurch erhält jede Schicht einen eigenen Fehlerwert der aber vom Ausgangsfehler beeinflusst wird.

Für diese Schritte werden vier wesentliche Gleichungen benötigt:

**1. Berechnung des Fehlers in der Ausgangsschicht**

\begin{eqnarray} \label{eq:backprop_1}
  \delta^L_j = \frac{\partial C}{\partial a^L_j} g'(in^L_j)
\end{eqnarray}

Die Gleichung \ref{eq:backprop_1} besteht aus zwei Terme. Der erste, linke Term beschreibt wie schnell sich der Fehler $\delta^L_j$, des $j$-ten Neuron der Ausgangsschicht $L$, anhand der Konstenfunktion respektive der dessen Aktivierung $a^L_j$ ändert. Der zweite Term misst, wie schnell sich die Aktivierungsfunktion $g$ durch den Eingabefunktionswert $in^L_j$ ändert. Diese Berechnung muss für jedes Ausgangsneuron gemacht werden. Dafür gibt es die Vektordarstellung $\delta^L = \nabla_{a^L} C \odot \sigma'(in^L)$, wobei $\odot$ das *Hadamard Produkt* darstellt. $\delta^L$ steht somit für einen Vektor aller Fehlern, $a^L$ für alle Aktivierungswerte und $in^L$ für alle Eingabefunktionswerte der Ausgangsschicht.

**2. Berechnung des Fehlervektors $\delta^l$ einer unsichtbaren Schicht $l$ anhand des Fehlervektors der darauffolgenden Schicht $\delta^{l+1}$**

\begin{eqnarray} \label{eq:backprop_2}
  \delta^l = ((w^{l+1})^T \delta^{l+1}) \odot g'(in^l)
\end{eqnarray}

Der rechte Term der Gleichung \ref{eq:backprop_2} schliesst auf die Fehlerdifferenz der Schicht $l$ ausgehend vom Fehlervektors $\delta^{l+1}$ der Folgeschicht und deren aktuellen Gewichtsvektor $w^{l+1}$. Durch das *Hadamard Produkt* wird dieser Term der Änderungsrate der Aktivierungsfunktion $g$ der Schicht $l$ angerechnet und ergibt den angenommenen Fehlervektor $\delta^l$. So wird der Fehler Schichtweise von der Ausgangsschicht auf beliebig viele vorhergehende, unsichtbare Schichten zurückgeführt.

**3. Berechung des Gradienten Kostenfunktion in Reation zu den Biase im Netzwerk**

\begin{eqnarray} \label{eq:backprop_3}
  \frac{\partial C}{\partial b^l_j} = \delta^l_j
\end{eqnarray}

Die Gleichung \ref{eq:backprop_3} zeigt, dass sich die partielle Ableitung, also Änderungsrate, der Kostenfunktion an der Position des $j$-ten Neurons der $l$-ten Schichtrespektive gleich verhält wie der bereits berechnete Fehler $\delta^l_j$

**4. Berechung des Gradienten der Kostenfunktion in Relation zu den Gewichten im Netzwerk**

\begin{eqnarray} \label{eq:backprop_4}
  \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j.
\end{eqnarray}

Die Gleichung \ref{eq:backprop_4} zeigt wie die Änderungsrate, der Gradient, der Kostenfunktion an der Stelle von jedem Neuron in jeder Schicht in relation zu den jeweiligen Gewichten berechnet werden kann. Dafür muss der Aktivierungswert $a^{l-1}_k$ der vorhergehenden Schicht $l-1$ , welcher dem Wert der Eingabeverknüpfung entspricht, mit dem Fehler $\delta^l_j$ der zu berechnenden Schicht $l$ multipliziert werden.

#### Stochastisches Gradientenabstiegsverfahren

Aufbauend auf dem Kapitel \ref{backprop}, welches anhand dem *Backpropagation* Algorithmus aufzeigt wie die Gradienten der Kostenfunktion an den Punkten der einzelnen Neuron berechnet werden kann, ist das *stochastische Gradientenabstiegsverfahren* ein Algorithmus um die Gewichte in Anbetracht der berechneten Gradienten zu modifizieren. Es wird *stochastisch* genannt, da die durch Backpropagation berechneten Gradienten, anhand des Ausgabefehlers eine stochastische Annahme sind.

Die Gewichte werden für jede Schicht der Gleichung $w^l \rightarrow w^l-\frac{\eta}{m} \sum_x \delta^{x,l} (a^{x,l-1})^T$ angepasst. Die Biase mit der Gleichung $b^l \rightarrow b^l-\frac{\eta}{m} \sum_x \delta^{x,l}$.

Verleicht man diese zwei Gleichungen mit den im Kapitel \ref{gradientenabstiegsverfahren} beschriebenen Gleichungen \ref{eq:update_gewichte} und \ref{eq:update_bias}, wird ersichtlich, dass die darin zu berechneten Gradienten nun durch die im Backpropagation algorithmus berechneten Gradienten ausgetauscht werden.

#### RMSProp / Root Mean Square Propagation

Die *RMSProp* wurde von Tijemen Tieleman [@Tieleman2012] vorgeschlagen und von wird von Geoffrez Hinton im Kurs *COURSERA: Neural Networks for Machine Learning* vermittelt. Dieses Verfahren zur Gewichtsmodifikation hat in dieser Bachelorarbeit zu sehr guten Ergebnissen geführt. Leider gibt es keine offizielle Veröffentlichung des Verfahrens.

Das Verfahren erweitert das *stochastische Gradientenabstiegsverfahren* insofern, dass der Gradient aller Gewichte über die Zeit hinweg, durch dem *root-mean-square* gemittelt, mitgeführt, und bei der Modifikation der Gewichte miteinbezogen wird.

\begin{eqnarray} \label{eq:rmsprop_1}
  MeanSquare(w^l_{jk}, t) = 0.9 * MeanSquare(w^l_{jk}, t-1) + 0.1 (G^{lt}_{jk})^2
\end{eqnarray}

Die Gleichung \ref{eq:rmsprop_1} zeigt wie der durch Backpropagation angenommene Gradient $G^{lt}_{jk}$ der Kostenfunktion $C$ in Relation der Gewichte $w^l_{jk}$ rekursiv über die Trainingszeit $t$, im Quadrat gemittelt, mitgeführt wird. Dabei ist der mitgeführte, gemittelte Gradient mit $0.9$ stärker Gewichtet als der Aktuelle.

\begin{eqnarray} \label{eq:rmsprop_2}
  G =  \frac{G^{lt}_{jk}}{\sqrt[2]{MeanSquare(w^l_{jk}, t)}}
\end{eqnarray}

Der für die Gewichtsanpassung Analog der *stochastischen Gradientenabstiegsverfahren* zu Verwendende Gradient $G$ wird berechnet, indem dem durch Backpropagation angenommene Gradient $G^{lt}_{jk}$ durch die Wurzel des mitgeführten, gemittelten Gradienten geteilt wird.

![Vergleich verschiedener Backpropagation Methoden \label{rmsprop-compair}](images/rmsprop_comairation.png)

In der Abbildung \ref{rmsprop-compair} ist sichtbar, dass der Rmsprop (Schwarz) gegenüber dem SGD (Rot) viel schneller das Optimum (Stern) erreicht. SGD mit Momentum (grün), welches als nächstes erläutert wird, schneidet besser ab als der SGD, legt jedoch einen viel weiteren Weg als der Rmsprop zurück. Auf die drei anderen aufgeführten Methoden NAG, Adagrad und Adadelta wird in dieser Bachelorarbeit nicht eingegangen.

#### Momentum

Der Term *Momentum*, übersetzt ins Deutsche mit Schwung, Eigendynamik, Moment oder auch Wucht übersetzt, versucht das Gradientenabstiegsverfahren mit der aus der Physik bekannten Geschwindigkeitszuwachs beim Abstieg zu ergänzen.

\begin{eqnarray}
  v & \rightarrow  & v' = \mu v - \eta \nabla C \label{eq:momentum_1}\\
  w & \rightarrow & w' = w+v'.  \label{eq:momentum_2}
  \end{eqnarray}

In der Gleichung \ref{eq:momentum_1} steht die Variable $v$ für *velocity* zu Deutsch Geschwindigkeit. Die Variable $\mu$ steht für den *momentum co-effizient* und kann als eine Art Reibung interpretiert werden. Der Term $\eta \nabla C$ ist der bereits bekannte Gradient in Relation zum Ausgangsfehler. So wird nun bei auf Zeit eine Geschwindigkeit aufgebaut, welche vom Gradienten abhängig ist. Zeigt der Gradient mehrere Iterationen in die gleiche Richtung, erhöht sich die Geschwindigkeit. Diese wird nun den Gewichten $w$ mit der Gelchung \ref{eq:momentum_2} hinzu addiert. Der *momentum co-effizient* soll zwischen 0 und 1 liegen, wobei 1 keine Reibung und 0 hohe Reibung bedeutet. Die Reibung soll verhindern, dass die Geschwindigkeit so groß wird, dass der Wert beim erreichen des Optimums nicht über das Ziel hinaus schießt. [@nielsen_2015, K.3]

*Momentum* kann auch in Kombination mit dem *RMSprop* Verfahren angewendet werden. Der Nutzen ist dabei noch Gegenstand der Forschung.

#### Overfitting

*Overfitting* passiert, wenn ein Modell die Trainingsdaten so gut gelernt hat, dass es für genau diesen Datensatz sehr gute Resultate liefert, für einen Anderen jedoch wieder super schlechte. Das Modell hat also zu wenig generalisiert und zu viele spezielle Details gemerkt. Um *Overfitting* zu vermeiden gibt es mehrere Strategien:

- Einen Möglichst großen Trainingsdatensatz erstellen
- Ein kNN mit weniger Parameter (Neuronen) wählen. Nur im Notfall!
- Anhand eines vom Trainingsdatensatz ausgegliederten Validationsdatensatz die Präzision regelmäßig überprüfen und das Training stoppen wenn diese abnimmt (*early-stopping*)
- L2 Regularisation welche im folgekapitel erläutert wird (*weight-decay*)
- Dropout

In dieser Bachelorarbeit werden alle Methoden angewandt.

#### L2 Regularisation

Die *l2-Regularisation* versucht dem *Overfitting* entgegenzuwirken, indem es die Kostenfunktion mit einen zusätzlichen Term, den sogenannten *regularisation-term*, ergänzt.

\begin{eqnarray} \label{eq:l2}
  C = C_0 + \frac{\lambda}{2n}
  \sum_w w^2,
\end{eqnarray}

Die Gleichung \ref{eq:l2} Zeigt diesen Term, der dem Kostenfunktionswert $C_0$ hinzu addiert wird. Er entspricht der Summe der Quadraten aller Gewichte im Netzwerk. Dieser wird mit dem Faktor $\lambda / 2n$ skaliert. Dabei entspricht $\lambda$ dem *Regularisations-Parameter* und $n$ der Größe vom Trainingsdatensatz.

Eine Interpretation der L2 Regularisation ist, kleine Gewichte über große Gewichte vor zu ziehen. Je größer der *Regularisations-Parameter* $\lambda$ gewählt wird, desto eher werden kleine Gewichte bevorzugt. Größere Gewichte werden gleich behandelt. Ist der $\lambda$ klein wird das minimieren der original Kostenfunktion bevorzugt. Bei dem Wert 0 wird der gesamte Reglarisations-Term eliminiert.

Eine Aussage wieso die *l2-Regularisation* funktioniert lautet, dass kleinere Gewichte eine kleinere Komplexität besitzen und damit eine einfachere und mächtigere Beschreibung der Daten ermöglichen. Eine andere Aussage lautet. Durch das gleichbehandeln von großen Mustern, werden die kleinen oft vorkommende Muster welche eher dem generellen Modell entsprechen, bevorzugt. Das kNN lernt also ein einfacheres, generelleres Modell, welches auf neuen Daten besser funktioniert.

Beide Aussagen stützen sich vor allem auf empirische Studien und sind keine wasserdichten Erklärungen. Es gibt sehr wohl Beispiele, bei welchen ein Komplexes Modell das einfachere überbietet. Deswegen darf die *l2-Regularisation* nicht blind angewendet werden.

Es ist empirisch ebenfalls bewiesen, das die *l2-Regularisation* nicht nur *Overfitting* vorbeugt sondern auch zu konstanteren Ergebnissen bei mehreren Trainingsgängen erlangt.

#### Dropout / Rauswerfen

*Dropout* ist ein weiteres Verfahren *Overfitting* zu vermeiden. Dabei werden bei jedem Trainingsdurchgang zufallsbedingt eine definierte Anzahl Neuronen deaktiviert. Dies wird in der Abbildung \ref{dropout} visuel dargestellt. Damit verändert sich der Aufbau das kNN. Wird dieser Rauswurf nun bei jedem Trainingsdurchgang (oder *Mini-Batch*) neu initialisiert, hat dies den Effekt, dass mehrere neuronale Konstellationen vom gleichen kNN trainiert werden. Dies kann verglichen werden, als würde man mehrere kNN gleichzeitig lernen und deren Resultat am ende Mitteln.

![Dropout, temporäres deaktivieren zufälliger Neuronen \label{dropout}](images/dropout.png)

Die Idee dahinter ist, dass wenn mehre kNN trainiert werden, sich alle auf ihre eigene Art überanpassen. Werden nun die jeweiligen Resultate gemittelt, heben sich die Einzelnen überanpassungen gegenseitig auf.

#### Gewichte und Bias initialisieren

Vor der Trainieren eines kNN müssen die Gewichte und Biase auf einen Startwert gesetzt werden. Standardmäßig wird eine zufällige, unabhängige Gaussverteilung zwischen $0$ und $1$ gewählt. Zu einem besseres Resultat führt die normalisierte Gaussverteilung $W \sim N(0,1)$.

In den letzen Jahren haben mehrere Arbeiten darauf hingewiesen, das durch intelligentes initialisieren der Gewichte das Lernen erheblich beeinflussen kann. Auch ist die Initialisierung von der Aktivierungsfunktion abhängig.

\begin{eqnarray} \label{eq:init_lecun}
  W \sim N(0,1 / \sqrt{n_{\rm in}})
\end{eqnarray}

Die von 1998 LeCun et al in [@lecun_backprop] vorgeschlagene Verteilung \ref{eq:init_lecun} hat immer noch bestand und wird oft eingesetzt.

In dieser Bacheloarbeit wird für die *Sigmoid* Aktivierungsfunktion die Methode von Glorot und benigo aus dem Jahre 2010 verwendet. Diese besitzt für die *Sigmoid* Aktivierungsfunktion folgende Verteilung. Diese wird auch im offiziellen Theano Tutorial zu kNN verwendet.

\begin{eqnarray} \label{eq:init_relu}
  W \sim N(0,\sqrt{\frac{12}{n_{in} + n_{out}}})
\end{eqnarray}

Für die Aktivierungsfunktion $ReLU$ wird die Initialisierung nach \ref{eq:init_relu} vorgeschlagen. Diese ist identisch zu der der *Sigmoid* Funktion. Es werden aber auch Negativwerte zugelassen.

\begin{eqnarray} \label{eq:init_relu}
  W \sim N(-\sqrt{\frac{12}{n_{in} + n_{out}}},\sqrt{\frac{12}{n_{in} + n_{out}}})
\end{eqnarray}



