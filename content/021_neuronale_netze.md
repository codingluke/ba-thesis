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

#### Gradientenabstiegsverfahren

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

Um das Gradientenabstiegsverfahren auf kleinere Untergruppen der Trainingsdaten, auch *Baches* genannt, anzuwenden, müssen diese Aufsummiert werden.

\begin{equation} \label{eq:update_gewichte}
  w \to w'  = w -\frac{n}{m}
  \displaystyle\sum_{j} \frac{\partial C_{X_j}}{\partial w}
\end{equation}

\begin{equation} \label{eq:update_bias}
  b \to b'  = b -\frac{n}{m}
  \displaystyle\sum_{j} \frac{\partial C_{X_j}}{\partial b}
\end{equation}

#### Backpropagation Algorithmus



#### Gute Startwerte finden
