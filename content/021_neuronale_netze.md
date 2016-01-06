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

Bei dem einfachen Perzeptron wird eine Schwellenwertfunktion \ref{eq:schwellenwert} verwendet welche den Wert 0 oder 1 zurück gibt. Diese symbolisierten ob das Neuron entweder aktiv 1 oder inaktiv 0 ist. Die Schwellenwertfunktion hat sich in der Praxis als zu grob herausgestellt. Bei kleinen Änderungen am Bias und den Gewichten kann zu einem großen Unterschied des Resultats führen und ist dadurch unberechenbar. [vgl. @nielsen_2015, Kapitel 1]

\begin{equation} \label{eq:schwellenwert}
g(in_j) =
  \begin{cases}
    0  & \quad \text{if } in_j \leq \text{ Schwellenwert}\\
    1  & \quad \text{if } in_j > \text{ Schwellenwert}\\
  \end{cases}
\end{equation}

Bei dem Sigmoid-Perzeptron wird als Aktivierungsfunktion die Sigmoid-Funktion \ref{eq:sigmoid} verwendet, diese gibt immer eine Gleitkommazahl zwischen $0$ und $1$ zurück. Somit ist ein Neuron nicht aktiv oder inaktiv, sondern eher aktiv oder eher inaktiv. Eine kleine Änderung am Bias oder an den Gewichten führt auch zu einer kleinen Veränderung des Resultats. Das Sigmoid-Perzeptron ist daher berechenbarer und besser zu trainieren. [vgl. @nielsen_2015, Kapitel 1]

\begin{equation} \label{eq:sigmoid}
g(in_j) = \frac{1}{1 + e^{-in_j}}
\end{equation}

Der Tangens Hyperbolicus, hat ähnliche Eigenschaften wie die Sigmoid-Funktion. Der Wertebereich befindet sich jedoch zwischen $-1$ und $1$. Es sind also auch Negativwerte möglich. [vgl. @nielsen_2015, Kapitel 6]

Die Rectified Linear Unit (ReLU), zu deutsch Gleichgerichtete Lineare Einheit, auch bekannt als "Rampenfunktion", ist eine Lineare Funktion, welche alle Negativwerte auf $0$ anhebt und die positiven Werte unverändert stehen lässt. Die Rampenfunktion hat sich gegenüber der Sigmoid-Funktion als natürlicher und plausibler erwiesen und erhält immer mehr Beachtung, vor allem im Deep-Learning [@GlorotBB11].

\begin{equation} \label{eq:relu}
  g(in_j) = \max(0, in_j)
\end{equation}

#### Ausgabe und Ausgabeverknüpfungen

Die Ausgabe ist das Resultat der Aktivierungsfunktion ($a_j$). Dieser Wert kann nun als Ausgabeverknüpfung an beliebige weitere Neuronen weitergegeben werden. Aus der Sicht der Empfängerneuronen ist eine Ausgabeverknüpfung eine Stelle im ihrem Eingabevektor, welcher wiederum eine eigene Gewichtung besitzt.

### Das Neuron als logisches Bauteil

Ein Neuron, wie im Kapitel \ref{neuron_aufbau} beschrieben, kann durch die Veränderung der Gewichte der einzelnen Eingangsverknüpfungen und dem Bias jedes beliebiges logisches Bauteil repräsentieren. Ein kNN kann dadurch als eine virtuelle, Schaltplatine angesehen werden, bei welcher die Bauteile dynamisch, durch Anpassung der Gewichte, verändert werden können. Somit kann ein kNN mit einer Anzahl von $N$ Neuronen theoretisch jede mögliche Schaltung mit der gleichen Anzahl Bauteilen darstellen.

![Der logische Addierer (links) dargestellt durch Neuronen (rechts). Die der Bias ist bei allen Neuronen 3 und die Gewichte -2 [@nielsen_2015, Kapitel 1] \label{addierer-perceptron}](images/schaltung-knn.png)

### Vom Neuron zum Netz {#kNN_aufbau}

KNN bestehen aus Neuronen, welche untereinander zu einem Netz verbunden werden, dargestellt in der Abbildung \ref{mlp-generell}. Dabei wird zwischen den Eingangsneuronen, unsichtbaren Neuronen und Ausgangsneuronen, gekennzeichnet durch die jeweiligen Schichten, unterschieden [vgl. @ki-norvig, S.845].

![Darstellung eines mehrschichtigen feed-forward kNN [vgl. @nielsen_2015, Kapitel 1] \label{mlp-generell}](images/mlp-generell.png)

Ein kNN besitzt immer je eine Schicht von Eingangs- und Ausgangsneuronen. Dazwischen können sich keine, eine oder mehrere Schichten von unsichtbaren Neuronen befinden. Die mittleren Schichten werden "unsichtbar", oder auch "hidden", genannt, da auf diese von außen in der Regel nicht zugegriffen werden kann. Die einzelnen schichten können eine beliebige Anzahl Neuronen enthalten, in Abbildung \ref{mlp-generell} wird nur eine Möglichkeit aufgezeigt.

Besitzt ein kNN keine unsichtbare Schicht handelt es sich um ein Kernel-Perzeptron. Auf diese wird im Rahmen dieser Arbeit nicht eingegangen.

Besitzt ein kNN eine unsichtbare Schicht handelt es sich um ein einschichtiges kNN oder auch Perzeptron-Netz genannt. Perzeptron-Netze sind universal, das heißt sie sind theoretisch im Stande jede beliebige Funktion darzustellen.

Besitzt ein kNN mehrere unsichtbare Schichten, spricht man auch von einem mehrschichtigen kNN. Im Englischen auch MLP, Multi Layered Perzeptron, genannt. Bei der Verwendung von mehrschichtigen kNN wird auch häufig von Deep-Learning gesprochen. Deep, da diese eine gewisse tiefe durch die unsichtbaren Schichten besitzen. [vgl. @ki-norvig, S.846-850]

### KNN als Funktion

### Gradienten abstiegsverfahren

### Kostenfunktion

### Backpropagation Algorithmus

### Gute Startwerte finden

