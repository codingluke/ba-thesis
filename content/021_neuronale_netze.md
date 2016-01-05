## Künstliche neuronale Netze {#kNN}

### Ursprung

Künstliche neuronale Netze, kurz kNN, sind der Versuch, die aus der Neurowissenschaft bekannten Konzepte der neuronalen Netze, künstlich, mittels Programmcode, nachzubauen. Erste logische Nachahmungen eines organischen Neuron und deren Verknüpfungen zu Netze wurde bereits 1943 von McCulloch und Pitts beschrieben [@mcculloch].

Weiterentwickelt wurde es durch Frank Rosenblatt. Dieser entwickelte vor allem Ende der 1950er Jahre das Konzept des Perzeptron [@rosenblatt1958perceptron].

### Das Perzeptron / Neuron

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

$\begin{aligned} \label{test}
in_j = \displaystyle\sum_{i=0}^{n} w_{i,j}*a_i
\end{aligned}$

#### Aktivierungsfunktion

Die Aktivierungsfunktion definiert den Wertebereich, welcher von dem Neuron ausgegeben wird. Dabei wird die Ausgabe der Eingabefunktion als Eingebe für die Aktivierungsfunktion verwendet.

Bei dem einfachen Perzeptron wird eine Schwellenwertfunktion verwendet welche nur die Werte 0 oder 1 zurück gibt. Diese symbolisiert ob das Neuron entweder aktiv 1 oder inaktiv 0 ist. Diese Aktivierungsfunktion hat sich in der Praxis als zu grob herausgestellt. Bei kleinen Änderungen am Bias und den Gewichten kann ein großen Unterschied vom Resultat verursachen.

Bei dem Sigmoid-Perzeptron wird als Aktivierungsfunktion die Sigmoid-Funktion verwendet, diese gibt immer eine Gleitkommazahl zwischen $0$ und $1$ zurück. Somit ist ein Neuron nicht aktiv oder inaktiv, sondern eher aktiv oder eher inaktiv. Eine kleine Änderung am Bias oder an den Gewichten führt auch zu einen kleinen Unterschied beim Resultat. Das Sigmoid-Perzeptron ist daher berechenbarer und besser zu trainieren. [vgl. @nielsen_2015, Kapitel 1]

Der Tangens Hyperbolicus, hat ähnliche Eigenschaften wie die Sigmoid-Funktion. Der Wertebereich befindet sich jedoch zwischen $-1$ und $1$. Es sind also auch Negativwerte möglich. [vgl. @nielsen_2015, Kapitel 6]

Die Rectified Linear Unit (ReLU), zu deutsch Gleichgerichtete Lineare Einheit, auch bekannt als "Rampenfunktion", ist eine Lineare Funktion, welche alle Negativwerte auf $0$ anhebt und die positiven Werte unverändert stehen lässt. Die Rampenfunktion hat sich gegenüber der Sigmoid-Funktion als natürlicher und plausibler erwiesen und erhält immer mehr Beachtung, vor allem im Deep-Learning [@GlorotBB11].

#### Ausgabe und Ausgabeverknüpfungen

Die Ausgabe ist das Resultat der Aktivierungsfunktion. Dieser Wert kann nun als Ausgabeverknüpfung an beliebige weitere Neuronen weitergegeben werden. Aus der Sicht der Empfängerneuronen ist eine Ausgabeverknüpfung eine Stelle im ihrem Eingabevektor, welcher wiederum eine eigene Gewichtung besitzt.

### Aufbau {#kNN_aufbau}

KNN bestehen aus Neuronen, welche untereinander zu einem Netz verbunden werden. Dabei wird zwischen den Eingangsneuronen, unsichtbaren Neuronen und Ausgangsneuronen unterschieden. [vgl. @ki-norvig, S.845]

Ein kNN besitzt immer je eine Schicht von Eingangs- und Ausgangsneuronen. Dazwischen können sich keine, eine oder mehrere Schichten von unsichtbaren Neuronen befinden. Diese mittlere Schichten werden "unsichtbar" genannt, da auf diese von außen in der Regel nicht zugegriffen werden kann.

Besitzt ein kNN keine unsichtbare Schicht handelt es sich um ein Kernel-Perzeptron. Auf diese wird im Rahmen dieser Arbeit nicht eingegangen.

Besitzt ein kNN eine unsichtbare Schicht handelt es sich um ein einschichtiges kNN oder auch Perzeptron-Netz genannt. Perzeptron-Netze sind universal, das heißt sie sind theoretisch im Stande jede beliebige Funktion darzustellen.

Besitzt ein kNN mehrere unsichtbare Schichten, spricht man auch von einem mehrschichtigen kNN. Im Englischen auch MLP, Multi Layered Perzeptron, genannt. Bei der Verwendung von mehrschichtigen kNN wird auch häufig von Deep-Learning gesprochen. Deep, da diese eine gewisse tiefe durch die unsichtbaren Schichten besitzen. [vgl. @ki-norvig, S.846-850]


#### Eingabefunktion

#### Aktivierungsfunktion

#### Ausgabe




### Gradienten abstiegsverfahren

### Kostenfunktion

### Backpropagation Algorithmus

### Gute Startwerte finden

