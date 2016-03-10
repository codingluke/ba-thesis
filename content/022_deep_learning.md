## Deep-Learning

### Definition

Sobald ein *kNN* über mehr als eine versteckte Schicht verfügt, handelt es sich um ein tiefes, mehrschichtiges *kNN*, welches auch auch *Multi-Layered-Perceptron*, *MLP*, genannt wird. Kommen *MLP* zum Einsatz, wird generell von *Deep-Learning* gesprochen [vgl. @nielsen_2015, K.6].

Das Verwenden von mehren Schichten, kann sich dadurch positiv auf das Resultat auswirken, dass jede weitere Schicht die gelernten Eigenschaften der vorhergehenden wiederverwenden und verfeinern kann. Dies führt zu höheren, abstrakteren Eigenschaften pro zusätzlicher Schicht [vgl. @nielsen_2015, K.6].

Vor allem in der Bild- und Sprachverarbeitung wurden durch *Deep-Learning* neue Standards gesetzt. Google Now, Microsoft's Cortana und auch Apple's Siri verwenden unter anderem alle mehrschichtige *kNN*. [vgl. @deep-learning-commerz]

### Gradientenschwund \label{gradietenschwund}

Das Trainieren von *MLP* mit mehr als drei Schichten, war lange Zeit durch Rechenkapazität beschränkt. Ein großes Problem stellte vor allem der *vanishing-gradient* (deut. *Gradientenschwund*) dar. [vgl. @nielsen_2015, K.6]

Der *Gradientenschwund* findet durch den *Backpropagation-Algorithmus* zwischen den Schichten statt. Beim *Backpropagation-Algorithmus* wird der Fehler von der Ausgangsschicht zur Eingangsschicht zurück geleitet. Nun verliert jedoch der Gradient, der partiell abgeleiteten *Aktivierungsfunktion* jeder vorhergehenden Schicht, an Größe, bis er fast nicht mehr existent ist. Dies hat zur Folge, dass die ersten Schichten eines *MLP* nur sehr langsam lernen, was sich wiederum auf die Trainingszeit (Rechenkapazität) auswirkt. [vgl. @nielsen_2015, K.6]

### Unsupervised-Feature-Learning \label{unsubervised-feature-learning}

Ein weiteres Thema von *Deep-Learning* ist das *Unsupervised-Feature-Learning* (deut. nicht überwachtes Lernen von Eigenschaften). Dabei wird versucht einem *kNN* zu ermöglichen, Eigenschaften in den Eingabedaten von Selbst zu erkennen und zu lernen. [vgl. @Bengio12]

Das *Unsupervised-Feature-Learning* steht dem Gruppieren unbekannter Daten anhand ähnlichen Eigenschaften, *Clustering*, nahe. Bei Beiden werden Zusammenhänge in Trainingsdaten ohne zugewiesene Zieldaten gefunden, wogegen das klassische *Supervised-Feature-Learning* das Resultat mit erlesenen Zieldaten validiert. [vgl. @ki-norvig, S.811]

### Autoencoder

Das Konzept des *Autoencoders* wird von Bengio Yoshua im Artikel "Learning Deep Architectures for AI" [@Bengio09] beschrieben und ist ein Ansatz die Probleme der vorhergehenden Kapitel \ref{gradietenschwund} und \ref{unsubervised-feature-learning} anzugehen.

Der *Autoencoder* ist ein *kNN* mit der Form $X-h-X$, wobei die Eingangs- und Ausgangsschicht (X) dieselbe Anzahl Neuronen besitzen müssen und nur die unsichtbare Schicht (h) eine beliebige Größe besitzen darf.

![Autoencoder: Wiederverwendung der unsichtbaren Schicht [@hodel] \label{fig:autoencoder}](images/Autoencoder.pdf)

Es wird versucht die zu lernenden Daten so in der unsichtbaren Schicht abzubilden (encode), damit sie wieder möglichst gut auf die Ausgangsschicht reproduziert (decode) werden können. Der *Autoencoder* versucht somit die *Identität* zu lernen.

Die gelernten Gewichte aus der unsichtbaren Schicht, bilden die Trainingsdaten in einer abstrakteren Ebene ab und können, wie in Abbildung \ref{fig:autoencoder}, für weitere Netze als Initialgewichte verwendet werden. Trainiert wurden diese Gewichte ohne zusätzliche Zieldaten. Somit handelt es sich um *Unsupervised-Feature-Learning*.

### Denoising-Autoencoder \label{head:dA}

In der Arbeit "Extracting and Composing Robust Features with Denoising Autoencoders" [@VincentPLarochelleH2008] wird beschrieben, dass die Identitätsfunktion noch abstrakter gelernt werden kann, wenn die Trainingsdaten während derer Eingabe zufällig verrauscht werden.

Ein *Denoising-Autoencoder* ist somit ein *Autoencoder*, welcher die Eingangsdaten zuerst zufällig verrauscht, auf die unsichtbare Schicht abbildet und schlussendlich versucht diese bereinigt an der Ausgangsschicht auszugeben.

Das Rauschen veranlasst den *Denoising-Autoencoder* dazu, die wesentlichen Eigenschaften der Daten zu finden. Ansonsten ist es nicht möglich diese zu bereinigen. Ist das Training erfolgreich, wird das "Wesen" der Daten erkannt und die Gewichte und Bias der unsichtbaren Schicht können als Initialgewichte verwendet werden.

### Stacked-denoising-Autoencoder \label{head:stacked-autoencoder}

Der *Stacked-denoising-Autoencoder*, *SdA*, wird in der Arbeit "Stacked denoising autoencoders: learning useful representations in a deep network with a local denoising criterion" [@Vincent10stackeddenoising] beschrieben. Es werden dabei mehrere *Denoising-Autoencoder* so aneinander gekoppelt, dass die unsichtbare Schicht des Vorgehenden zur Eingangsschicht des Nachfolgenden wird (siehe Abbildung \ref{fig:stacked-autoencoder}).

![SdA mit zwei unsichtbaren Schichten [@hodel] \label{fig:stacked-autoencoder}](images/Stacked-Autoencoder.pdf)

Die jeweiligen *Denoising-Autoencoder* werden einzeln, beim Vordersten beginnend, autonom trainiert. So werden die unsichtbaren Schichten nicht miteinander, sondern nacheinander trainiert und umgehen dadurch den *Gradientenschwund*. Die Trainingsdaten werden jeweils von den vorhergehenden Schichten modifiziert, bevor eine nachfolgende Schicht trainiert wird. Wurden alle Schichten einzeln trainiert, wird das gesamte Netz abschließend nochmals "normal" trainiert. Dieser Schritt wird auch *Fine-tunig* genannt.

### ReLU / Rampenfunktion

Beim *Fine-tuning* des Netzes ist das Problem vom *Gradientenschwund* immer noch präsent. Die *Aktivierungsfunktion* *ReLU* hat sich im *Deep-Learning* dadurch durchgesetzt, dass bei ihr der *Gradientenschwund* nicht so ausgeprägt ist wie bei der *Sigmoid-Funktion*. Dies wurde vor allem in der Arbeit "Deep Sparse Rectifier Neural Networks" [@GlorotBB11] beschrieben.
