## Deep-Learning

### Definition

Sobald ein *kNN* über mehr als eine versteckte Schicht verfügt, handelt es sich um ein tiefes, mehrschichtiges *kNN*, welches auch auch *Multi-Layered-Perceptron*, *MLP*, genannt wird. Kommen *MLP* zum Einsatz, wird generell von *Deep-Learning* gesprochen.

Das Verwenden von mehren Schichten, kann sich dadurch positiv auf das Resultat auswirken, dass jede weitere Schicht die gelernten Eigenschaften der vorhergehenden wiederverwenden und verfeinern kann. Dies führt zu höheren, abstrakteren Eigenschaften pro zusätzlicher Schicht.

Vor allem in der Bild- und Sprachverarbeitung wurden durch *Deep-Learning* neue Standards gesetzt. Google Now, Microsoft's Cortana und auch Apple's Siri verwenden unter anderem alle mehrschichtige *kNN*. [@deep-learning-commerz]

### Gradientenschwund \label{gradietenschwund}

Das Trainieren von *MLP* mit mehr als drei Schichten, war lange Zeit durch Rechenkapazität beschränkt. Ein großes Problem stellte vor allem der *vanishing-gradient* (deut. *Gradientenschwund*) dar.

Der *Gradientenschwund* findet durch den *Backpropagation Algorithmus* zwischen den Schichten statt. Beim *Backpropagation Algorithmus* wird der Fehler von der Ausgangsschicht zur Eingangsschicht zurück geleitet. Nun verliert der Gradient der partiell abgeleiteten *Aktivierungsfunktion* jeder vorhergehenden Schicht an Größe, bis er fast nicht mehr existent ist. Dies hat zur Folge, dass die ersten Schichten in einem *MLP* sehr langsam bis gar nicht lernen.

### Unsupervised Feature Learning \label{unsubervised-feature-learning}

Ein weiteres Thema von *Deep-Learning* ist das *Unsupervised-Feature-Learning*, nicht beaufsichtigtes Lernen von Eigenschaften. Es wird versucht ein *kNN* dazu veranlassen Eigenschaften in den Eingabedaten selber zu erkennen und lernen.

Das *Unsupervised-Feature-Learning* ist mit dem Gruppieren unbekannter Daten anhand ähnlichen Eigenschaften (*Clustering*) zu vergleichen. Es werden Zusammenhänge neuer Daten ohne Zieldaten gefunden, wobei beim *Supervised-Feature-Learning* das Resultat mit erlesenen Zieldaten validiert wird.

### Autoencoder

Das Konzept des *Autoencoder* wird von Bengio Yoshua im Artikel "Learning Deep Architectures for AI" [@Bengio09] beschrieben und ist ein Ansatz die Probleme der vorhergehenden Kapitel \ref{gradietenschwund} und \ref{unsubervised-feature-learning} zu lösen.

Der *Autoencoder* ist ein *kNN* mit der Form $X-h-X$, wobei die Eingangs- und Ausgangsschicht (X) dieselbe Anzahl Neuronen besitzen und die unsichtbare Schicht (h) eine beliebige Größe haben kann.

![Autoencoder: Wiederverwendung der unsichtbaren Schicht [Hodel] \label{fig:autoencoder}](images/Autoencoder.pdf)

Es wird versucht die zu lernenden Daten so in der unsichtbaren Schicht abzubilden (encode), damit sie wieder möglichst gut auf die Ausgangsschicht reproduziert (decode) werden können. Der *Autoencoder* versucht somit die *Identitätsfunktion* zu lernen.

Die gelernten Gewichte aus der unsichtbaren Schicht, bilden die Trainingsdaten in einer abstrakteren Ebene ab und können, wie in Abbildung \ref{fig:autoencoder}, für weitere Netze als Initialgewichte verwendet werden. Trainiert wurden diese Gewichte ohne zusätzliche Zieldaten. Somit handelt es sich um *Unsupervised-Feature-Learning*.

### Denoising-Autoencoder

In der Arbeit "Extracting and Composing Robust Features with Denoising Autoencoders" [@VincentPLarochelleH2008] wird beschrieben, dass die Identitätsfunktion noch abstrakter gelernt werden kann, wenn die Trainingsdaten bei der Eingabe zufällig verrauscht werden.

Ein *Denoising-Autoencoder* ist somit ein *Autoencoder*, welcher die Eingangsdaten zuerst zufällig verrauscht, auf die unsichtbare Schicht abbildet und schlussendlich versucht diese bereinigt an der Ausgangsschicht auszugeben.

Das Rauschen veranlasst den *Denoising-Autoencoder* dazu, die wesentlichen Eigenschaften der Daten zu finden. Ansonsten ist es nicht möglich diese zu bereinigen. Ist das Training erfolgreich, wird das "Wesen" der Daten erkannt und die Gewichte und Bias der unsichtbaren Schicht können als Initialgewichte verwendet werden.

### Stacked-Denoising-Autoencoder \label{head:stacked-autoencoder}

Der *Stacked-Denoising-Autoencoder*, *SdA*, ist die Kombination von mehreren Autoencodern zu einem "Stapel" und soll dem *Gradientenschwund* entgegenwirken. Es werden mehrere *Autoencoder* so aneinander gekoppelt, dass die unsichtbare Schicht des Vorgehenden zur Eingangsschicht des Nachfolgenden wird.

![SdA mit zwei unsichtbaren Schichten [Hodel] \label{fig:stacked-autoencoder}](images/Stacked-Autoencoder.pdf)

Die jeweiligen *Autoencoder* werden einzeln, beim vordersten beginnend, autonom trainiert. So werden die unsichtbaren Schichten nicht alle miteinander, sondern nacheinander trainiert und umgehen den *Gradientenschwund*.

Sind alle Schichten fertig trainiert, wird das gesamte Netz nochmals, *"supervised"* nachtrainiert (*Fine-tunig*).

### ReLU / Rampenfunktion

Beim *Fine-tuning* des Netzes ist das Problem vom *Gradientenschwund* immer noch vorhanden. Die *Aktivierungsfunktion* *ReLU* hat sich im *Deep-Learning* dadurch durchgesetzt, dass bei ihr der *Gradientenschwund* nicht so ausgeprägt ist wie bei der *Sigmoid-Funktion*. [@GlorotBB11]

