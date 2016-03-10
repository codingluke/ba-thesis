# Einleitung \label{head:einleitung}

## Motivation \label{head:motivation}

Es existieren nach wie vor immense Wissensbestände, welche die Menschheit seit der Erfindung der Schrift niedergeschrieben hat, in reiner Papierform. Diese Art der Wissenskonservierung hat sich über Jahrtausende bewährt. Ein großer Nachteil dieses Mediums liegt jedoch in der, durch manuelles Lesen vorgegebenen, ineffizienten Durchsuchbarkeit.

Neue Arten der Wissensspeicherung in Form von digitaler Schrift bieten den Vorteil das Wissen, durch dessen Volltext-Indexierung in Kombination mit modernen Suchalgorithmen, effizient durchsuchbar zu machen. Um diese Errungenschaft ebenfalls auf das in alten Schriften festgehaltene Wissen anwenden zu können, müssen diese digitalisiert werden.

Durch die jahrelange Archivierung sind solche Schriftstücke oft nicht mehr mühelos lesbar. Dazu können Staub, Luftfeuchtigkeit, Sonneneinstrahlung sowie sonstiger, unangemessener Umgang beigetragen haben. Dadurch verursachte Verunreinigungen können wiederum dazu beitragen, dass beim Digitalisieren diverse Buchstaben und Wörter nicht korrekt erkannt werden.

An diesem Punkt soll die vorliegende Arbeit ansetzen. Mit Hilfe künstlicher neuronaler Netze wird nach einer Möglichkeit gesucht, eingescannte Schriften zu bereinigen, um die Präzision der späteren Schrifterkennung zu erhöhen.

## Zielstellung \label{head:zielstellung}

Das Hauptziel dieser Bachelorarbeit ist ein künstliches neuronales Netzwerk, *kNN*, zu entwerfen, implementieren und trainieren, welches möglichst gut im *Kaggle*-Wettbewerb "Denoising Dirty Documents" [@kaggleDDD] abschneidet. Im Wettbewerb geht es darum verrauschte Bilder automatisch so zu bereinigen, dass das darin enthaltene Schriftbild optimal hervorgehoben wird.

Dazu wird auf verschiedene Architekturmodelle für *kNN* zurückgegriffen, welche untereinander verglichen werden. Somit ist ein weiteres Ziel der Arbeit die Stärken und Schwächen der unterschiedlichen Architekturen aufzuzeigen. Vor allem wird der Unterschied von einschichtigen *kNN* zu mehrschichtigen, tiefen *kNN* untersucht. Bei den mehrschichtigen *kNN* wird ein besonderes Augenmerk auf die in der Arbeit "Stacked denoising autoencoders: learning useful representations in a deep network with a local denoising criterion" [@Vincent10stackeddenoising] beschriebenen Methode der *Stacked-denoising-Autoencoder* gelegt.

Eine weitere Zielsetzung ist es, einen Prozess zur komfortablen Hyperparametersuche und Analyse des Lernvorgans herauszuarbeiten. Dieses Verfahren soll sogleich für das Lernen und Analysieren der verschiedenen *kNN* dieser Bachelorarbeit mit angewendet werden.

## Aufbau der Arbeit \label{head:aufbau}

Die Arbeit ist in neun aufeinander aufbauende Hauptkategorien unterteilt. Der Einleitung folgt ein Kapitel über die Grundlagen der in der Arbeit verwendeten Techniken. Nachdem die Grundlagen erörtert wurden, wird in Kapitel \ref{head:aufgabenstellung} auf das Ziel und die Randbedingungen der Arbeit, sowie auf alternative Ansätze eingegangen. Fortführend wird in Kapitel \ref{head:analyse} nach dafür benötigte Prozesse und Umsetzungsvarianten gesucht, welche in Kapitel \ref{head:implementierung} in der Programmiersprache *Python* implementiert werden. Die Implementierung wird durch Softwaretests verifiziert, welche unter dem Kapitel \ref{head:tdd} genauer erläutert werden. Darauf aufbauend wird in Kapitel \ref{head:anwendung} aufgezeigt, wie die implementierten Module kombiniert, zu verschiedenen *kNN* konfiguriert, trainiert und anschließend verwendet werden können. Mit Hilfe der bisherigen Erörterungen wird im Kapitel \ref{head:evaluierung} versucht ein möglichst generelles *kNN* zu konfigurieren und trainieren. Die Resultate werden miteinander verglichen und ausgewertet. Abschließend wird in Kapitel \ref{head:fazit} eine kritische Retrospektive der Arbeit niedergeschrieben und einen Ausblick auf weiterführende Möglichkeiten gegeben.

