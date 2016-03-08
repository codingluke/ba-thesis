# Einleitung \label{head:einleitung}

## Motivation \label{head:motivation}

Es existieren immer noch immense Wissensbestände, welche die Menschheit seit der Erfindung der Schrift niedergeschrieben hat, in reiner Papierform. Diese Art der Wissenskonservierung hat sich über Jahrtausende bewährt. Ein großer Nachteil dieses Mediums liegt jedoch im effizienten Durchsuchen.

Neue Arten der Wissensspeicherung in Form von digitaler Schrift bieten den Vorteil das Wissen durch dessen Volltext-Indexierung effizient durchsuchbar zu machen. Um diese Errungenschaft auch auf alte Schriften anwenden zu können, müssen diese digitalisiert werden.

Durch die jahrelange Archivierung sind die Schriften oft nicht mehr gut lesbar. Dazu können Staub, Luftfeuchtigkeit, Sonneneinstrahlung oder ungeschickter Umgang geführt haben. Diese Verunreinigung kann dazu führen, dass beim Digitalisieren diverse Buchstaben und Wörter nicht korrekt erkannt werden.

An diesem Punkt soll die vorliegende Arbeit ansetzen. Mit Hilfe von künstlichen neuronalen Netzen wird versucht, eine Möglichkeit zu erarbeiten, eingescannte Schriften vor der eigentlichen Schrifterkennung zu bereinigen, um das Endresultat der Schrifterkennung zu verbessern.

## Zielstellung \label{head:zielstellung}

Das Hauptziel dieser Bachelorarbeit ist ein künstliches neuronales Netzwerk, *kNN*, zu entwerfen, implementieren und
trainieren, welches möglichst gut im Kaggle Wettbewerb *Denoising Dirty Documents* [@kaggleDDD] abschneidet. Im Wettbewerb geht es darum verrauschte Bilder automatisch so zu bereinigen, dass das darin enthaltene Schriftbild optimal hervorgehoben wird.

Es wird auf verschiedene Architekturmodelle für *kNN* zurückgegriffen, welche untereinander verglichen werden. Somit ist ein weiteres Ziel der Arbeit die Stärken und Schwächen der unterschiedlichen Architekturen aufzuzeigen. Vor allem wird der Unterschied von einschichtigen *kNN* zu mehrschichtigen, tiefen *kNN* untersucht. Bei den mehrschichtigen *kNN* wird ein besonderes Augenmerk auf die Methode der geschichteten *Denoising-Autoencoder* [@Bengio12] gelegt.

Eine weitere Zielsetzung ist es, einen Prozess zur komfortablen Hyperparametersuche und Analyse des Lernvorgans herauszuarbeiten. Dieses Verfahren soll direkt für das Lernen und Analysieren der verschiedenen *kNN* dieser Bachelorarbeit angewendet werden.

## Aufbau der Arbeit \label{head:aufbau}

Die Arbeit ist in neun aufeinander aufbauende Hauptkategorien unterteilt. Der Einleitung folgt ein Kapitel über die Grundlagen der in der Arbeit verwendeten Techniken.

Nachdem die Grundlagen erörtert wurden, wird in der Aufgabenstellung darauf eingegangen, was genau das Ziel und die Randbedingungen der Arbeit sind und welche alternativen Ansätze existieren.

Darauf folgt die Analyse, in welcher die benötigten Prozesse und Umsetzungsvarianten gesucht und aufgezeigt werden. Die gefundenen Prozesse werden danach im Kapitel Implementierung in der Programmiersprache *Python* umgesetzt. Die Implementierung wird durch Softwaretests verifiziert, welche unter dem Kapitel Test Driven Development genauer erläutert werden. In Kapitel Anwendung wird darauf eingegangen, wie die implementierten Module kombiniert, zu verschiedenen *kNN* konfiguriert, trainiert und anschließend verwendet werden können.

Mit Hilfe des zuvor implementierten Programms wird im Kapitel Evaluierung versucht ein möglichst gutes *kNN* zu konfigurieren und trainieren. Die Resultate werden miteinander verglichen und ausgewertet.

Abschließend wird im Fazit eine kritische Retrospektive der Arbeit niedergeschrieben und einen Ausblick auf weitere Möglichkeiten gegeben.


