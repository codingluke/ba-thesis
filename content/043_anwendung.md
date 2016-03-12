# Anwendung \label{head:anwendung}

## Konfigurieren und Trainieren

Die drei implementierten Module *network.py*, *metric.py* und *preprocessor.py* können wie im Codeblock \ref{lst:training-sda} dargestellt, für das Trainieren eines *kNN* kombiniert werden.

~~~~~~~{#lst:training-sda .python caption="Konfigurieren und Trainieren eines SdA"}
from network import Network, FullyConnectedLayer as FCL,
                    AutoencoderLayer as Ae
from metric import MetricRecorder
from preprocessor import BatchProcessor

mr = MetricRecorder('/pfad/zur/config.json')
td = BatchProcessor(X_dirpath='...', y_dirpath='...', border=2,
                    random=True, random_mode='fully', batchsize=10000)
pd = BatchProcessor(X_dirpath='...', y_dirpath='...', border=2,
                    random=True, random_mode='fully', batchsize=10000)
tv = BatchProcessor(X_dirpath='...', y_dirpath='...', border=2,
                    batchsize=10000)
in = (2*2+1)**2 == 25 # Anz Pixel der Subbilder
net = Network([Ae(n_in=in, n_hidden=199), corruption_level=0.14),
               Ae(n_in=199, n_hidden=81, corruption_level=0.14)
               FC(n_in=199, n_out=1), mbs=500])
net.pretrain_autoencoders(tdata=pd, metric_recorder=mr, mbs=500,
          eta=0.025, save_dir='./models/pretrain_', epochs=10)
net.train(tdata=td, vdata=vd, metric_recorder=mr, mbs=500,
          eta=0.045, eta_min=0.01, algorithm='rmsprop',
          save_dir='./models/train_', lambda=0.0, epochs=15)
~~~~~~~

Für die Train-, Pretrain- und Validierungsdaten wird je eine *BatchProcessor*-Instanz erstellt (*td*, *pd* und *vd*). Bei der Erstellung wird mitgegeben, wie viele Nachbarpixel berücksichtigt werden sollen und wie die *Batchgröße* pro Iteration sein soll. Den *BatchProcessor*en für die Trainings- sowie Pretaindaten sollten zusätzlich der Parameter *random* auf *True* gesetzt werden. Damit wird sichergestellt, dass die Trainingsdaten durchmischt werden. Der *BatchProcessor* für die Validierungsdaten darf die Daten nicht mischen, dies benötigt unnötig Zeit.

Zusätzlich wird eine *MetricRecorder*-Instanz, *mr*, erstellt. Mit Hilfe des *MetricRecorder*s wird beim Training der Verlauf aufgezeichnet. Beim Instanziieren muss eine Konfigurationsdatei angegeben werden, welche einen beliebigen Experimentname sowie die Datenbankverbindung beinhaltet.

Als nächstes werden beliebig viele Klasseninstanzen, vom Basistyp *Layer*, mit beliebiger Anzahl Ein- und Ausgänge erstellt. Es stehen die Typen *AutoencoderLayer* und *FullyConnectedLayer* zur Verfügung. Hier muss beachtet werden, dass die Anzahl Ausgänge des Vorgängers mit der Anzahl Eingänge der Folgeschicht übereinstimmen. Diese Schichtinstanzen werden bei der darauffolgenden Instanziierung der Klasse *Network* als Liste mitgegeben. Die *AutoencoderLayer* Klasse kann zusätzlich der Parameter *corruption_level*, Maß der Verunreinigung, mitgegeben werden.

Mit der Methode *pretrain_autoencoders* können nun die *AutoencoderLayer* im Voraus trainiert werden. Hier muss als Trainingsmenge *tdata* die *BatchProcessor*-Instanz, *pd*, welche zum Ordner mit den Pretraindaten zeigt, mitgegeben werden.

Das eigentliche Training, oder *fine-tuning*, wird durch die Methode *train* der *Network*-Instanz *net* gestartet. Dieser werden die Trainings- und Validierungsdaten, sowie die *Recorder*-Instanz und die Hyperparameter mit übergeben.

Ist das Training zu Ende, werden die Validierungskosten der besten Validierung ausgegeben. Das als Datei abgespeicherte beste Modell, kann später von der Klasse *Cleaner* geladen und verwendet werden.

## Trainieren mit Spearmint

Um mit *Spearmint* automatische Hyperparametersuche durchzuführen, muss sich das Netzwerk in einer *Python*-Datei innerhalb einer Methode mit der Signatur *main(job_id, params)* befinden (siehe Codebeispiel \ref{lst:spearmint}). Der Parameter *job_id* beinhaltet die automatisch zugewiesene Identität als Integer. Der Parameter *params* ist vom Datentyp *dict* welcher die von *Spearmint* ausgewählten Werten der zu überprüfenden *Hyperparameter*, beinhaltet. Zurückgeben muss die Methode das Resultat, hier die niedrigsten Validierungskosten.

*Spearmint* wird durch das Aufrufen der Datei *main.py* im *Spearmint* Projektordner gestartet. Der Pfad zum Ordner Trainingsdatei wird als Parameter mitgegeben. In dem Ordner wird nach der Datei *config.json* gesucht. Innerhalb der Datei *config.json* werden die gewünschten *Hyperparameter* deklariert, sowie der Name der Trainingsdatei, die Datenbankverbindung und ein Experimentname. Jedes *Spearmint*-Experiment befindet sich somit in einem eigenen Ordner. In dieser Arbeit befinden sich diese als Unterordner des Ordners */trainings*.

~~~~~~~{#lst:spearmint .python caption="Minimalsetup zum Trainineren mit Spearmint."}
from network import Network,
from network import FullyConnectedLayer as FC
from metric import MetricRecorder
from preprocessor import BatchProcessor

def main(job_id, params):
  mr = MetricRecorder('/pfad/zur/config.json', job_id=job_id)
  train_data, valid_data = BatchProcessor(...), BatchProcessor(...)
  net = Network([FC(n_in=params['in1'], n_out=params['out1']),
                 FC(n_in=params['out1'], n_out=1)])
  return net.train(tdata=train_data, vdata=valid_data,
                   metric_recorder=mr, params[...])
~~~~~~~


## Verwenden eines gespeicherten Modells

Das die während dem Trainieren gespeicherten Modelle, können mit der Klasse *Cleaner*, wie in Codebeispiel \ref{lst:clean}, geladen werden. Die *Cleaner*-Instanz (*c*) bietet nun die Methoden *clean_and_show* und *clean_and_save* zur Verfügung, mit welchen beliebige Bilder bereinigt werden können.

~~~~~~~{#lst:clean .python caption="Bereinigen eines einzelnen Bildes"}
from cleaner import Cleaner
c = Cleaner('/pfad/zur/modell/datei.pkl')
c.clean_and_show('/pfad/zum/verrauschten/bild.png')
c.clean_and_save(img_path='/pfad/zum/verrauschten/bild.png',
                 savepath='/pfad/zum/bereinigten/bild.png')
~~~~~~~

Mit den Methoden *clean_and_save*, sowie *clean_for_submission* der Klasse *BatchCleaner* kann, wie in Codebeispiel \ref{lst:batch-clean} dargestellt, direkt ein ganzer Ordner bereinigt werden.

~~~~~~~{#lst:batch-clean .python caption="Bereinigen eines gesamten Ordners"}
from cleaner import BatchCleaner
c = BatchCleaner(dirty_dir='/pfad/zum/eingabe/ordner/',
                 model_path='/pfad/zur/modell/datei.pkl')
c.clean_and_save(output_dir='/pfad/zum/ausgabe/ordner')
c.clean_for_submission(output_dir='/pfad/zum/ausgabe/ordner')
~~~~~~~
