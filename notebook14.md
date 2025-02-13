---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 14. Neuronale Netze

Durch neuronale Netze, die tief verschachtelt sind (= tiefe neuronale Netze =
deep neural network), gab es im Bereich des maschinellen Lernens einen
Durchbruch. Neuronale Netze sind eine Technik aus der Statistik, die bereits in
den 1940er Jahren entwickelt wurde. Seit ca. 10 Jahren erleben diese Techniken
verbunden mit Fortschritten in der Computertechnologie eine Renaissance.

Neuronale Netze bzw. Deep Learning kommt vor allem da zum Einsatz, wo es kaum
systematisches Wissen gibt. Damit neuronale Netze erfolgreich trainiert werden
können, brauchen sie sehr große Datenmengen. Nachdem in den letzten 15 Jahren
mit dem Aufkommen von Smartphones die Daten im Bereich Videos und Fotos massiv
zugenommen haben, lohnt sich der Einsatz der neuronalen Netze fúr
Spracherkennung, Gesichtserkennung oder Texterkennung besonders.

Beispielsweise hat ein junges deutsches Start-Up-Unternehmen 2017 aus einem
neuronalen Netz zum Übersetzen Englisch <-> Deutsch eine Webanwendung entwickelt
und ins Internet gestellt, die meinen Alltag massiv beeinflusst: DeepL.com,
siehe

> [https://www.deepl.com/en/blog/how-does-deepl-work](https://www.deepl.com/en/blog/how-does-deepl-work)

Mittlerweile beherrscht DeepL 23 Sprachen, siehe auch den Wikipedia-Artikel zu
DeepL:

> [https://de.wikipedia.org/wiki/DeepL](https://de.wikipedia.org/wiki/DeepL)

## 14.1 Mehrschichtiges Perzeptron

### Lernziele

* Sie wissen, was ein **Multilayer-Perzeptron** (MLP), also ein mehrschichtiges
  Perzeptron, ist.
* Sie können den Begriff **Deep Learning** erklären.
* Sie können mit Scikit-Learn ein neuronales Netz trainieren.

### Viele Perzeptronen sind ein neuronales Netz

In einem vorhergehenden Kapitel haben wir das Perzeptron, ein künstliches Neuron
kennengelernt. Schematisch können wir es folgendermaßen darstellen:

![Schematische Darstellung eines Perzeptrons](https://gramschs.github.io/book_ml4ing/_images/perceptron.svg)

Jedes Eingangssignal wird mit einem Gewicht multipliziert. Anschließend werden
die gewichteten Eingangssignale summiert. Übersteigt die gewichtete Summe einen
Schwellenwert, feuert sozusagen das künstliche Neuron. Das Ausgabesignal wird
aktiviert.

Mathematisch gesehen, wurde nach dem Bilden der gewichteten Summe die
Heaviside-Funktion angewendet. Im Kapitel über die logistische Regression haben
wir bereits gelernt, dass auch andere Funktionen zum Einsatz kommen können. Bei
der logistischen Regression wird beispielsweise die Sigmoid-Funktion verwendet.
Bei neuronalen Netzen sind insbesondere die
[ReLU-Funktion](https://de.wikipedia.org/wiki/Rectifier_(neuronale_Netzwerke))
(rectified linear unit)

![ReLU-Funktion](https://gramschs.github.io/book_ml4ing/_images/plot_relu_function.svg)

und der [Tangens hyperbolicus](https://de.wikipedia.org/wiki/Tangens_hyperbolicus_und_Kotangens_hyperbolicus)

![Tangens hyperbolicus](https://gramschs.github.io/book_ml4ing/_images/plot_tanh_function.svg)

häufig eingesetzte Aktivierungsfunktionen.

Oft werden diese beiden Schritte -- Bilden der gewichteten Summe und Anwenden
der Aktivierungsfunktion -- in einem Symbol gemeinsam dargestellt, wie in der
folgenden Abbildung zu sehen.

![Vereinfachte schematische Darstellung eines Perzeptrons](https://gramschs.github.io/book_ml4ing/_images/neuron.svg)

Tatsächlich sind sogar häufig Darstellungen verbreitet, bei denen nur noch durch
die Kreise das Perzeptron oder das künstliche Neuron symbolisiert wird.

![Symbolbild eines Perzeptrons bzw. eines künstlichen Neurons](https://gramschs.github.io/book_ml4ing/_images/neuron_symbolisch.svg)

Die Idee des mehrschichtigen Perzeptrons ist es, eine oder mehrere
Zwischenschichten einzuführen. In dem folgenden Beispiel wird eine
Zwischenschichtmit zwei Neuronen eingeführt:

![Ein mehrschichtiges Perzeptron (Mulitilayer Perceptron)](https://gramschs.github.io/book_ml4ing/_images/MLP_1layer_2neurons.svg)

Es können beliebig viele Zwischenschichten eingeführt werden. Jede neue
Zwischenschicht kann dabei unterschiedliche Anzahlen von Neuronen enthalten.
Insgesamt nennen wir die so entstehende Rechenvorschrift **mehrschichtiges
Perzeptron** oder **Multilayer Perceptron** oder **neuronales Netz**.

### Viele Schichten = Deep Learning

Bei neuronalen Netzen werden viele Schichten mit vielen Neuronen in die
Rechenvorschrift einbezogen. Das führt dazu, dass vor allem sogenannte tiefe
neuronale Netze, also solche mit vielen Schichten, extrem leistungsfähig sind.
Umgekehrt benötigen neuronale Netze aber auch eine große Anzahl an
Trainingsdaten mit guter Qualität.

Die Firma Linguee verfügte genau über solche Deutsch-Englisch-Übersetzungen.
2017 trainierten Mitarbeiter dieses Unternehmens auf Basis dieser Übersetzungen
ein neuronales Netz, das die bisher dahin existierenden Übersetzungsdienste von
beispielsweise Google Translate bei Weitem übertraf. 2022 wurde das daraus
gegründete Start-Up DeepL zum sogenannten Einhorn, also zu einem Start-Up, das
mit mehr als 1 Milliarde Dollar bewertet wird (siehe
[Artikel](https://www.faz.net/aktuell/wirtschaft/deepl-der-online-uebersetzungsdienst-wird-zum-einhorn-18467883.html)).

## 14.2 Neuronale Netze mit Scikit-Learn

### Lernziele Kapitel 14.2

* Sie können mit Scikit-Learn ein neuronales Netz zur Klassifikation trainieren.

### Neuronale Netze zur Klassifikation

Schauen wir uns an, wie das Training eines tiefen neuronalen Netzes in
Scikit-Learn funktioniert. Dazu benutzen wir aus dem Untermodul
``sklearn.neural_network`` den ``MLPClassifier``, also ein
Multi-Layer-Perzeptron für Klassifikationsaufgaben:

> [https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)

Wir benutzen künstliche Daten, um die Anwendung des MLPClassifiers zu
demonstrieren.

```{code-cell} ipython3
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_circles


# Generiere künstliche Daten
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

# Konvertierung in ein DataFrame-Objekt für Plotly Express
df = pd.DataFrame({
    'Feature 1': X[:, 0],
    'Feature 2': X[:, 1],
    'Category': pd.Series(y, dtype='category')
})

# Visualisierung
fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Category',
                 title='Künstliche Daten')
fig.show()
```

```{code-cell} ipython3
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Auswahl des Models
# solver = 'lbfgs' für kleine Datenmengen, solver = 'adam' für große Datenmengen, eher ab 10000
# hidden_layer: Anzahl der Neuronen pro verdeckte Schicht und Anzahl der verdeckten Schichten
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[5, 5])

# Split Trainings- / Testdaten
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

# Training
model.fit(X_train, y_train)

# Validierung 
score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)
print(f'Score für Trainingsdaten: {score_train:.2f}')
print(f'Score für Testdaten: {score_test:.2f}')
```

Funktioniert gar nicht mal schlecht :-) Wir zeichen die Entscheidungsgrenzen
ein, um zu sehen, wo das neuronale Netz die Trennlinien zieht.

```{code-cell} ipython3
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles

# Generate synthetic data
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

# Create grid for contour plot
gridX, gridY = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
gridZ = model.predict_proba(np.column_stack([gridX.ravel(), gridY.ravel()]))[:, 1]
Z = gridZ.reshape(gridX.shape)

# Create scatter plot
scatter = go.Scatter(x=df['Feature 1'], y=df['Feature 2'], mode='markers',
                     marker=dict(color=df['Category'], colorscale='BlueRed_r'))

# Create contour plot
contour = go.Contour(x=np.linspace(-1.5, 1.5, 50), y=np.linspace(-1.5, 1.5, 50), z=Z, 
                     opacity=0.2, colorscale='BlueRed_r')

# Create figure and add plots
fig = go.Figure()
fig.add_trace(contour)
fig.add_trace(scatter)
fig.update_layout(title='Künstliche Messdaten und Konturen des Modells',
                  xaxis_title='Feature 1',
                  yaxis_title='Feature 2')
fig.show()
```

Im Folgenden wollen wir uns ansehen, welche Bedeutung die optionalen Parameter
haben. Dazu zunächst noch einmal der komplette Code, aber ohne einen Split in
Trainings- und Testdaten. Probieren Sie nun unterschiedliche Werte für die
Architektur der verdeckten Schicht 'hidden_layer_sizes' aus.

```{code-cell} ipython3
# setze verschiedene Werte für die Architektur der verdeckten Schicht
my_hidden_layers = [10,10]

# erzeuge künstliche Daten
X,y = make_circles(noise=0.2, factor=0.5, random_state=1)

# Auswahl des Model
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=my_hidden_layers)

# Training und Validierung
model.fit(X, y)
print('Score: {:.2f}'.format(model.score(X, y)))

# Visualisierung
# Create grid for contour plot
gridX, gridY = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
gridZ = model.predict_proba(np.column_stack([gridX.ravel(), gridY.ravel()]))[:, 1]
Z = gridZ.reshape(gridX.shape)

# Create scatter plot
scatter = go.Scatter(x=df['Feature 1'], y=df['Feature 2'], mode='markers',
                     marker=dict(color=df['Category'], colorscale='BlueRed_r'))

# Create contour plot
contour = go.Contour(x=np.linspace(-1.5, 1.5, 50), y=np.linspace(-1.5, 1.5, 50), z=Z, 
                     opacity=0.2, colorscale='BlueRed_r')

# Create figure and add plots
fig = go.Figure()
fig.add_trace(contour)
fig.add_trace(scatter)
fig.update_layout(title='Künstliche Messdaten und Konturen des Modells',
                  xaxis_title='Feature 1',
                  yaxis_title='Feature 2')
fig.show()
```

Wie Sie sehen, ist es schwierig, eine gute Architektur des neuronalen Netzes (=
Anzahl der Neuronen pro verdeckter Schicht und Anzahl verdeckter Schichten) zu
finden. Auch fällt das Ergebnis jedesmal ein wenig anders aus, weil
stochastische Verfahren im Hintergrund für das Trainieren der Gewichte benutzt
werden. Aus diesem Grund sollten neuronale Netze nur eingesetzt werden, wenn
sehr große Datenmengen vorliegen und dann noch ist das Finden der besten
Architektur eine große Herausforderung.

### Zusammenfassung und Ausblick Kapitel 14.2

Das Training eines neuronalen Netzes erfordert sehr viele Daten und
Hyperparameter-Tuning. In dieser Vorlesung haben wir nur das Prinzip der
neuronalen Netze kennengelernt. Weiterführende Details gehören in eine
eigenständige Vorlesung »Deep Learning«.
