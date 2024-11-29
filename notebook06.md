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

# 6. Entscheidungsbäume (Decision Trees)

## 6.1 Was ist ein Entscheidungsbaum?

Ein beliebtes Partyspiel ist das Spiel "Wer bin ich?". Die Spielregel sind
simpel. Eine Person wählt eine berühmte Person oder eine Figur aus einem Film
oder Buch, die die Mitspieler:innen erraten müssen. Die Mitspieler:innen
dürfen jedoch nur Fragen stellen, die mit "Ja" oder "Nein" beantwortet werden.

Hier ist ein Beispiel, wie eine typische Runde von "Wer bin ich?" ablaufen
könnte:

* Spieler 1: Bin ich männlich?
* Spieler 2: Ja.
* Spieler 3: Bist du ein Schauspieler?
* Spieler 1: Nein.
* Spieler 4: Bist du ein Musiker?
* Spieler 1: Ja.
* Spieler 5: Bist du Michael Jackson?
* Spieler 1: Ja! Richtig!

Als nächstes wäre jetzt Spieler 5 an der Reihe, sich eine Person oder Figur
auszuwählen, die die anderen Spieler erraten sollen. Vielleicht kennen Sie auch
die umgekehrte Variante. Der Name der zu ratenden Person/Figur wird der Person
mit einem Zettel auf die Stirn geklebt. Und nun muss die Person raten, während
die Mitspieler:innen mit Ja/Nein antworten.

Dieser Partyklassiker lässt sich auch auf das maschinelle Lernen übertragen.

### Lernziele Kapitel 6.1

* Sie wissen, was ein **Entscheidungsbaum (Decision Tree)** ist.
* Sie kennen die Bestandteile eines Entscheidungsbaumes:
  * Wurzelknoten (Root Node)
  * Knoten (Node)
  * Zweig oder Kante (Branch)
  * Blatt (Leaf)
* Sie können einen Entscheidungsbaum mit Scikit-Learn trainieren.
* Sie können mit Hilfe eines Entscheidungsbaumes Prognosen treffen.

### Ein Entscheidungsbaum im Autohaus

Ein **Entscheidungsbaum** gehört zu den überwachten Lernverfahren (Supervised
Learning). Es ist auch üblich, die englische Bezeichnung **Decision Tree**
anstatt des deutschen Begriffes zu nutzen. Ein großer Vorteil von
Entscheidungsbäumen ist ihre Flexibilität, denn sie können sowohl für
Klassifikations- als auch Regressionsaufgaben eingesetzt werden. Im Folgenden
betrachten wir als Beispiel eine Klassifikationsaufgabe. In einem Autohaus
vereinbaren zehn Personen eine Probefahrt. In der folgenden Tabelle ist notiert,
welchen

* `Kilometerstand [in km]` und
* `Preis [in EUR]`

das jeweilige Auto hat. In der dritten Spalte `verkauft` ist vermerkt, ob das
Auto nach der Probefahrt verkauft wurde (`True`) oder nicht (`False`). Diese
Information ist die Zielgröße. Die Tabelle mit den Daten lässt sich effizient
mit einem Pandas-DataFrame organisieren:

```{code-cell}
import pandas as pd 

daten = pd.DataFrame({
    'Kilometerstand [km]': [32908, 20328, 13285, 17162, 27449, 13715, 32889,  3111, 15607, 18295],
    'Preis [EUR]': [15960, 20495, 17227, 17851, 5428, 22772, 13581, 16793, 23253, 11382],
    'verkauft': [False, True, False, True, False, True, False, True, True, False],
    },
    index=['Auto 1', 'Auto 2', 'Auto 3', 'Auto 4', 'Auto 5', 'Auto 6', 'Auto 7', 'Auto 8', 'Auto 9', 'Auto 10'])
daten.head(10)
```

Da in unserem Beispiel von den Autos nur die beiden Eigenschaften
`Kilometerstand [km]` und `Preis [EUR]` erfasst wurden, können wir die
Datenpunkte anschaulich in einem zweidimensionalen Streudiagramm (Scatterplot)
visualisieren. Dabei wird der Kilometerstand auf der x-Achse und der Preis auf
der y-Achse abgetragen. Die Zielgröße `verkauft` kennzeichnen wir durch die
Farbe. Dabei steht die Farbe Rot für »verkauft« (True) und Blau für »nicht
verkauft« (False).

```{code-cell}
import plotly.express as px

fig = px.scatter(daten, x = 'Kilometerstand [km]', y = 'Preis [EUR]', 
                 color='verkauft', title='Künstliche Daten: Verkaufsaktion im Autohaus')
fig.show()
```

Als nächstes zeigen wir, wie die Autos anhand von Fragen in die beiden Klassen
»verkauft« und »nicht verkauft« sortiert werden können. Im Streudiagramm
visualisieren wir die Autos mit ihren Eigenschaften `Kilometerstand [km]` und
`Preis [EUR]` als Punkte. Dazu passend werden wir schrittweise den
Entscheidungsbaum entwickeln. Ein Entscheidungsbaum visualisiert
Entscheidungsregeln in Form einer Baumstruktur. Zu Beginn wurde noch keine Frage
gestellt und alle Autos befinden sich gemeinsam in einem **Knoten** (Node) des
Entscheidungsbaumes, der visuell durch einen rechteckigen Kasten symbolisiert
wird. Dieser erste Knoten wird als **Wurzelknoten** (Root Node) bezeichnet, da
er die Wurzel des Entscheidungsbaumes darstellt.

Dann wird eine erste Frage gestellt. *Ist der Verkaufspreis kleiner oder gleich
16376.50 EUR?* Entsprechend dieser Entscheidung werden die Autos in zwei Gruppen
aufgeteilt. Wenn ja, wandern die Autos nach links und ansonsten nach rechts. Im
Entscheidungsbaum wird diese Aufteilung durch einen **Zweig** (Branch) nach
links und einen Zweig nach rechts symbolisiert. Ein alternativer Name für Zweig
ist **Kante**. Die Autos »rutschen« die Zweige/Kanten entlang und landen in zwei
separaten Knoten. Im Streudiagramm (Scatterplot) entspricht diese Fragestellung
dem Vergleich mit einer horizontalen Linie bei y = 16376.5. Da alle Autos mit
einem Verkaufspreis kleiner/gleich 16376.5 EUR blau sind, also »nicht verkauft«
wurden, wird im Streudiagramm (Scatterplot) alles unterhalb der horizontalen
Linie blau eingefärbt.

Bei den Autos mit einem Preis kleiner oder gleich 16376.50 EUR müssen wir nicht
weiter sortieren bzw. weitere Fragen stellen. Da aus diesem Knoten keine Zweige
mehr wachsen, wird dieser Knoten auch **Blatt** (Leaf) genannt. Aber in dem
Knoten des rechten Zweiges befinden sich fünf rote (also verkaufte) Autos und
ein blaues (also nicht verkauftes) Auto. Wir wollen diese Autos durch weitere
Fragen sortieren. Doch obwohl nur ein Auto (nämlich Auto 3) aus dieser Gruppe
separiert werden soll, ist dies nicht durch nur eine einzige Frage möglich.
Lautet die Frage: »Ist der Preis kleiner oder gleich 17300 EUR?«, dann wandern
das rote Auto 8 und das blaue Auto 3 nach links. Wählen wir die Frage: »Ist der
Kilometerstand kleiner oder gleich 13500 km?«, dann wandern ebenfalls Auto 3 und
Auto 8 nach links. Beide Fragen sind also gleichwertig, welches sollen wir
nehmen? Wir gehen nach der Reihenfolge der Eigenschaften vor. Da der
Kilometerstand in der Tabelle in der ersten Spalte steht und der Preis in der
zweiten Spalte, entscheiden wir uns für die Frage nach dem Kilometerstand: *»Ist
der Kilometerstand kleiner oder gleich 13500 km?«* Alternativ könnten wir auch
den Zufall entscheiden lassen.

Im Streudiagramm (Scatterplot) wird die noch nicht eingefärbte Fläche rechts der
vertikalen Linie 13500 km rot gefärbt. Im linken Knoten (Node) sind aber nur
noch zwei Autos, so dass diesmal eine weitere Frage ausreicht, die beiden Autos
in zwei Klassen zu sortieren. Wir fragen: *»Ist der Kilometerstand kleiner oder
gleich 8198 km?«*

Alle Autos sind nun durch die Fragen sortiert und befinden sich in Blättern
(Leaves). Im Streudiagramm (Scatterplot) wird dieser Zustand kenntlich gemacht,
indem auch die letzte verbleibende Fläche (oberhalb eines Preises von 16376.50
EUR) links von Kilometerstand 8198 km rot und rechts davon blau eingefärbt wird.

> Was ist ... ein Entscheidungsbaum?
Ein Entscheidungsbaum (Decision Tree) ist ein Modell zur Entscheidungsfindung,
das Daten mit Hilfe einer Baumstruktur sortiert. Die Datenobjekte starten beim
Wurzelknoten (= Ausgangssituation) und werden dann über Knoten (=
Entscheidungsfrage) und Zweige/Kanten (= Ergebnis der Entscheidung) in Blätter
(= Endzustand des Entscheidungsprozesses) sortiert.

### Entscheidungsbäume mit Scikit-Learn trainieren

In der Praxis verwenden wir die ML-Bibliothek Scikit-Learn, um einen
Entscheidungsbaum zu trainieren. Das Modul [Scikit-Learn →
Tree](https://scikit-learn.org/stable/modules/tree.html) stellt sowohl einen
Entscheidungsbaum-Algorithmus für Klassifikationsprobleme als auch einen
Algorithmus für Regressionsprobleme zur Verfügung. Für das obige Beispiel
Autohaus importieren wir den Algorithmus für Klassifikationsprobleme namens
`DecisionTreeClassifier`:

```{code-cell}
from sklearn.tree import DecisionTreeClassifier
```

Dann erzeugen wir ein noch untrainiertes Entscheidungsbaum-Modell und weisen es
der Variable `modell` zu:

```{code-cell}
modell = DecisionTreeClassifier()
```

Bei der Erzeugung könnten wir noch verschiedene Optionen einstellen, die in der
[Dokumentation Scikit-Learn →
DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
nachgelesen werden können. Zunächst belassen wir es aber bei den
Standardeinstellungen.

Als nächstes adaptieren wir die Daten aus dem Pandas-DataFrame so, dass das
Entscheidungsbaum-Modell trainiert werden kann. Der `DecisionTreeClassifier`
erwartet für das Training zwei Argumente. Als erstes Argument müssen die
Eingabedaten übergeben werden, also die Eigenschaften der Autos. Als zweites
Argument erwartet der `DecisionTreeClassifier` die Zielgröße, also den Status
»nicht verkauft« oder »verkauft«. Wir trennen daher den Pandas-DataFrame `daten`
auf und verwenden die Bezeichnung `X` für die Eingabedaten und `y` für die
Zielgröße.

```{code-cell}
X = daten[['Kilometerstand [km]', 'Preis [EUR]']]
y = daten['verkauft']
```

Als nächstes wird der Entscheidungsbaum trainiert. Dazu wird die Methode
`.fit()` mit den beiden Argumenten `X` und `y` aufgerufen.

```{code-cell}
modell.fit(X,y)
```

Jetzt ist zwar der Entscheidungsbaum trainiert, doch wir sehen nichts. Als
erstes überprüfen wir mit der Methode `.score()`, wie gut die Prognose des
Entscheidungsbaumes ist.

```{code-cell}
score = modell.score(X,y)
print(score)
```

Eine 1 steht für 100 %, also alle 10 Autos werden korrekt klassifiziert. Dazu
hat der `DecisionTreeClassifier` basierend auf den Eingabedaten `X` eine
Prognose erstellt und diese Prognose mit den echten Daten in `y` verglichen. Für
die Trainingsdaten funktioniert der Entscheidungsbaum also perfekt. Ob der
Entscheidungsbaum ein neues, elftes Auto korrekt klassifizieren würde, kann so
erst einmal nicht entschieden werden.

### Prognosen mit Entscheidungsbäumen treffen

Soll für neue Autos eine Prognose abgegeben werden, ob sie sich eher verkaufen
lassen oder nicht, müssen die neuen Daten die gleiche Struktur wie die
Eingangsdaten haben. Wir erzeugen daher einen neuen Pandas-DataFrame, bei dem
die erste Eigenschaft der Kilometerstand der neuen Autos ist und die zweite
Eigenschaft ihr Preis.

```{code-cell}
neue_autos = pd.DataFrame({
    'Kilometerstand [km]': [7580, 11300, 20000],
    'Preis [EUR]': [20999, 12000, 14999]
    },
    index=['Auto 11', 'Auto 12', 'Auto 13']) 
```

Mit Hilfe der `predict()`-Methode kann dann der Entscheidungsbaum
prognostizieren, ob die Autos verkauft werden oder nicht.

```{code-cell}
:tags: [remove-input]
modell = DecisionTreeClassifier(random_state=0);
modell.fit(X,y);
```

```{code-cell}
prognose = modell.predict(neue_autos)
print(prognose)
```

Um für ein neues Auto eine Prognose abzugeben, werden zunächst den Blättern
Klassen zugeordnet. Sind alle Blätter **rein**, d.h. befinden sich nur Autos
einer einzigen Klasse in einem Blatt, dann bekommt das Blatt diese Klasse
zugeordnet. Ist ein Blatt nicht rein, sondern enthält noch Autos mit
unterschiedlichen Klassen »verkauft« oder »nicht verkauft« so wird diesem Blatt
diejenige Klasse zugeordnet, die am häufigsten auftritt. Um diese Idee zu
visualisieren, färben wir im Entscheidungsbaum die Blätter entsprechend rot und
blau ein.

Jedes neue Auto durchläuft jetzt die Entscheidungen, bis es in einem Blatt
angekommen ist. Die Klasse des Blattes ist dann die Prognose für dieses Auto.

Der Entscheidungsbaum prognostiziert, dass Auto 11 und Auto 12 nicht verkauft
werden, aber Auto 13 könnte verkaufbar sein.

## Zusammenfassung und Ausblick Kapitel 6.1

In diesem Kapitel haben Sie den Entscheidungsbaum (Decision Tree) anhand einer
Klassifikationsaufgabe kennengelernt. Mit Hilfe von Scikit-Learn wurde ein
Entscheidungsbaum trainiert und dazu benutzt, eine Prognose für neue Daten
abzugeben. Im nächsten Kapitel werden wir uns damit beschäftigen, weitere
Einstellmöglichkeiten beim Training des Entscheidungsbaumes zu nutzen und
Entscheidungsbäume durch Scikit-Learn visualisieren zu lassen.

## 6.2 Entscheidungsbäume visualisieren und trainieren

Im letzten Kapitel haben wir gelernt, wie mit Scikit-Learn ein Entscheidungsbaum
für binäre Klassifikationsaufgaben trainiert wird. In diesem Kapitel werden wir
uns damit beschäftigen, den trainierten Entscheidungsbaum von Scikit-Learn
visualisieren zu lassen. Darüber hinaus lernen wir, was das
Gini-Impurity-Kriterion ist und welche weiteren Einstellmöglichkeiten es für
Entscheidungsbäume in Scikit-Learn gibt.

### Lernziele Kapitel 6.2

* Sie können einen Entscheidungsbaum mit `plot_tree` visualisieren.
* Sie wissen, was die Angaben `samples` und `value` bei der Visualisierung des
  Entscheidungsbaumes bedeuten.
* Sie wissen, was das **Gini-Impurity-Kriterium** ist.
* Sie kennen weitere Parameter für Entscheidungsbäume wie `random_state=` oder
  `criterion=`.

### Entscheidungsbäume visualisieren

Im letzten Kapitel haben wir den Entscheidungsbaum für das Autohaus mit Hilfe
des Moduls Scikit-Learn trainiert. Scikit-Learn bietet in dem Untermodul
`sklearn.tree` nicht nur Algorithmen für Entscheidungsbäume an, sondern auch ein
dazu passendes Visualisierungswerkzeug. Die Funktion `plot_tree` zeichnet den
Entscheidungsbaum. Um diese Funktion auszuprobieren, wird zunächst der Datensatz
mit den Autodaten erneut geladen, das Modell Entscheidungsbaum gewählt und
anschließend trainiert.

```{code-cell} ipython3
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

# Sammlung der Daten 
daten = pd.DataFrame({
    'Kilometerstand [km]': [32908, 20328, 13285, 17162, 27449, 13715, 32889,  3111, 15607, 18295],
    'Preis [EUR]': [15960, 20495, 17227, 17851, 5428, 22772, 13581, 16793, 23253, 11382],
    'verkauft': [False, True, False, True, False, True, False, True, True, False],
    },
    index=['Auto 1', 'Auto 2', 'Auto 3', 'Auto 4', 'Auto 5', 'Auto 6', 'Auto 7', 'Auto 8', 'Auto 9', 'Auto 10'])
daten.head(10)

# Auswahl des Modells: Entscheidungsbaum für Klassifikation
modell = DecisionTreeClassifier(random_state=0)

# Adaption der Daten
X = daten[['Kilometerstand [km]', 'Preis [EUR]']]
y = daten['verkauft']

# Training des Modells
modell.fit(X,y)
```

Nun können wir die Funktion `plot_tree` importieren und das trainierte Modell
visualisieren lassen.

```{code-cell} ipython3
from sklearn.tree import plot_tree

plot_tree(modell)
```

`plot_tree` produziert eine Textausgabe und ein Diagramm. Die Textausgabe kann
unterdrückt werden, indem hinter den Funktionsaufruf `plot_tree(modell)` ein
Semikolon `;` gesetzt wird. Das Diagramm zeichnet wie erwartet die Baumstruktur
vom Wurzelknoten über die Knoten und Zweige bis hin zu den Blättern. Die
Entscheidungsfragen stehen in der erste Zeile der Knoten. Danach folgen weitere
Angaben wie `gini`, `samples` und `value`. Um diese Angaben zu erklären,
ergänzen wir zunächst weitere Angaben. Mit der Option `feature_names=` wird eine
Liste mit den Eigenschaften ergänzt, die Option `class_names=` ergänzt die
Klassenbezeichnugnen. So erhalten wir folgendes Diagramm:

```{code-cell} ipython3
plot_tree(modell, 
    feature_names=['Kilometerstand [km]', 'Preis [EUR]'],
    class_names=['nicht verkauft', 'verkauft']);
```

Was `gini` bedeuten könnte, erschließt sich so immer noch nicht, aber die
Angaben `samples` und `values` können so leichter von ihrer Bedeutung her
eingeordnet werden. `samples` gibt die Anzahl der Datenobjekte an, die sich in
diesem Knoten befinden. `values` listet auf, wie viele Datenobjekte die
Zielgröße `nicht verkauft` (= False bzw. 0) haben und wie viele zu der Klasse
`verkauft` (= True bzw. 1) gehören.

Weitere Details zu den Optionen der `plot_tree`-Funktion finden Sie in der
[Dokumentation Scikit-Learn →
plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html).

Als nächstes widmen wir uns der Bedeutung von `gini`.

### Was ist das Gini-Impurity-Kriterium?

Das Gini-Impurity-Kriterium ist ein Maß für die Unreinheit eines Datensatzes.
Beim Beispiel mit dem Autohaus sind im Wurzelknoten fünf Autos, die nicht
verkauft wurden, und fünf verkaufte Autos. Bei zwei Klassen ist das die maximale
Unreinheit, die auftreten kann. Der Anteil der verkauften Autos ist genau 50 %.
Diesem prozentualen Anteil wird das Gini-Impurity-Kriterium von 0.5 zugeordnet.
Es gibt zwei weitere Extremfälle. Entweder sind nur verkaufte Autos im Datensatz
(100 % verkaufte Autos) oder gar keine verkaufte Autos (0 % verkaufte Autos). In
beiden Fällen ist der Datensatz rein, das Gini-Impurity-Kriterium ist 0. In
allen anderen Fällen liegt das Gini-Impurity-Kriterium zwischen 0 und 0.5. Die
Formel zur Berechnung des genauen Wertes des Gini-Impurity-Kriteriums lautet

$$\text{GI} = 1 - p^2 - (1-p)^2,$$

wenn $p$ der prozentuale Anteil der verkauften Autos ist (das gilt natürlich
allgemein für binäre Klassifikationsaufgaben und nicht nur das
Autohaus-Beispiel).

Die folgende Abbildung zeigt die konkreten Werte des Gini-Impurity-Kriteriums
für den prozentualen Anteil an verkauften Autos.

```{code-cell} ipython3
from numpy import linspace

p = linspace(0,1)
gini = 1 - p**2 - (1-p)**2

import plotly.express as px

fig = px.line(x = p, y = gini,
        title='Gini-Impurity-Kriterium',
        labels={'x': 'prozentualer Anteil', 'y': 'Wert des Gini-Impurity-Kriteriums'})
fig.show()
```

Im Diagramm können wir direkt ablesen, dass bei einem nicht verkauften Auto und
fünf verkauften Autos ($p = 0.8\bar{3}$) das Gini-Impurity-Kriterium den Wert
$0.27\bar{7} \approx 0.278$ hat.

Das Gini-Impurity-Kriterium ist sehr wichtig für das Training eines
Entscheidungsbaumes. Der Algorithmus probiert im Hintergrund verschiedene
Möglichkeiten durch, mit Hilfe der Entscheidungsfragen den Datensatz zu
splitten. Zu jedem Split werden dann die zugehörigen Werte des
Gini-Impurity-Kriteriums berechnet. Dann wählt der Algorithmus den Split aus,
der die höchste Reinheit hat (also den niedrigsten Gini-Impurity-Wert). Gilt das
für mehrere Splits, dann wird zufällig ein Split ausgewählt.  

Neben dem Gini-Impurity-Kriterium gibt es noch weitere Bewertungsmaße, um einen
Entscheidungsbaum zu trainieren. In Scikit-Learn sind die beiden Alternativen
`log_less` und `entropy` für den **Shannonschen Informationsgewinn** verfügbar.
Wir schauen uns im Folgenden an, wie diese ausgewählt werden können.

### Entscheidungsbäume trainieren

Der Entscheidungsbaum-Klassifikationsalgorithmus von Scikit-Learn bietet noch
weitere Optionen an, wie die Hilfe verrät

```python
help(DecisionTreeClassifier())
```

oder in der [Dokumentation Scikit-Learn → DecisionTreeClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) nachgelesen werden kann.

Sowohl bei der Initalisierung des Entscheidungsbaumes können Parameter gesetzt
werden, als auch beim Verwenden der verschiedenen Methoden. Tatsächlich haben
wir bereits weiter oben aus didaktischen Gründen den Parameter `random_state=0`
bei der Initialisierung gesetzt, damit immer der gleiche Entscheidungsbaum
entsteht. In einem echten Projekt würde dieser Parameter nie verwendet werden.

Probieren Sie andere Werte für den Start des Zufallszahlengenerators aus und
testen Sie, was sich verändert, wenn Sie andere Kriterien für das Splitting
verwenden.

```{code-cell} ipython3
modell = DecisionTreeClassifier(criterion='entropy', random_state=3)
modell.fit(X,y)

plot_tree(modell, 
    feature_names=['Kilometerstand [km]', 'Preis [EUR]'],
    class_names=['nicht verkauft', 'verkauft']);
```

### Zusammenfassung und Ausblick Kapitel 6.2

In diesem Kapitel haben wir das Training von Entscheidungsbäumen mit Hilfe der
Bibliothek Scikit-Learn vertieft. Im nächsten Kapitel widmen wir uns den Vor-,
aber auch den Nachteilen von Entscheidungsbäumen.

## 6.3 Entscheidungsbäume in der Praxis

Entscheidungsbäume bieten viele Vorteile, haben aber auch Nachteile, die wir in
diesem Kapitel diskutieren werden. Darüber hinaus lernen wir Methoden kennen,
bei Entscheidungsbäumen diese Nachteile zu reduzieren.

### Lernziele Kapitel 6.3

* Sie können in eigenen Worten erklären, was **Overfitting** (deutsch:
  **Überanpassung**) ist.
* Sie wissen, was **Underfitting** bedeutet.
* Sie wissen, dass Entscheidungsbäume eine Tendenz zu Overfitting haben und
  Maßnahmen zur Reduzierung von Overfitting ergriffen werden müssen.
* Sie wissen, was **Hyperparameter** sind.
* Sie kennen Hyperparameter der Entscheidungsbäume wie beispielsweise
  * maximale Baumtiefe,
  * minimale Anzahl an Datenpunkten in Knoten oder
  * minimale Anzahl an Datenpunkten in Blättern.
* Sie können die Hyperparameter zum **Prä-Pruning** (deutsch: vorab
  Zurechtschneiden) geeignet wählen.

### Die Tendenz von Entscheidungsbäumen zum Overfitting

Entscheidungsbaummodelle bieten zahlreiche Vorteile. Ein wesentlicher Vorzug ist
die Möglichkeit, den trainierten Entscheidungsbaum zu visualisieren, wodurch es
leicht nachvollziehbar wird, welche Merkmale einen signifikanten Einfluss haben.
Ein weiterer Vorteil ist ihre Effizienz bei heterogenen Daten; sowohl numerische
als auch kategoriale Eigenschaften können problemlos verarbeitet werden.
Entscheidungsbäume sind selbst bei unterschiedlichen Datenskalen robust und
erfordern nur wenig Vorverarbeitung.

Trotz dieser Stärken besitzen Entscheidungsbäume eine Neigung zum
**Overfitting**. Overfitting, auch als Überanpassung bekannt, beschreibt ein
Problem im maschinellen Lernen, bei dem ein Modell die Trainingsdaten zu genau
lernt. Das klingt zunächst gut, aber das Modell kann dadurch seine Fähigkeit
verlieren, Vorhersagen für neue, unbekannte Daten zu treffen. Im Gegensatz dazu
steht das **Underfitting**, das eine zu geringe Anpassung an die Daten bedeutet
und ebenfalls unerwünscht ist.

Um uns das Problem des Overfittings zu veranschaulichen, betrachten wir erneut
das Autohaus-Beispiel, aber diesmal mit mehr Autos. Wir lassen die Autos diesmal
mit einer in Scikit-Learn eingebauten Funktion zur Generierung von künstlichen
Daten erzeugen, der sogenannten `make_moons`-Funktion (siehe [Dokumentation
Scikit-Learn →
make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html))
aus dem Module `sklearn.datasets`.

```{code-cell}
from sklearn.datasets import make_moons 

X_array, y_array = make_moons(noise = 0.5, n_samples=50, random_state=3)
```

Damit die künstlichen Daten besser zu dem Autohaus-Beispiel passen,
transformieren wir sie und nutzen die Pandas-Datenstrukturen, um sie effizient
zu verwalten.

```{code-cell}
import numpy as np
import pandas as pd

# Transformation der Merkmalswerte in einen positiven Bereich und 
# Umwandlung in eine Integer-Matrix
X_array = X_array + 1.2 * np.abs(np.min(X_array))
X_array = X_array + 1.2 * np.abs(np.min(X_array))
X_array[:,0] = np.ceil(X_array[:,0] * 30000)
X_array[:,1] = np.ceil(X_array[:,1] * 10000)
X = pd.DataFrame(X_array, columns=['Kilometerstand [km]', 'Preis [EUR]'], dtype=(int, int))

# Zuweisung von True/False basierend auf den Kategorien 1 bzw. 0
y_array = (y_array - 1.0) * (-1)
y = pd.Series(y_array, name='verkauft', dtype='bool')
```

Nach der Datenvorbereitung visualisieren wir diese:

```{code-cell}
import plotly.express as px

fig = px.scatter(x = X['Kilometerstand [km]'], y = X['Preis [EUR]'], color=y,
    title='Künstliche Daten Autohaus',
    labels={'x': 'Kilometerstand [km]', 'y': 'Preis [EUR]', 'color': 'verkauft'})
fig.show()
```

Das Training des Entscheidungsbaumes und dessen Visualisierung erledigt der
folgende Code.

```{code-cell}
from sklearn.tree import DecisionTreeClassifier, plot_tree

modell = DecisionTreeClassifier(random_state=0)
modell.fit(X,y)

plot_tree(modell,
    feature_names=['Kilometerstand [km]', 'Preis [EUR]'],
    class_names=['nicht verkauft', 'verkauft']);
```

Die Visualisierung offenbart zahlreiche Verzweigungen und eine schwer lesbare
Beschriftung. Die Entscheidungsgrenzen, die im Folgenden mit
`DecisionBoundaryDisplay` visualisiert werden, zeigen eine zu starke Anpassung
an die Trainingsdaten.

```{code-cell}
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

fig = DecisionBoundaryDisplay.from_estimator(modell, X, cmap=ListedColormap(['#EF553B33', '#636EFA33']), grid_resolution=1000)
fig.ax_.scatter(X['Kilometerstand [km]'], X['Preis [EUR]'], c=y, cmap=ListedColormap(['#EF553B', '#636EFA']))
fig.ax_.set_title('Entscheidungsgrenzen');
```

Es ist fraglich, ob dieser Entscheidungsbaum nicht zu genau an die
Trainingsdaten angepasst wurde. Der dünne blaue vertikale Streifen bei ungefähr
97000 km ist wahrscheinlich keine sinnvolle Entscheidung, sondern eher einem
Ausreißer geschuldet (dem Auto mit einem Kilometerstand von 97098 km und einem
Preis von 28229 EUR). Der Entscheidungsbaum hat sich zu stark an die Daten
angepasst. Es ist wahrscheinlich, dass dieser Entscheidungsbaum für Autos mit
einem Kilometerstand von ungefähr 97000 km falsche Prognosen treffen wird. Wenn
wir mit den gleichen Daten erneut einen Entscheidungsbaum trainieren lassen und
den Zufallszahlengenerator mit dem Zustand `random_state=1` initialisieren,
erhalten wir ein völlig anderes Ergebnis.

```{code-cell}
modell_alternative = DecisionTreeClassifier(random_state=1)
modell_alternative.fit(X,y)

fig = DecisionBoundaryDisplay.from_estimator(modell_alternative, X, cmap=ListedColormap(['#EF553B33', '#636EFA33']), grid_resolution=1000)
fig.ax_.scatter(X['Kilometerstand [km]'], X['Preis [EUR]'], c=y, cmap=ListedColormap(['#EF553B', '#636EFA']))
fig.ax_.set_title('Entscheidungsgrenzen des alternativen Modells');
```

Eine Möglichkeit, das Overfitting (Überanpassung) an die Daten zu bekämpfen, ist
das Zurechtschneiden (Pruning) der Entscheidungsbäume. Eine andere ist, aus
mehreren Entscheidungbäumen einen »durchschnittlichen« Entscheidungsbaum zu
bilden. Dieses Verfahren heißt Zufallswald (Random Forest) und wird ausführlich
in einem eigenen Kapitel behandelt werden. In diesem Kapitel betrachten wir nur
das Zurechtschneiden der Entscheidungsbäume.

### Zurechtschneiden von Entscheidungsbäumen

Eine effektive Strategie zur Bekämpfung des Overfitting-Problems bei
Entscheidungsbäumen ist das sogenannte **Pruning**, also das Beschneiden des
Baumes. Pruning hilft, die Komplexität des Modells zu reduzieren, indem weniger
relevante Entscheidungszweige nach bestimmten Kriterien entfernt werden. Im
Kontext unseres Autohaus-Beispiels würde dies bedeuten, dass
Entscheidungszweige, die beispielsweise aufgrund von Ausreißern entstanden sind,
abgeschnitten werden. Dies könnte beispielsweise den zuvor erwähnten dünnen
blauen Streifen bei einem Kilometerstand von ungefähr 97000 km betreffen, der
wahrscheinlich durch einen Ausreißer entstanden ist. Durch das Entfernen solcher
spezifischen Anpassungen kann der Entscheidungsbaum besser verallgemeinern und
wird robuster gegenüber neuen, unbekannten Daten. Das Ergebnis ist ein Modell,
das eine bessere Balance zwischen Anpassung an die Trainingsdaten und
Generalisierungsfähigkeit aufweist.

Für Entscheidungsbäume gibt es prinzipiell zwei Methoden des Prunings:
**Prä-Pruning** und **Post-Pruning**. Das Prä-Pruning findet *vor* dem Training
des Entscheidungsbaumes statt, das Post-Pruning *nach* dem Training. Die beiden
wichtigsten Prä-Pruning-Maßnahmen sind

* die Begrenzung der maximalen Tiefe des Baumes und
* die Forderung nach einer Mindestanzahl von Datenpunkten (entweder pro Knoten
  oder pro Blatt).

Beim Post-Pruning werden im Nachhinein Knoten mit wenig Informationen aus dem
Entscheidungsbaum entfernt oder es werden Knoten zusammengelegt. Scikit-Learn
hat nur Prä-Pruning implementiert, so dass wir hier nicht weiter auf
Post-Pruning eingehen.

#### Prä-Pruning: Baumtiefe

Wir schauen uns zunächst an, wie bei Scikit-Learn-Entscheidungsbäumen die
maximale Tiefe festgelegt wird. Bisher haben wir das Modell ohne weitere
Parameter initialisiert (einzige Ausnahme: wir haben ggf. den
Zufallszahlengenerator aus didaktischen Gründen fixiert, damit die Ergebnisse
vergleichbar sind). Nun verwenden wir bei der Initialisierung des
DecisionTreeClassifiers das optionale Argument `max_depth=` und setzen es auf
`1`.

```{code-cell}
modell_tiefe1 = DecisionTreeClassifier(random_state=0, max_depth=1)
modell_tiefe1.fit(X,y)

plot_tree(modell_tiefe1,
    feature_names=['Kilometerstand [km]', 'Preis [EUR]'],
    class_names=['nicht verkauft', 'verkauft']);
```

Eine Tiefe von 1 bedeutet, dass nur noch eine einzige Entscheidungsfrage
gestellt wird. Das reicht nicht mehr, um die Autos in reine Blätter zu
sortieren. Im linken Blatt sind 13 nicht verkaufte Autos und 24 verkaufte Autos,
weshalb diesem Blatt die Kategorie »verkauft« zugeordnet wird. Im rechten Blatt
sind 12 nicht verkaufte Autos und ein verkauftes Auto, so dass dieses Blatt
insgesamt als »nicht verkauft« gilt. Die Visualisierung der Entscheidungsgrenzen
zeigt, um welche Autos es sich handelt.

```{code-cell}
fig = DecisionBoundaryDisplay.from_estimator(modell_tiefe1, X, cmap=ListedColormap(['#EF553B33', '#636EFA33']), grid_resolution=1000)
fig.ax_.scatter(X['Kilometerstand [km]'], X['Preis [EUR]'], c=y, cmap=ListedColormap(['#EF553B', '#636EFA']))
fig.ax_.set_title('Entscheidungsgrenzen');
```

Insbesondere die Visualisierung der Entscheidungsgrenzen zeigt aber auch, dass
dieser Entscheidungsbaum nicht besonders gut die Daten erklärt. Der Score ist
mit

```{code-cell}
print(f'Score des Entscheidungsbaumes mit Tiefe 1: {modell_tiefe1.score(X,y)}')
```

auch nicht so gut. Daher verwenden wir nun als maximale Tiefe des Entscheidungsbaumes einen Wert von 2.

```{code-cell}
modell_tiefe2 = DecisionTreeClassifier(random_state=0, max_depth=2)
modell_tiefe2.fit(X,y)

plot_tree(modell_tiefe2,
    feature_names=['Kilometerstand [km]', 'Preis [EUR]'],
    class_names=['nicht verkauft', 'verkauft']);

print(f'Score des Entscheidungsbaumes mit Tiefe 2: {modell_tiefe2.score(X,y)}')
```

Mit einem Score von 0.78 ist der Entscheidungsbaum mit einer maximalen Tiefe von
2 zwar besser als der Baum mit einer maximalen Tiefe von 1, aber deutlich
entfernt von dem Score 1.0 bei einer Baumtiefe von 7. Die Entscheidungsgrenzen
sehen folgendermaßen aus:

```{code-cell}
fig = DecisionBoundaryDisplay.from_estimator(modell_tiefe2, X, cmap=ListedColormap(['#EF553B33', '#636EFA33']), grid_resolution=1000)
fig.ax_.scatter(X['Kilometerstand [km]'], X['Preis [EUR]'], c=y, cmap=ListedColormap(['#EF553B', '#636EFA']))
fig.ax_.set_title('Entscheidungsgrenzen');
```

Was ist jetzt besser, eine maximale Tiefe von 1 oder 2? Oder doch 3 vielleicht?
Die Einführung der maximalen Tiefe bietet den Vorteil, das Overfitting zu
bekämpfen. Der Nachteil davon ist, dass wir jetzt einen neuen Parameter haben,
der das Training und die Prognose des Modells bestimmt. Und für diesen Parameter
muss ein passender Wert eingestellt werden. Solche Parameter nennt man
**Hyperparameter**.

> Was ist ... ein Hyperparameter?
Ein Hyperparameter ist ein Parameter, der vor dem Training eines Modells
festgelegt wird und nicht aus den Daten während des Trainings gelernt wird. Die
Hyperparameter steuern den gesamten Lernprozess und haben einen wesentlichen
Einfluss auf die Leistung des Modells.

Kommen wir nun zu einem anderen Hyperparameter der Entscheidungsbäume, der
Mindestanzahl von Datenpunkten.

#### Prä-Pruning: Mindestanzahl Datenpunkte

Genau wie der Hyperparameter zur Begrenzung der Baumtiefe wird die Mindestanzahl
der Datenpunkte vorab bei der Initialisierung des Entscheidungsbaumes
festgelegt. Scikit-Learn bietet wiederum zwei Möglichkeiten, über die minimale
Anzahl von Datenpunkten den Entscheidungsbaum zurechtzuschneiden. Zum einen kann
für die *Knoten* eine minimal erforderliche Anzahl von Datenpunkten festgelegt
werden, ab der es erlaubt ist, durch Entscheidungsfragen weiter zu verzweigen.
Zum anderen kann eine minimale Anzahl an Datenpunkten für jedes *Blatt*
festgelegt werden, das am Ende der Verzweigungen erreicht werden muss.

Wir probieren beide Möglichkeiten aus und vergleichen die Ergebnisse
miteinander. Die Option zur Einstellung der Mindestanzahl pro Knoten heißt
`min_samples_split` und die Option zur Einstellung des Mindestanzahl Datenpunkte
pro Blatt heißt `min_samples_leaf`. Beiden optionalen Argumenten kann entweder
ein Integer übergeben werden oder ein Float. Wird ein Integer übergeben, so ist
damit die tatsächliche minimale Anzahl an Datenpunkten gemeint. Ein Float wird
als Bruch interpretiert und meint die relative Anzahl der Datenpunkte. Der Bruch
wird mit der Gesamtzahl der Datenpunkte multipliziert und dann wird auf die
nächste ganze Zahl aufgerundet.

Schauen wir uns beide Varianten an. Zunächst begrenzen wir die Knoten und
fordern, dass sich in jedem Entscheidungsknoten mindestens sechs Datenpunkte
befinden müssen.

```{code-cell}
modell_knotenbegrenzung = DecisionTreeClassifier(random_state=0, min_samples_split=6)
modell_knotenbegrenzung.fit(X,y)

plot_tree(modell_knotenbegrenzung,
    feature_names=['Kilometerstand [km]', 'Preis [EUR]'],
    class_names=['nicht verkauft', 'verkauft']);

print(f'Score des Entscheidungsbaumes mit Prä-Pruning Mindestanzahl Datenpunkte pro Knoten: {modell_knotenbegrenzung.score(X,y)}')
```

Der Score ist 0.92. Nun fordern wir, dass in jedem Blatt mindestens sechs
Datenpunkte verbleiben müssen.

```{code-cell}
modell_blattbegrenzung = DecisionTreeClassifier(random_state=0, min_samples_leaf=6)
modell_blattbegrenzung.fit(X,y)

plot_tree(modell_blattbegrenzung,
    feature_names=['Kilometerstand [km]', 'Preis [EUR]'],
    class_names=['nicht verkauft', 'verkauft']);

print(f'Score des Entscheidungsbaumes mit Prä-Pruning Mindestanzahl Datenpunkte pro Knoten: {modell_blattbegrenzung.score(X,y)}')
```

In diesem Fall erhalten wir einen Entscheidungsbaum mit einem Score von 0.82.
Was jetzt die bessere Wahl ist -- Begrenzung der Baumtiefe oder Festlegung einer
Mindestanzahl von Datenpunkten Knoten/Blatt -- und vor allem welche Wert der
Hyperparameter haben soll, muss gesondert untersucht werden.

## Zusammenfassung und Ausblick Kapitel 6.3

In diesem Kapitel haben wir die Tendenz der Entscheidungsbäume zum Overfitting
diskutiert. Um dem Problem des Overfittings zu begegnen, bietet Scikit-Learn die
Möglichkeit des Prä-Prunings. Durch die Begrenzung der maximalen Baumtiefe oder
die Festlegung einer Mindestanzahl von Datenpunkten in Knoten oder Blättern kann
Overfitting reduziert werden. Diese zusätzlichen Parameter des
Entscheidungsbaum-Modells werden Hyperparameter genannt und müssen adjustiert
werden. Eine weitere Alternative, das Overfitting von Entscheidungsbäumen zu
minimieren, bieten die Random Forests, die wir in einem späteren Kapitel
kennenlernen werden.
