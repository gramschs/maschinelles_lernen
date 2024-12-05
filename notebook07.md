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

# 7. Lineare Regression

In diesem Kapitel beschäftigen wir uns mit der linearen Regression. Wie lernen,
wie mit Scikit-Learn ein Regressionsmodell trainiert wird und betrachten auch
die multiple Regression.

## 7.1 Einfache lineare Regression

Die lineare Regression gehört zu den überwachten maschinellen Lernverfahren
(Supervised Learning). Meist ist sie das erste ML-Modell, das eingesetzt wird,
um Regressionsprobleme zu lösen. In diesem Kapitel stellen wir in das Konzept
und die Umsetzung der einfachen linearen Regression mit Scikit-Learn ein.

### Lernziele Kapitel 7.1

* Sie kennen das **lineare Regressionsmodell**.
* Sie können erklären, was die **Fehlerquadratsumme** ist.
* Sie wissen, dass das Training des lineare Regressionsmodells durch die
  **Minimierung** der Fehlerquadratsumme (Kleinste-Quadrate-Schätzer) erfolgt.
* Sie können mit Scikit-Learn ein lineares Regressionsmodell trainieren.
* Sie können mit einem trainierten linearen Regressionsmodell Prognosen treffen.
* Sie können mit dem **Bestimmtheitsmaß** bzw. **R²-Score** beurteilen, ob das
  lineare Regressionsmodell geeignet zur Erklärung der Daten ist.

### Regression kommt aus der Statistik

In der Statistik beschäftigen sich Mathematikerinnen und Mathematiker bereits
seit Jahrhunderten damit, Analyseverfahren zu entwickeln, mit denen
experimentelle Daten gut erklärt werden können. Falls wir eine "erklärende”
Variable haben und wir versuchen, die Abhängigkeit einer Messgröße von der
erklärenden Variable zu beschreiben, nennen wir das Regressionsanalyse oder kurz
**Regression**. Bei vielen Problemen suchen wir nach einem linearen Zusammenhang
und sprechen daher von **linearer Regression**. Mehr Details finden Sie auch bei
[Wikipedia → Regressionsanalyse](https://de.wikipedia.org/wiki/Regressionsanalyse).

Etwas präziser formuliert ist lineare Regression ein Verfahren, bei dem es eine
Einflussgröße $x$ und eine Zielgröße $y$ gibt. In der ML-Sprechweise wird die
Einflussgröße $x$ typischerweise als **Merkmal** (oder englisch **Input** oder
**Feature**) bezeichnet. Die **Zielgröße** (manchmal auch **Output** oder
**Target** genannt), soll stetig sein (manchmal auch kontinuierlich, metrisch
oder quantitativ genannt). Zu dem Merkmal oder den Merkmalen liegen $M$
Datenpunkte mit den dazugehörigen Werte der Zielgröße vor. Diese werden
üblicherweise als Paare (wenn nur ein Merkmal vorliegt) zusammengefasst:

$$(x^{(1)},y^{(1)}), \, (x^{(2)},y^{(2)}), \, \ldots, \, (x^{(M)},y^{(M)}).$$

Ziel der linearen Regression ist es, zwei Parameter $w_0$ und $w_1$ so zu
bestimmen, dass möglichst für alle Datenpunkte $(x^{(i)}, y^{(i)})$ die lineare
Gleichung

$$y^{(i)} \approx w_0 + w_1 x^{(i)}$$

gilt. Geometrisch ausgedrückt: durch die Daten soll eine Gerade gelegt werden,
wie die folgende Abbildung zeigt. Die Datenpunkte sind blau, die
Regressionsgerade ist in rot visualisiert.

![Lineare Regression](https://gramschs.github.io/book_ml4ing/_images/Linear_regression.svg)

In der Praxis werden die Daten nicht perfekt auf der Geraden liegen. Die Fehler
zwischen dem echten $y^{(i)}$ und dem Funktionswert der Gerade $f(x^{(i)}) = w_0 +
w_1 x^{(i)}$ werden unterschiedlich groß sein, je nachdem, welche Parameter
$w_0$ und $w_1$ gewählt werden. Wie finden wir jetzt die beste Kombination $w_0$
und $w_1$, so dass diese Fehler möglichst klein sind?

### Wie groß ist der Fehler?

Das Prinzip für das lineare Regressionsmodell und auch die folgenden ML-Modelle
ist jedesmal gleich. Das Modell ist eine mathematische Funktion, die aber noch
Parameter (hier beispielsweise die Koeffizienten der Gerade) enthält. Dann wird
festgelegt, was eine gute Prognose ist, also wie Fehler berechnet und beurteilt
werden sollen. Das hängt jeweils von dem betrachteten Problem ab. Sobald das
sogenannte Fehlermaß feststeht, werden die Parameter der Modellfunktion so
berechnet, dass das Fehlermaß (z.B. Summe der Fehler oder Mittelwert der Fehler)
möglichst klein wird. In der Mathematik sagt man dazu **Minimierungsproblem**.

Für die lineare Regression wird als Fehlermaß die Kleinste-Quadrate-Schätzung
verwendet (siehe [Wikipedia  → Methode der kleinsten
Quadrate](https://de.wikipedia.org/wiki/Methode_der_kleinsten_Quadrate)). Dazu
berechnen wir, wie weit weg die Gerade von den Messpunkten ist. Wie das geht,
veranschaulichen wir uns mit der folgenden Grafik.

![R2-Score](https://gramschs.github.io/book_ml4ing/_images/kq_regression.png)

Unsere rote Modellfunktion trifft die Messpunkte mal mehr und mal weniger gut.
Wir können jetzt für jeden Messpunkt berechnen, wie weit die rote Kurve von ihm
weg ist (= grüne Strecke), indem wir die Differenz der y-Koordinaten errechnen:
$r = y_{\text{blau}}-y_{\text{rot}}$. Diese Differenz nennt man **Residuum**.
Danach summieren wir die Fehler (also die Residuen) auf und erhalten den
Gesamtfehler. Leider kann es dabei passieren, dass am Ende als Gesamtfehler 0
herauskommt, weil beispielsweise für den 1. Messpunkt die blaue y-Koordinate
unter der roten y-Koordinate liegt und damit ein negatives Residuum herauskommt,
aber für den 5. Messpunkt ein positives Residuum. Daher quadrieren wir die
Residuen. Dann wird diese **Fehlerquadratsumme** minimiert, um die Koeffizienten
des Regressionsmodells zu berechnen.

### Einfache lineare Regression mit Scikit-Learn

Nach diesem theoretischen Exkurs möchten wir Scikit-Learn nutzen, um eine
einfache lineare Regression durchzuführen. Aus didaktischen Gründen erzeugen wir
uns dazu künstliche Daten mit der Funktion `make_regression` des Moduls
`sklearn.datasets`. Wir transformieren die zufällig erzeugten Zahlen und packen
sie in einen Pandas-DataFrame mit den Merkmalen »Leistung \[PS\]« eines Autos
und dem »Preis \[EUR\]« eines Autos.

```{code-cell} ipython3
import numpy as np 
import pandas as pd 
from sklearn.datasets import make_regression

X_array, y_array = make_regression(n_samples=100, n_features=1, noise=10, random_state=0)

daten = pd.DataFrame({
    'Leistung [PS]': np.floor(50*(X_array[:,0] + 3)),
    'Preis [EUR]': 100*(y_array+150)
    })
```

Mehr Details zu der Funktion `make_regression` finden Sie in der [Dokumentation
Scikit-Learn →
make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression).
Wir visualisieren nun den Preis in Abhängigkeit von der Leistung des Autos.

```{code-cell} ipython3
import plotly.express as px 

fig = px.scatter(daten, x = 'Leistung [PS]', y = 'Preis [EUR]',
    title='Künstliche Daten: Verkaufspreise Autos')
fig.show()
```

Es drängt sich die Vermutung auf, dass der Preis eines Autos linear von der
Leistung abhängt. Je mehr PS, desto teurer das Auto.

Als nächstes trainieren wir ein lineares Regressionsmodell auf den Daten.
Lineare ML-Modelle fasst Scikit-Learn in einem Untermodul namens `linear_model`
zusammen. Um also das lineare Regressionsmodell `LinearRegression` verwenden zu
können, müssen wir es folgendermaßen importieren und initialisieren:

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

modell = LinearRegression()
```

Mit der Methode `.fit()` werden die Parameter des linearen Regressionsmodells an
die Daten angepasst. Dazu müssen die Daten in einem bestimmten Format vorliegen.
Bei den Inputs wird davon ausgegangen, dass mehrere Merkmale in das Modell
eingehen sollen. Die Eigenschaften stehen normalerweise in den Spalten des
Datensatzes. Beim Output erwarten wir zunächst nur eine Eigenschaft, die durch
das Modell erklärt werden soll. Daher geht Scikit-Learn davon aus, dass der
Input eine Tabelle (Matrix) $X$ ist, die M Zeilen und N Spalten hat. M ist die
Anzahl an Datenpunkten, hier also die Anzahl der Autos, und N ist die Anzahl der
Merkmale, die betrachtet werden sollen. Da wir momentan nur die Abhängigkeit des
Preises von der PS-Zahl analysieren wollen, ist $N=1$. Beim Output geht
Scikit-Learn davon aus, dass eine Datenreihe (eindimensionaler Spaltenvektor)
vorliegt, die natürlich ebenfalls M Zeilen hat. Wir müssen daher unsere
PS-Zahlen noch in das Matrix-Format bringen. Dazu verwenden wir den Trick, dass
mit `[ [list] ]` eine Tabelle extrahiert wird.

```{code-cell} ipython3
# Adaption der Daten
X = daten[['Leistung [PS]']]
y = daten['Preis [EUR]']
```

Danach können wir das lineare Regressionsmodell trainieren.

```{code-cell} ipython3
modell.fit(X,y)
```

Es erfolgt keine Ausgabe, aber jetzt ist das lineare Regressionsmodell
trainiert. Die durch das Training bestimmten Parameter des Modells sind im
Modell selbst abgespeichert. Bei dem linearen Regressionsmodell sind das die
beiden Parameter $w_0$ und $w_1$, also Steigung `.coef_` und den
y-Achsenabschnitt `.intercept_`.

```{code-cell} ipython3
print(f'Steigung: {modell.coef_}')
print(f'y-Achsenabschnitt: {modell.intercept_}')
```

Damit lautet das lineare Regressionsmodell, um aus der PS-Zahl eines Autos $x$
den Verkaufspreis $y$ zu berechnen, folgendermaßen:

$$y = 85.2 \cdot x + 2179.$$

### Prognosen treffen

Wenn wir die Parameter des trainierten Modells ausgeben lassen bzw. die lineare
Funktion $y = 85.2 \cdot x + 2179$ verwenden, können wir mit dem linearen Modell
Prognosen treffen. Den Umweg über das Ausgeben der trainierten Parameter und dem
Basteln einer linearen Funktion können wir uns aber sparen, denn Scikit-Learn
stellt für Prognosen mit dem trainierten Modell direkt eine Methode zur
Verfügung. Mit Hilfe der `predict()`-Methode können für jedes Scikit-ML-Modell
Prognosen getroffen werden.

Wir möchten uns den kompletten Bereich zwischen 20 PS und 270 PS ansehen und
erzeugen daher 100 Punkte in diesem Bereich. Diese packen wir in einen
Pandas-DataFrame und verwenden dann die `predict()`-Methode.

```{code-cell}
testdaten = pd.DataFrame({
    'Leistung [PS]': np.linspace(20, 270)
    })
prognose = modell.predict(testdaten[['Leistung [PS]']])   
```

Diese Prognose wird dann zusammen mit den Verkaufsdaten in einem Diagramm
visualisiert. Dazu generieren wir zuerst den Scatter-Plot mit den Verkaufsdaten
und fügen dann mit der Funktion `add_scatter()` einen zweiten Scatter-Plot zu
dem ersten hinzu. In diesem Scatter-Plot sollen die Punkte jedoch durch eine
Linie verbunden werden, weshalb wir die Option `mode='lines'` nutzen. Zusätzlich
kennzeichnen wir die Regressionsgerade noch mit dem Namen `name='Prognose'`.

```{code-cell}
fig = px.scatter(daten, x = 'Leistung [PS]', y = 'Preis [EUR]',
    title='Verkaufspreise von Autos')
fig.add_scatter(x = testdaten['Leistung [PS]'], y = prognose, mode='lines', name='Prognose')
fig.show()
```

Der visuelle Eindruck ist gut, aber ist diese Regressionsgerade wirklich das
beste Modell? Im nächsten Abschnitt sehen wir uns ein statistisches Bewertungsmaß
an, um die Güte des Modells zu beurteilen.

### Ist das beste Modell gut genug? Der R²-Score

Auch wenn wir mit der Minimierung der Fehlerquadratsumme bzw. der
Kleinsten-Quadrate-Methode die besten Parameter für unsere Modellfunktion
gefunden haben, heißt das noch lange nicht, dass unser Modell gut ist. Bereits
die Modellfunktion kann ja völlig falsch gewählt sein. Beispielsweise könnten
wir Messungen rund um eine sinus-förmige Wechselspannung vornehmen und dann wäre
ein lineares Regressionsmodell völlig ungeeignet, auch wenn die
Fehlerquadratsumme minimal wäre.

Wir brauchen daher noch ein Kriterium dafür, ob das trainierte Modell auch
valide ist. Für die lineare Regression nehmen wir das **Bestimmtheitsmaß**, das
in der ML-Community auch **R²-Score** genannt wird. Der R²-Score wird dabei
folgendermaßen interpretiert:

* Wenn $R^2 = 1$  ist, dann gibt es den perfekten linearen Zusammenhang und die
  Modellfunktion ist eine sehr gute Anpassung an die Messdaten.
* Wenn $R^2 = 0$ oder gar negativ ist, dann funktioniert die lineare
  Modellfunktion überhaupt nicht.

Auf der Seite [https://mathweb.de](https://mathweb.de) finden Sie eine Reihe von
Aufgaben und interaktiven Demonstrationen rund um die Mathematik. Insbesondere
gibt es dort auch eine interaktive Demonstration des R²-Scores.

Drücken Sie auf den zwei kreisförmigen Pfeile rechts oben. Dadurch wird ein
neuer Datensatz erzeugt. Die Messdaten sind durch grüne Punkte dargestellt, das
lineare Regressionsmodell durch eine blaue Gerade. Im Titel wird der aktuelle
und der optimale R²-Wert angezeigt. Ziehen Sie an den weißen Punkten, um die
Gerade zu verändern. Schaffen Sie es, den optimalen R²-Score zu treffen?
Beobachten Sie dabei, wie die Fehler (rot) kleiner werden.

> <https://lti.mint-web.de/examples/index.php?id=01010320>

Wie ist nun der R²-Score für das trainierte lineare Regressionsmodell? Dazu
verwenden wir die `score()`-Methode.

```{code-cell} ipython3
r2_score = modell.score(X,y)
print(f'Der R2-Score für das lineare Regressionsmodell ist: {r2_score:.2f}.')
```

Das lineare Regressionsmodell kann für die Trainingsdaten sehr gut die
Verkaufspreise prognostizieren. Wie gut es allerdings noch unbekannte Daten
prognostizieren könnte, ist ungewiss.

### Zusammenfassung und Ausblick Kaptel 7.1

In diesem Abschnitt haben Sie das theoretische Modell der linearen Regression
kennengelernt. Das Training eines linearen Regressionsmodells mit Scikit-Learn
erfolgt wie üblich mit der `fit()`-Methode, die Prognose mit der
`predict()`-Methode. Bewerten können Sie Prognosequalität mit der
`score()`-Methode. Im nächsten Kapitel betrachten wir die lineare Regression,
bei der die Zielgröße von mehreren Merkmalen abhängt

## 7.2 Multiple lineare Regression

Bisher haben wir nur ein einzelnes Merkmal aus den gesammelten Daten
herausgegriffen und untersucht, ob es zwischen diesem Merkmal und der Zielgröße
einen linearen Zusammenhang gibt. So simpel ist die Welt normalerweise nicht,
oft wirken mehrere Einflussfaktoren gleichzeitig. Daher steht die **multiple
lineare Regression** in diesem Kapitel im Fokus.

### Lernziele Kapitel 7.2

* Sie wissen, was eine **multiple lineare Regression** ist und können sie mit
  Scikit-Learn durchführen.
* Sie wissen, was **positive lineare Korrelation** und **negative lineare
  Korrelation** bedeuten.
* Sie können die lineare Korrelation der Merkmale miteinander mit Hilfe der
  **Korrelationsmatrix** beurteilen.
* Sie können die Korrelationsmatrix als **Heatmap** visualisieren.

### Zwei Merkmale: PS und Alter beinflussen Preis

Im vorherigen Kapitel haben wir den Einfluss des Merkmals `Leistung [PS]` auf
die Zielgröße `Preis [EUR]` betrachtet. Nun wollen wir noch das Merkmal `Alter`
gemessen in Jahren hinzunehmen. 0 Jahre meint dabei einen Neuwagen. Aus
didaktischen Gründen werden wir auch hier künstlich erzeugte Daten nutzen, um
die multiple lineare Regression zu erklären. Als erstes erzeugen wir die Daten,
diesmal direkt mit Hilfsmitteln des Moduls NumPy.

```{code-cell} ipython3
import numpy as np 
import pandas as pd 

np.random.seed(0)
anzahl_autos = 100

x = np.floor( np.random.uniform(0, 11, anzahl_autos) )
y = np.floor( np.random.uniform(50, 301, anzahl_autos) )
z = np.floor( -2000 * x + 200 * y + 500 * np.random.normal(0, 1, anzahl_autos) + 10000 )

daten = pd.DataFrame({
    'Alter': x,
    'Leistung [PS]': y,
    'Preis [EUR]': z
    })
```

Dann visualisieren wir die Daten.

```{code-cell} ipython3
import plotly.express as px

fig = px.scatter_3d(daten, x = 'Alter', y = 'Leistung [PS]', z = 'Preis [EUR]',
  title='Künstliche Verkaufspreise für Autos')
fig.show()
```

Mehr als zwei Merkmale und eine Zielgröße können wir nicht sinnvoll
visualisieren. Um insbesondere zu visualisieren, wie die Zielgröße von jedem
einzelnen Merkmal abhängt, verwenden wir die Scattermatrix. Das hilft uns auch
zu erkennen, ob vielleicht ein Merkmal von einem anderen Merkmal abhängt.

```{code-cell} ipython3
fig = px.scatter_matrix(daten,
    title='Künstliche Daten: Verkaufspreise Autos')
fig.show()
```

Wir betrachten die letzte Zeile, in der die Zielgröße auf der y-Achse
aufgetragen ist. In der ersten Spalte wird der Preis abhängig vom Alter
dargestellt. Je älter das Auto, desto geringer der Preis. In der zweiten Spalte
wird der Preis abhängig von der Leistung gezeigt. Je leistungsstärker ein Auto,
desto höher der Preis. Insbesondere vermittelt das Diagramm den Eindruck, dass
durch die Punktewolke sehr gut eine Regressionsgerade gelegt werden könnte, was
bei der Abhängigkeit Alter -- Preis eher fraglich ist.

Wir trainieren jetzt ein lineares Regressionsmodell

$$y = w_0 + w_1 \cdot x_1 + w_2 \cdot x_2.$$

Damit ist gemeint, dass wir die Gewichte $w_0, w_1$ und $w_2$ des Modells so
bestimmen wollen, dass der Preis $y$ möglichst gut durch die beiden Merkmale
Alter $x_1$ und Leistung $x_2$ prognostiziert wird.

In einem ersten Schritt laden wir das lineare Regressionsmodul.

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

modell = LinearRegression()
```

Dann adaptieren wir die Daten.

```{code-cell} ipython3
# Adaption der Daten
X = daten[['Alter', 'Leistung [PS]']]
y = daten['Preis [EUR]']
```

Jetzt können wir das lineare Regressionsmodell von Scikit-Learn mit der
`.fit()`-Methode trainieren. Wir lassen auch gleich den R²-Score mit ausgeben.

```{code-cell} ipython3
# Training
modell.fit(X, y)

# Validierung
r2_score = modell.score(X, y)
print(f'Der R2-Score ist: {r2_score:.4f}')
```

Schauen wir uns doch einmal an, welche Koeffizienten von Scikit-Learn für unsere
mehrdimensionale lineare Modellfunktion gefunden wurden.

```{code-cell} ipython3
print(f'Achsenabschnitt w0: {modell.intercept_:.2f}')
print(f'Koeffizienten (Steigungen): {modell.coef_}')
```

Damit lautet unsere Modellfunktion abhängig von Alter und Leistung also

$$y = f(x_1, x_2) = 10168 -2025\cdot x_1 + 199\cdot x_2.$$

### Korrelationsmatrix

Das Prognoseergebnis des multiplen linearen Regressionsmodell ist für die
Trainingsdaten sehr gut. Beim Betrachten der Scattermatrix wirkt das Merkmal
Leistung mehr einen linearen Einfluss zu haben als das Alter. Als nächstes
wollen wir bewerten, wie viel mehr der lineare Einfluss jedes einzelnen Merkmals
auf die Zielgröße ist. Dazu betrachten wir die sogenannte
**Korrelationsmatrix**. Mit der Methode `corr()` können wir sie einfach
berechnen lassen:

```{code-cell} ipython3
daten.corr()
```

In der ersten Zeile 'Alter' wird die Stärke der Korrelation von der Ursache
Alter auf die Zielgrößen Alter, Leistung und Preis bewertet. Wenn eine Erhöhung
der Ursache zu einer Erhöhung der Wirkung führt, nennt man das **positiv
korreliert**. Der umgekehrte Fall ist, wenn eine Verminderung der Ursache zu
einer Erhöhung der Wirkung führt. Dann spricht man von **negativ korreliert**.
Die Zahl 1 drückt dabei aus, dass die beiden Merkmale perfekt linear positiv
korreliert sind. In der ersten Zeile und der ersten Zeile wird der Einfluss des
Alters auf die Zielgröße Alter bewertet. Dort muss eine 1 stehen, denn hier sind
ja Ursache und Wirkung identisch. In der ersten Zeile und der zweiten Spalte
wird die lineare Korrelation zwischen Alter und Leistung bewertet. Die Zahl
-0.074244 ist nahe bei 0 und bedeutet daher, dass es nur einen sehr, sehr
schwachen Zusammenhang zwischen Alter und Leistung gibt, wenn überhaupt. Unser
technisches Verständnis eines Autos bestätigt, dass Alter und PS nicht
zusammenhängen (zumidnest, wenn man die Leistung des Autos nimmt, wie sie im
Fahrzeugschien eingetragen ist). Dahingegen scheint es eine schwache negative
Korrelation zwischen Alter und Preis zu geben. Je älter ein Auto ist, desto
geringer ist sein Preis. -1 würde bedeuten, dass die negative Korrelation
perfekt ist.

Am stärksten linear scheint sich die Leistung auf den Preis auszuwirken. In der
zweiten Zeile und der dritten Spalte findet sich der Eintrag 0.914003. Je größer
die Leistung des Autos, desto höher sein Preis.

### Heatmaps

Es ist üblich, die Korrelationsmatrix als sogenanntes Heatmap-Diagramm zu
visualisieren. Bei einer Heatmap wird die Zahlenwerte der Matrix durch Farben
visualisiert. Ploty Express bietet dazu die Funktion `imshow()` an.

```{code-cell} ipython3
korrelationsmatrix = daten.corr()

fig = px.imshow(korrelationsmatrix)
fig.show()
```

Es ist hilfreich, die Werte der Korrelationsmatrix direkt in der Heatmap
anzeigen zu lassen. Daher verwenden wir die zusätzliche Option `text_auto=True`.

```{code-cell} ipython3
fig = px.imshow(korrelationsmatrix, text_auto=True)
fig.show()
```

Weitere Optionen zum Stylen der Heatmaps finden Sie in der [Plotly Dokumentation
→ Heatmaps in Plotly](https://plotly.com/python/heatmaps/).

### Zusammenfassung Kapitel 7.2

In diesem Kapitel haben wir uns mit der linearen multiplen Regression
beschäftigt. Es wird eine lineare Modellfunktion für einen oder mehrere
Einflussfaktoren gesucht. Die Parameter der Modellfunktion, also die
Koeffizienten der mehrdimensionalen linearen Funktion werden so an die Daten
angepasst, dass die Fehlerquadratsumme möglichst klein wird. Um beurteilen zu
können, ob die beste gefundene Modellfunktion eine gute Prognose liefert, werten
wir den R²-Score aus.

Um zu analysieren, ob einzelne Merkmale miteinander linear korreliert sind,
werden die Korrelationsmatrix und die Heatmap eingesetzt

## 7.3 Polynomiale Regression

In den letzten beiden Kapiteln haben wir uns mit der linearen Regression
befasst. Dabei haben wir die einfache lineare Regression betrachtet, bei der die
Zielgröße von einem einzelnen Merkmal abhängt, sowie die multiple lineare
Regression, bei der die Zielgröße von mehreren Merkmalen beeinflusst wird. In
diesem Kapitel werden wir uns damit beschäftigen, wie eine Regression für
quadratische, kubische oder allgemein für polynomiale Modelle durchgeführt wird.

### Lernziele Kapitel 7.3

* Sie können eine **polynomiale Regression** durchführen.
* Sie wissen, dass die Wahl des Polynomgrades entscheidend dafür ist, ob
  **Underfitting (Unteranpassung)**, **Overfitting (Überanpassung)** oder ein
  geeignetes Modell vorliegt.
* Sie wissen, dass der Polynomgrad ein **Hyperparameter** ist.

### Künstliches Experiment zu Bremswegen eines Autos

Ausnahmsweise werden wir uns in diesem Kapitel nicht mit dem Verkauf von Autos
beschäftigen, sondern mit dem Bremsweg von Autos. Die Faustformel zur Berechnung
des Bremsweges $s$ in Metern lautet

$$s = \frac{1}{100} \cdot v^2,$$

wobei die Geschwindigkeit $v$ des Autos in km/h angegeben wird. Natürlich
variiert der tatsächliche Bremsweg abhängig von der Straßenoberfläche (trocken /
nass / vereist) oder dem Fahrzeugtyp (inbesondere Leistung der Bremse). Wird die
Bremsung aufgrund eines plötzlich auftauchenden Hindernisses eingeleitet, kommt
zum Bremwsweg noch der Reaktionsweg hinzu. Mehr Details finden Sie auf den
Internetseiten des ADAC unter [Bremsweg berechnen: Mit dieser Formel
geht's](https://www.adac.de/verkehr/rund-um-den-fuehrerschein/erwerb/bremsweg-berechnen/).

Wir simulieren nun ein Experiment, bei dem der Bremsweg von Autos abhängig von
der Geschwindigkeit gemessen wird. In einem ersten schritt erzeugen wir zufällig
50 Geschwindigkeiten zwischen 30 km/h und 150 km/h. Gemäß der obigen Faustformel
lassen wir zunächst die dazugehörigen Bremswege berechnen, addieren dann aber
noch zufällige Schwankungen.

```{code-cell}
import numpy as np 
import pandas as pd 

np.random.seed(0)
anzahl_experimente = 50
v_min = 30
v_max = 151

v = np.floor( np.random.uniform(v_min, v_max, anzahl_experimente) )
zufaellige_schwankungen = 3 * np.random.normal(0, 1, anzahl_experimente)
bremsweg = 1/100 * v**2 

daten = pd.DataFrame({
    'Geschwindigkeit [km/h]': v,
    'Bremsweg [m]': bremsweg + zufaellige_schwankungen,
    })
```

Als nächstes lassen wir die künstlich erzeugten Bremsweg-Experimente visualisieren.

```{code-cell}
import plotly.express as px 

fig = px.scatter(daten, x = 'Geschwindigkeit [km/h]', y = 'Bremsweg [m]',
    title='Künstliche Daten: Bremsweg eines Autos')
fig.show()
```

### Erster Versuch: lineare Regression

Als erstes verwenden wir die lineare Regression, um ein Modell für die Messdaten
zu finden. Wenn wir die Geschwindigkeit mit $x$ bezeichnen und den Bremsweg mit
$y$, dann lautet das lineare Regressionsmodell

$$y = \omega_0 + \omega_1 \cdot x.$$

```{code-cell}
from sklearn.linear_model import LinearRegression

# Adaption der Daten
X = daten[['Geschwindigkeit [km/h]']]
y = daten['Bremsweg [m]']

# Training des Modells
model = LinearRegression()
model.fit(X, y)

# Bewertung
r2_score = model.score(X, y)
print(f'R2-score Trainingsdaten: {r2_score}')
```

Der R2-Score sieht sehr gut aus. Um uns einen Eindruck zu verschaffen, wie gut
das lineare Modell tatsächlich ist (wir wissen ja, dass es eigentlich
quadratisch ist!), erzeugen wir nun systematisch Geschwindigkeiten in dem
Bereich von 30 km/h und 150 km/h und verwenden die Faustformel für die
Berechnung der Bremswege.

```{code-cell}
v_test = np.linspace(v_min, v_max, 200)
s_test = 1/100 * v_test**2
testdaten = pd.DataFrame({
    'Geschwindigkeit [km/h]': v_test,
    'Bremsweg [m]': s_test
    })
```

Mit Hilfe des linearen Regressionsmodells prognostizieren wir die Bremswege für
diese Geschwindigkeiten und lassen den R2-Score berechnen.

```{code-cell}
# Prognose 
X_test = testdaten[['Geschwindigkeit [km/h]']]
y_test = testdaten['Bremsweg [m]']
y_prognose = model.predict(X_test)

# Bewertung
r2_score = model.score(X_test, y_test)
print(f'R2-score Testdaten: {r2_score}')
```

Zuletzt visualisieren wir die Prognose.

```{code-cell}
fig = px.scatter(daten, x = 'Geschwindigkeit [km/h]', y = 'Bremsweg [m]',
    title='Bremsweg eines Autos: lineares Modell')
fig.add_scatter(x = testdaten['Geschwindigkeit [km/h]'], y = y_prognose, mode='lines', name='Prognose')
fig.add_scatter(x = testdaten['Geschwindigkeit [km/h]'], y = y_test, mode='lines', name='Faustformel')
fig.show()
```

Vor allem die Visualisierung zeigt die Schwächen des linearen Modells. Bei
niedrigen Geschwindigkeiten wie in der Stadt unterschätzt das lineare Modell den
Bremsweg. Unterhalb von 40 km/h prognostiziert das Modell sogar einen negativen
Bremsweg. Zwischen 60 km/h und 120 km/h überschätzt das Modell den Bremsweg und
oberhalb von 120 km/h unterschätzt es den Bremsweg wieder. Das Modell ist zu
einfach für die Prognose, es liegt **Underfitting** vor. Daher probieren wir
als nächstes ein quadratisches Regressionsmodell aus.

### Quadratische Regression

Wenn Sie in der Dokumentation von Scikit-Learn nun nach einer Funktion zur
quadratischen Regression suchen, werden Sie nicht fündig werden. Tatsächlich
brauchen wir auch keine eigenständige Funktion, sondern können uns mit einem
Trick weiterhelfen.

Das lineare Regressionsmodell, das wir eben ausprobiert haben, lautet
mathematisch formuliert folgendermaßen:

$$y = \omega_0 + \omega_1 \cdot x$$

mit nur einem Merkmal $x$, nämlich der Geschwindigkeit.

Wenn wir eine quadratische Funktion als Modellfunktion wählen möchten, erzeugen
wir einfach ein zweites Merkmal. Wir nennen die bisherigen x-Werte $x$
jetzt $x_1$ und fügen als zweites Merkmal die neue Eigenschaft

$$x_2 = \left( x_1 \right)^2$$

hinzu. Damit wird aus dem quadratischen Regressionsmodell

$$y = w_0 + w_1 \cdot x + w_2 \cdot x^2$$

das *multiple* lineare Regressionsmodell

$$y = w_0 + w_1 \cdot x_1 + w_2 \cdot x_2.$$

Scikit-Learn stellt auch hier passende Methoden bereit. Aus dem
Vorbereitungsmodul `sklearn.preprocessing` importieren wir `PolynomialFeatures`.
Mehr Details dazu finden Sie in der [Dokumentation Scikit-Learn →
PolynomialFeature](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html).
Wir erzeugen das PolynomialFeature-Objekt mit der Option `degree=2`, um die
Quadrate hinzuzufügen. Dann transformieren wir die Input-Daten, indem wir die
`fit_transform()`-Methode auf den Input anwenden.

```{code-cell}
from sklearn.preprocessing import PolynomialFeatures

# Adaption der Daten
polynom_transformator = PolynomialFeatures(degree = 2)
X = polynom_transformator.fit_transform(daten[['Geschwindigkeit [km/h]']])
y = daten['Bremsweg [m]']
```

Danach können wir das multiple lineare Regressionsmodell trainieren und bewerten
lassen.

```{code-cell}
# Training des Modells
model = LinearRegression()
model.fit(X, y)

# Bewertung
r2_score = model.score(X, y)
print(f'R2-score Trainingsdaten: {r2_score}')
```

Zuletzt lassen wir das quadratische Regressionsmodell noch visualiseren. Wichtig
ist, dass nun auch die Testdaten quadriert werden müssen, da das ML-Modell für
Prognosen voraussetzt, dass die Daten in der gleichen Art und Weise vorliegen
wie die Trainingsdaten. Wir müssen denselben  Transformator nehmen wie zum
Training der Daten und neutzen daher nur die `transform()`-Methode.

```{code-cell}
# Prognose 
X_test = polynom_transformator.transform(testdaten[['Geschwindigkeit [km/h]']])
y_prognose = model.predict(X_test)
   
fig = px.scatter(daten, x = 'Geschwindigkeit [km/h]', y = 'Bremsweg [m]',
    title='Bremsweg eines Autos: quadratisches Modell')
fig.add_scatter(x = testdaten['Geschwindigkeit [km/h]'], y = y_prognose, mode='lines', name='Prognose')
fig.add_scatter(x = testdaten['Geschwindigkeit [km/h]'], y = y_test, mode='lines', name='Faustformel')
fig.show()
```

Die Prognose ist so gut, dass wir die Prognose (rot) und die Faustformel (grün)
kaum unterscheiden können. Kleinere Abweichungen gibt es bei den Bremswegen für
Geschwindigkeiten oberhalb von 130 km/h. Zoomen Sie in den Plot hinein, um sich
die Unterschiede anzusehen.

**Bemerkung:** Dieser Trick wird auch bei anderen ML-Verfahren angewandt. Aus
einem Merkmal, aus einer Eigenschaft werden jetzt eine neue Eigenschaften
erzeugt. Aus einem eindimensionalen Input wird ein zweidimensionaler Input.
Mathematisch gesehen haben wir die Input-Daten in einen höherdimensionalen
Bereich projiziert. Diese Methode nennt man **Kernel-Trick**. Es ist auch
möglich, andere Funktionen zu benutzen, um die Daten in einen höherdimensionalen
Raum zu projizieren, z.B. radiale Gaußsche Basisfunktionen. Das nennt man dann
**Kernel-Methoden**. In dieser Vorlesung bleiben wir aber bei den Polynomen als
Basisfunktion.

### Polynomiale Regression

Mit diesem Trick, die Merkmale zu quadrieren, können wir weitermachen, z.B. die
Merkmale mit 3 potenzieren. Letztendlich können wir so jedes gewünschte Polynom
als Regressionspolynom trainieren lassen. Das ein höheres Polynom nicht
unbedingt besser sein muss, zeigt das folgende Beispiel. Wir wählen als
Polynomgrad 14.

```{code-cell}
# Adaption der Daten
polynom_transformator = PolynomialFeatures(degree = 14)
X = polynom_transformator.fit_transform(daten[['Geschwindigkeit [km/h]']])
y = daten['Bremsweg [m]']

# Training des Modells
model = LinearRegression()
model.fit(X, y)

# Bewertung
r2_score = model.score(X, y)
print(f'R2-score Trainingsdaten: {r2_score}')

# Prognose
X_test = polynom_transformator.transform(testdaten[['Geschwindigkeit [km/h]']])
y_prognose = model.predict(X_test)

fig = px.scatter(daten, x = 'Geschwindigkeit [km/h]', y = 'Bremsweg [m]',
    title='Bremsweg eines Autos: Polynom 14. Grades')
fig.add_scatter(x = testdaten['Geschwindigkeit [km/h]'], y = y_prognose, mode='lines', name='Prognose')
fig.add_scatter(x = testdaten['Geschwindigkeit [km/h]'], y = y_test, mode='lines', name='Faustformel')
fig.show()
```

Nach 150 km/h wird der Brewmsweg wieder weniger, was natürlich nicht mit der
Praxis übereinstimmt. Die Visualisierung der Prognose zeigt, dass das ein
polynomiales Regressionsmodell mit Grad 14 zu sehr an die Trainingsdaten
angepasst ist und für neue Daten (siehe Geschwindigkeiten größer 150 km/h) nicht
gut geeignet ist. Es liegt **Overfitting** vor.

Bei der polynomialen Regression wird der Polynomgrad zu einem
**Hyperparameter**. Hyperparameter haben wir auch schon bei den
Entscheidungsbäumen (Decision Trees) kennengelernt. Zur Wiederholung geben wir
hier erneut die Definition an.

### Was ist ... ein Hyperparameter?

Ein Hyperparameter ist ein Parameter, der vor dem Training eines Modells
festgelegt wird und nicht aus den Daten während des Trainings gelernt wird. Die
Hyperparameter steuern den gesamten Lernprozess und haben einen wesentlichen
Einfluss auf die Leistung des Modells.

### Zusammenfassung und Ausblick

In diesem Kapitel haben wir die polynomiale Regression mit Scikit-Learn
kennengelernt. Auch bei der polynomialen Regression ist die Wahl des
Hyperparameters Polynomgrad wichtig. Im nächsten Kapitel werden wir uns ansehen.
wie Hyperparameter getunt werden können.

## Übungen

### Aufgabe 7.1 Bevölkerungszahlen in Deutschland

In dieser Aufgabe betrachten wir den Datensatz `population_germany.csv`. Führen
Sie zuerst eine explorative Datenanalyse durch. Geben Sie dann mit Hilfe eines
Entscheidungsbaumes (Decision Tree) und eines linearen Regressionsmodells
jeweils eine Prognose ab, wie viele Menschen in Deutschland im Jahr 2100 leben
werden. Unterscheiden sich die beiden Prognosen?

#### Überblick über die Daten

Laden Sie die csv-Datei `population_germany.csv`. Welche Daten enthält die
Datei? Wie viele Datenpunkte sind vorhanden? Wie viele und welche Merkmale gibt
es? Sind die Daten vollständig? Welche Datentypen haben die Merkmale?

```{code-cell}
#
```

#### Statistik der numerischen Daten

Erstellen Sie eine Übersicht der statistischen Kennzahlen für die numerischen
Daten. Visualisieren Sie anschließend die statistischen Kennzahlen mit Boxplots.
Interpretieren Sie die statistischen Kennzahlen. Gibt es Ausreißer? Sind die
Werte plausibel?

```{code-cell}
#
```

#### Statistik der kategorialen Daten

Erstellen Sie eine Übersicht der Häufigkeiten für die kategorialen Daten.
Visualisieren Sie anschließend die Häufigkeiten mit Barplots. Interpretieren Sie
die Häufigkeiten. Sind die Werte plausibel?

```{code-cell}
#
```

#### Ursache - Wirkung

Erstellen Sie einen Scatterplot mit dem Jahr auf der x-Achse und der Population
auf der y-Achse. Beschriften Sie den Scatterplot sinnvoll. Vermuten Sie einen
Zusammenhang zwischen Jahr und Bevölkerung? Was fällt Ihnen generell auf? Können Sie die Besonderheiten mit Geschichtswissen erklären?

```{code-cell}
#
```

#### Lineares Regressionsmodell

Adaptieren Sie die Daten. Wählen Sie als Input das Jahr und als Output die
Population. Trainieren Sie ein lineares Regressionsmodell und lassen Sie den
Score berechnen und ausgeben.

```{code-cell}
#
```

#### Entscheidungsbaum/Decision Tree

Lassen Sie nun einen Entscheidungsbaum/Decision Tree trainieren und den Score
ausgeben.

Tipp: Das Scikit-Learn-Modell heißt DecisionTreeRegressor.

```{code-cell}
#
```

#### Bewertung und Prognose

Für welches Modell würden Sie sich entscheiden? Begründen Sie Ihre Wahl.

Lassen Sie dann sowohl das lineare Regressionsmodell als auch den
Entscheidungsbaum die Populationen von 1800 bis 2100 prognostizieren. Verwenden
Sie dazu den folgenden Datensatz:

```python
prognosedaten = pd.DataFrame({
    'Jahr': range(1800, 2101)
})
```

Visualisieren Sie die Prognosen zusammen mit den gemessenen Populationen in
einem gemeinsamen Scatterplot.

Tipp: Mit

`fig.add_scatter(x = prognosedaten['Jahr'],y = prognose_linear, name='lineare Regression')``

können Sie einen weiteren Scatterplot zu einem schon existierenden (hier in der Variable `fig` gespeichert) hinzufügen.

Welches Modell würden Sie nach der Visualisierung bevorzugen?

```{code-cell}
#
```

## Aufgabe 7.2

Eine Firma erhebt statistische Daten zu ihren Verkaufszahlen (angegeben in
Tausend US-Dollar) abhängig von dem eingesetzten Marketing-Budget in den
jeweiligen Sozialen Medien (Quelle siehe
<https://www.kaggle.com/datasets/fayejavad/marketing-linear-multiple-regression>).

Erstellen Sie eine explorative Datenanalyse (EDA). Trainieren Sie dann
ML-Modelle und bewerten Sie, bei welchem sozialen Medium sich am ehesten lohnt zu investieren.

### Überblick über die Daten Marketing

Laden Sie die csv-Datei `marketing_data.csv`. Welche Daten enthält die
Datei? Wie viele Datenpunkte sind vorhanden? Wie viele und welche Merkmale gibt
es? Sind die Daten vollständig? Welche Datentypen haben die Merkmale?

```{code-cell}
#
```

#### Statistik der numerischen Daten Marketing

Erstellen Sie eine Übersicht der statistischen Kennzahlen für die numerischen
Daten. Visualisieren Sie anschließend die statistischen Kennzahlen mit Boxplots.
Interpretieren Sie die statistischen Kennzahlen. Gibt es Ausreißer? Sind die
Werte plausibel?

```{code-cell}
#
```

#### Ursache - Wirkung Marketing

Erstellen Sie zuerst eine Scattermatrix, um Ursache und Wirkung zu analysieren.
Lassen Sie dann die Korrelationsmatrix als Heatmap anzeigen und interpretieren
Sie das Ergebnis.

```{code-cell}
#
```

#### Lineare Regressionsmodelle

Trainieren Sie drei lineare Regressinsmodelle, jeweils mit einem anderen Merkmal
als Input, d.h. mit jeweils `youtube`, `facebook`, `newspaper`. Adaptieren Sie
dazu passend die Daten. Lassen Sie jeweils den Score berechnen und ausgeben.

```{code-cell}
#
```

#### Finales Modell

Trainieren Sie nun als finales Modell ein multiples Regressionsmodell und
stellen Sie mit den Koeffizienten (Gewichten) und dem Achsenabschnitt (Bias) die dazugehörige Modellfunktion auf.

```{code-cell}
#
```
