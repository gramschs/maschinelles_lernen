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

# 13. Logistische Regression

Das binäre Klassifikationsverfahren des Perzeptrons sortiert Datensätze in zwei
Klassen. Ob dabei die Entscheidung für eine Klasse sehr knapp getroffen wird
oder eindeutig erscheint, wird dabei nicht transparent. Im Folgenden
beschäftigen wir uns mit einem binären Klassifikationsverfahren namens
logistische Regression, das auf einer Prognose der Klassenwahrscheinlichkeit
basiert.

## 13.1 Logistische Regression ersetzt Perzeptron

### Lernziele

* Sie können die **logistische Funktion** zeichnen.
* Sie können die **Funktionsgleichung der logistischen Funktion** angeben.
* Sie können das **logistische Regressionsmodell** schematisch darstellen.

### Perzeptron ungeeignet bei nicht trennbaren Daten

Das Perzeptron ist ein simples ML-Verfahren für die binäre Klassifikation.
Allerdings ist das Perzeptron nur für trennbare Datensätze geeignet. Dazu
schauen wir uns zwei Beispiele aus dem Männerfußball an. Ziel der folgenden
Klassifikationsaufgabe ist es, anhand des Marktwertes eines Vereines zu
klassifizieren, ob der Verein in der Bundesliga oder in der 2. Bundesliga
spielt.

Dazu laden wir als erstes den Datensatz `20220801_Marktwert_Bundesliga.csv`
(Download
[csv](https://nextcloud.frankfurt-university.de/s/GESBZzRyXq6dLNC/download);
Quelle: <https://www.transfermarkt.de>; Stichtag: 01.08.2022).

```{code-cell} ipython3
:tags: [output-scroll]

import pandas as pd
data_raw = pd.read_csv('data/20220801_Marktwert_Bundesliga.csv', skiprows=5, header=0, index_col=0)
data_raw.info()
```

Die ersten fünf Einträge lauten wie folgt:

```{code-cell} ipython3
data_raw.head()
```

Die erste Spalte enthält den Vereinsnamen, die zweite Spalte listet die
Ligazugehörigkeit (Bundesliga, 2. Bundesliga oder 3. Liga), die dritte Spalte
beinhaltet den Marktwert des Vereins in Mio. Euro und die vierte Spalte listet
die Anzahl der Spieler.

Welche Ligen sind denn im Datensatz vertreten?

```{code-cell} ipython3
data_raw['Ligazugehörigkeit'].unique()
```

Zuerst erkunden wie die Daten und lassen uns den MArktwert abhängig von der
Ligazugehörigkeit visualisieren.

```{code-cell} ipython3
import plotly.express as px

fig = px.box(data_raw, x = 'Ligazugehörigkeit', y = 'Wert',
             title='Deutscher Vereinsfußball der Männer (Stichtag: 1.8.2022)')
fig.show()
```

In der Bundesliga gibt es bezogen auf den Kaderwert drei Ausreißer, in der 3.
Liga gibt es einen Ausreißer. Zoom man in die Visualisierung hinein, so sehen
wir, dass ein Marktwert von 36 Mio. Euro die Erstligisten von den Zweitligisten
trennt. Daher wäre für eine Klassifikation Bundesliga vs. 2. Bundesliga ein
Perzeptron trainierbar. Eine solche Trennung funktioniert bei den Vereinen der

1. Bundesliga und der 3. Liga nicht. Der minimale Marktwert der 2. Bundesliga
(8.83 Mio. EUR) ist niedriger als der maximale Marktwert der 3. Liga (13.05 Mio.
EUR). Wir betrachten nun nur noch 2. und 3. Liga und stellen die folgende Frage:

Kann aufgrund des Marktwertes die Zugehörigkeit zur 2. Bundesliga oder 3. Liga
prognostiziert werden? Wir visualisieren die einzelnen Datenpunkte mit der
Ursache 'Wert' auf der x-Achse und 'Ligazugehörigkeit' auf der y-Achse.

```{code-cell} ipython3
data = data_raw[ data_raw['Ligazugehörigkeit'] != 'Bundesliga' ]

fig = px.scatter(data, x = 'Wert', y = 'Ligazugehörigkeit',
             title='Deutscher Vereinsfußball der Männer (Stichtag: 1.8.2022)')
fig.show()
```

Als nächstes ersetzen wir die Klassenbezeichnungen durch numerische Werte. Mit
Bezeichnungen wie "3. Liga" und "2. Bundesliga" kann Python nämlich nicht
rechnen. Bei einem binären Klassifikationsverfahren wie hier werden hierfür
üblicherweise die Zahlen 0 und 1 verwendet, also

* '3. Liga --> 0'
* '2. Bundesliga --> 1'

Das Klassifikationsproblem lautet also: angenommen, ein
Verein hat den Marktwert x. Gehört der Verein dann zur Klasse 1 (= 2.
Bundesliga)?

Als zweites verabschieden wir uns von dem Perzeptron. Anstatt eine Gerade zur
Trennung einzuziehen, nehmen wir eine S-förmige Funktion. Diese S-förmige
Funktion interpretieren wir als **Wahrscheinlichkeit** der Ligazugehörigkeit.
Die Werte liegen dabei zwischen 0 und 1. Bei einer Wahrscheinlichkeit von 1 sind
wir also 100 % sicher, dass der Verein zur Klasse 1 (= 2. Bundesliga) gehört.
Bei einem Wert von 0.7 gehen wir mit 70 % Wahrscheinlichkeit davon aus, dass der
Verein zur Klasse 1 (= 2. Bundesliga) gehört.

Als drittes verwandeln wir die Wahrscheinlichkeit mit Werten *zwischen* 0 und 1
in die Klassen 0 oder 1. Dazu nutzen wir die Heaviside-Funktion. Die komplette
Vorgehensweise ist in der folgenden Grafik dargestellt.

![Die S-förmige schwarz gestrichelte Kurve gibt die Wahrscheinlichkeit an, dass ein Verein zur 2. Bundesliga gehört. Da nicht nach einer Wahrscheinlichkeit gefragt ist, sondern nur nach 2. Bundesliga -- ja oder nein -- werden alle Vereine mit einer Wahrscheinlichkeitgrößer 50 % (also $\geq 0.5$) als Zweitligisten klassifiziert.](https://gramschs.github.io/book_ml4ing/_images/bundesliga_decision_function_annotated.pdf)

Jetzt brauchen wir eine Funktion für die S-förmige Kurve, das wir dann mittels
eines ML-Verfahrens an unsere Trainingsdaten anpassen können.

+++

### Logistische Funktion ersetzt Heaviside-Funktion

Beim Perzeptron wird auf die gewichtete Summe von Inputs die Heaviside-Funktion
angewandt. So simpel die Heaviside-Funktion $\phi$ auch ist, sie hat einen
entscheidenen Nachteil. Die Heaviside-Funktion ist unstetig, sie springt von
Null auf Eins. Diese Sprungstelle hat die **logistische Funktion** nicht.

Die logistische Funktion ist defininiert als

$$\sigma(z) = \frac{1}{1+e^{-z}}.$$

Um die logistische Funktion abzukürzen, verwenden wir dabei den griechischen
Buchstaben Sigma $\sigma$, weil die logistische Funktion auch
**Sigmoid-Funktion** genannt wird. In der folgenden Abbildung ist der
Funktionsgraph der logistischen Funktion dargestellt.

![Funktionsgraph der logistischen Funktion, auch Sigmoid-Funktion genannt](https://gramschs.github.io/book_ml4ing/_images/plot_logit_function.pdf)

Damit haben wir die Bausteine des logistischen Regressionmodells komplettiert.
Genau wie bei der linearen Regression oder beim Perzeptron werden zuerst die
einzelnen Inputs gewichtet und aufsummiert. Auf die gewichtete Summe wird dann
die logistische Funktion als Aktivierungsfunktione angewendet. Das Ergebnis ist
die Wahrscheinlichkeit für die Klasse mit der Bezeichnung 1 (in unserem Beispiel
die Zugehörigkeit zur 2. Bundesliga). Zuletzt wird noch die Heaviside-Funktion
als Schwellenwertfunktion angewendet, um aus der Wahrscheinlichkeit eine Klasse
zu machen.

Schematisch dargestellt sieht das logistische Regressionsmodell also
folgendermaßen aus:

![Das logistische Regressionsmodell als neuronales Netz formuliert](https://gramschs.github.io/book_ml4ing/_images/topology_logistic_regression.svg)

Mathematisch formuliert lautet das logistische Regressionsmodell folgendermaßen:

$$\hat{P} = \sigma\left(\sum_{i=0}^{N} x_i \omega_i\right) = \frac{1}{1+e^{-\sum x_i \omega_i}}
\rightsquigarrow \hat{y} = \begin{cases} 0: & \hat{P} < 0.5 \\ 1: & \hat{P} \geq 0.5 \end{cases} $$

### Zusammenfassung und Ausblick Kapitel 13.1

In diesem Abschnitt haben wir das logistische Regressionsmodell formuliert. Als
nächstes betrachten wir ein Lernverfahren, um die Gewichte des logistischen
Regressionsmodells zu erlernen.

## 13.2 Logistische Regression lernt mit Maximum-Likelihood

### Lernziele

* Sie kennen die Begriffe **Fehlermaß**, **Kostenfunktion** und **Verlustfunktion**.

### Wir basteln eine Kostenfunktion

Bei der linearen Regression werden die prognostizierten Daten $\hat{y}$ mit den
echten Daten $y$ verglichen. Dazu wird der Abstand der beiden Werte berechnet
und quadriert, damit der Fehler immer positiv ist. Zuletzt wird der Mittelwert
aller Fehler für alle Trainingsdaten gebildet. Die Gewichte des linearen
Regressionsmodells werden dann so berechnet, dass die mittlere Summe der
Fehlerquadrate möglichst klein, also minimiert wird. In der Wirtschaft
verursachen Fehler Kosten. Daher wird ein **Fehlermaß**, in der ML Community
auch **Kostenfunktion (cost function)** genannt. Da das Fehlermaß bzw. die
Kostenfunktion minimiert werden soll, sprechen mache auch von **Verlustfunktion
(loss function)**.

So ein Maß, egal wie wir es am Ende nennen, brauchen wir hier auch. Wir müssen
beurteilen können, ob die gewählten Gewichte gut oder schlecht sind, also kleine
Fehler oder große Fehler produzieren. Nur passt diesmal leider die kleinste
Summe der Fehlerquadrate nicht. Bei der binären Klassifikation gibt es nämlich
nur zwei Fehler. Entweder, ich prognostiziere Klasse 1, aber es wäre Klasse 0
gewesen. Oder ich prognostiziere *nicht* Klasse 1, obwohl Klasse 1 richtig
gewesen wäre. Damit haben wir auch zwei Fehlerarten. Für beide Fehlerarten
führen wir ein eigenes Fehlermaß bzw. Kostenfunktion bzw. Verlustfunktion ein.

Damit wir uns bei den folgenden Überlegungen etwas Schreibarbeit sparen können,
kürzen wir die gewichtete Summe mit $z$ ab, also

$$z = \sum_{i=0}^{N} x_i \omega_i.$$

#### Kostenfunktion, wenn Klasse 1 richtig wäre

$\sigma(z)$ gibt die Wahrscheinlichkeit an, dass der prognostizierte Output
$\hat{y}$ in Klasse 1 einsortiert werden sollte. Wenn das richtig ist, also
tatsächlich für den echten Output $y=1$ gilt, dann ist der Fehler bzw. sind die
Kosten 0. Sollte jedoch das logistische Regressionsmodell $\sigma(z)$ fehlerhaft
Richtung 0 tendieren, so sollten auch hohe Kosten anfallen. Die Kostenfunktion
muss also stark steigen, je mehr sich die Wahrscheinlichkeit
$\sigma(z)$ der 0 nähert.

Es gibt einige Funktionen, die ein solches Verhalten beschreiben. Wir verwenden
die folgende Kostenfunktion:

$$c_{1}(z) =
- \log\left(\sigma(z)\right), \quad \text{ falls } y=1.$$

Vielleicht mag man sich jetzt wundern, wie man auf diese Funktion kommt. Diese
Funktion wie auch die nachfolgenden Überlegungen basieren auf der
[Maximum-Likelihood-Methode](https://de.wikipedia.org/wiki/Maximum-Likelihood-Methode),
einem sehr bekannten statistischen Verfahren.

Am einfachsten ist es wahrscheinlich, sich den Graph der Funktion anzusehen. Da
$\sigma(z)$ für jede Kombination von Inputs $x_i$ und Gewichten $\omega_i$
zwischen 0 und 1 liegt, brauchen wir die Kostenfunktion $c_1$ auch nur auf dem
Intervall $[0, 1]$ plotten.

```{code-cell} ipython3
import numpy as np
import plotly.express as px

sigma_z = np.linspace(0,1)[1:]
cost_1 = - np.log(sigma_z)

fig = px.line(x = sigma_z, y = cost_1,
              title='Kostenfunktion, falls Klasse 1 korrekt (y=1)',
              labels={'x': 'sigma(z)', 'y': 'Kosten c_1'})
fig.show()
```

#### Kostenfunktion, wenn Klasse 1 nicht richtig wäre

Nun betrachten wir den zweiten Fall. Der echte Output soll nicht der Klasse 1
angehören, also $y=0$. Sollte die Wahrscheinlichkeit des logistischen
Regressionsmodells $\sigma(z)$ in Richtung 0 gehen, so sollen kaum Kosten
anfallen. Falls korrekterweise die Wahrscheinlicheit 0 ist, so sollen gar keine
Kosten anfallen, denn der Fehler geht gegen 0. Und umgekehrt, wenn die
Wahrscheinlichkeit des logistischen Regressionsmodells Richtung 1 tendiert, so
sollen die Kosten stark steigen. Wir nehmen als zweite Kostenfunktion für den
Fall $y=0$ die folgende Funktion:

$$c_{0}(z) = - \log\left(1-\sigma(z)\right), \quad \text{ falls } y=0.$$

```{code-cell} ipython3
sigma_z = np.linspace(0,1)[:-1]
cost_0 = - np.log(1 - sigma_z)

fig = px.line(x = sigma_z, y = cost_0,
              title='Kostenfunktion, falls Klasse 1 nicht korrekt (y=0)',
              labels={'x': 'sigma(z)', 'y': 'Kosten c_0'})
fig.show()
```

#### Beide Kostenfunktionen kombiniert

Zusammengefasst haben wir also zwei Kostenfunktionen für die beiden Fälle.

$$
c(z) =
\begin{cases}
c_{0}(z): & y = 0 \\
c_{1}(z): & y = 1 \\
\end{cases} \qquad = \qquad
\begin{cases}

-\log\left(1-\sigma(z)\right): & y = 0 \\
-\log\left(\sigma(z)\right): & y = 1 \\
\end{cases}.

$$

Es wäre besser, die Fallunterscheidung weglassen zu können und nur eine
Kostenfunktion zu betrachten. Dafür gibt es einen bekannten Trick, die
sogenannte *Konvexkombination* beider Funktionen funktioniert:

$$
c(z) = y\cdot c_{1}(z) + (1-y) c_{0}(z).
$$

Wenn wir einmal $y=0$ einsetzen und einmal $y=1$, so sehen wir, dass entweder
$c_{1}(z)$ oder aber $c_{0}(z)$ übrig bleibt — so wie gewünscht.

Also lautet die konvex kombiniert Kostenfunktion für einen einzelnen
Trainingsdatensatz mit den Inputs $x_i$ und den Gewichten $\omega_i$

$$ c(\mathbf{x}; \boldsymbol{\omega}) = - y\cdot
\log\left(\sigma\left(\sum_{i=0}^{N}x_i \omega_i\right)\right) - (1-y)
\log\left(\sigma\left(1-\sum_{i=0}^{N}x_i \omega_i\right)\right) $$

mit der logistischen Funktion $\sigma(z) = \frac{1}{1+e^{-z}}$.

+++

### Lernregel für die logistische Regression

Wie bei der linearen Regression wird nun die Kostenfunktion für jeden einzelnen Trainingsdatensatz berechnet und anschließend wird über alle Kosten der Mittelwert gebildet. Nun müssen die Gewichte so gewähle werden, dass der Mittelwert der Kosten minimiert wird. Anders als bei der linearen Regression kann dafür nicht einfach eine Gleichgun gelöst werden, die die Gewichte berechnet. Stattdessen muss wie beim Pereptron ein iteratives Verfahren verwendet werden. Damit ist gemeint, dass mit zufällig gewählten Gewichten die mittleren Kosten berechnet werden. Danach wird solange an den Gewichten gedreht, bis ein Minumum der mittleren Kosten erreicht wird. Wie an den Gewichten gedreht wird, gibt der **Gradient der Kostenfunktion** vor.

In dieser Vorlesung gehen wir *nicht* auf das sehr mathematiklastige Thema Gradientenverfahren ein. Der KI-Campus bietet einen spielerischen Zugang zu dem Thema mit weiterführenden Texten an: [https://learn.ki-campus.org](https://learn.ki-campus.org/courses/explorables-schule-imaginary2021/items/7H9nZI186JgjC8jOjxdbaT) (eine Anmeldung ist dafür erforderlich, aber kostenfrei).

## 13.3 Logistische Regression mit Scikit-Learn

### Lernziele

* Sie können ein logistisches Regressionsmodell mit Scikit-Learn trainieren.

### LogisticRegression

Scikit-Learn bietet ein logistisches Regressionsmodell an, bei dem verschiedene
Gradientenverfahren im Hintergrund die Gewichte bestimmen, die zu einer
minimalen mittleren Kostenfunktion führen. Die Dokumentation zu dem logistischen
Regressionsmodell findet sich hier: [scikit-learn.org →
LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Wir wenden nun das Scikit-Learn-Modell auf unser Beispiel der binären
Klassifikation "Ligazugehörigkeit abhängig vom Marktwert" deutscher
Fußballvereine an. Dazu wiederholen laden wir die Daten und filtern zunächst
nach Vereinen der 2. Bundesliga oder der 3. Liga.

```{code-cell} ipython3
# import all data
import pandas as pd
data_raw = pd.read_csv('data/20220801_Marktwert_Bundesliga.csv', skiprows=5, header=0, index_col=0)

# filter wrt 2. Bundesliga and 3. Liga
data = data_raw[ data_raw['Ligazugehörigkeit'] != 'Bundesliga' ]

# print all data samples
data.head(38)
```

Als nächstes formulieren wir das Klassifikationsproblem: Gegeben ist ein Verein mit seinem Marktwert. Spielt der Verein in der 2. Bundesliga?

Die Klasse `2. Bundesliga` wird in den Daten als `1` codiert, da der ML-Algorithmus nur mit numerischen Daten arbeiten kann. Den String `3. Liga` ersetzen wir in den Trainingsdaten durch eine `0`.

```{code-cell} ipython3
# encode categorical data
data.replace('2. Bundesliga', 1, inplace=True)
data.replace('3. Liga', 0, inplace=True)
```

Jetzt können wir das logistische Regressionsmodell laden:

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
```

Die Daten werden jetzt in Matrizen gepackt und in Trainings- und Testdaten unterteilt:

```{code-cell} ipython3
import numpy as np
from sklearn.model_selection import train_test_split

X = data[['Wert']]
y = data['Ligazugehörigkeit']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Danach können wir das logistische Regressionsmodell trainieren:

```{code-cell} ipython3
logistic_regression.fit(X_train, y_train)
```

Und dann als nächstes beurteilen, wie viele der Testdaten korrekt klassfiziert werden.

```{code-cell} ipython3
logistic_regression.score(X_test, y_test)
```

90 % der Testdaten werden korrekt klassifiziert. Mit einer anderen Aufteilung in
Trainings- und Testdaten können wir auch höhere Erkennungsraten erzielen.
Beispielsweise führt ein Split mit `random_state=1` zu einer 100 % genauen
Klassifikation der Testdaten.

Als nächstes lassen wir Python alle Daten zusammen mit der
Wahrscheinlichkeitsfunktion visualisieren.

```{code-cell} ipython3
# extrahiere die Gewichte des logistischen Regressionsmodells
gewichte = np.concatenate((logistic_regression.intercept_, logistic_regression.coef_[:,0]))
print(f'Gewichte: {gewichte}')

# definiere Wahrschinelichkeitsfunktion
def wahrscheinlichkeitsfunktion(x, w):
    z = w[0] + x * w[1]
    return 1/(1+np.exp(-z))

# stelle Wartetabelle der Wahrscheinlichkeitsfunktion auf
x = np.linspace(0, 35)
sigma_z = wahrscheinlichkeitsfunktion(x, gewichte)

# trenne Daten gemäß Ligazugehörigkeit
data_zweite_liga = data[data['Ligazugehörigkeit'] == 1]
data_dritte_liga = data[data['Ligazugehörigkeit'] == 0]
```

```{code-cell} ipython3
import plotly.express as px
import plotly.graph_objects as go

fig3 = px.scatter(data_dritte_liga, x = 'Wert', y = 'Ligazugehörigkeit')
fig2 = px.scatter(data_zweite_liga, x = 'Wert', y = 'Ligazugehörigkeit')
fig_model = px.line(x = x, y = sigma_z)

fig = go.Figure(fig_model.data + fig2.data + fig3.data)
fig.update_layout(title='Klassifikation 2. Bundesliga / 3. Liga',
                  xaxis_title='Marktwert',
                  yaxis_title='Ligazugehörigkeit')
fig.show()
```

Aus der Visualisierung der Wahrscheinlichkeitsfunktion können wir grob
abschätzen, bei welchem Marktwert ein Verein als Zweit- oder Drittligist
klassifiziert wird. Die Wahrscheinlichkeitsfunktion schneidet die 50 %
Grenzlinie ungefähr bei einem Marktwert von 11 Mio. Euro. Etwas genauer können
wir diese Grenze durch das Kommando `fsolve` aus dem Scipy-Modul bestimmen
lassen:

```{code-cell} ipython3
from scipy.optimize import fsolve

x_grenze =  fsolve(lambda x: wahrscheinlichkeitsfunktion(x, gewichte) - 0.5, 11.0)
print('Grenze des Marktwertes: {:.2f} Mio. Euro'.format(x_grenze[0]))
```

### Zusammenfassung

In diesem Abschnitt haben wir an einem Beispiel gesehen, wie das logistische
Regressionsmodell von Scikit-Learn trainiert und bewertet wird.

## Klausurtypische Aufgaben

Der folgende Datensatz enthält die Preise und Eigenschaften von Diamanten. Die
Eigenschaften sind:

* Karat (Gewicht des Diamanten)
* Schliff (Qualität: befriedigend, gut, sehr gut, erstklassig, ideal)
* Farbe des Diamanten (von J (schlechteste) bis D (beste))
* Reinheit - ein Maß für die Klarheit des Diamanten (I1 (schlechteste), SI2,
  SI1, VS2, VS1, VVS2, VVS1, IF (beste))
* Tiefe (Gesamttiefe in Prozent = z / Mittelwert (x, y) = 2 * z / (x + y))
* Tafel (Breite der Oberseite des Diamanten im Verhältnis zur breitesten Stelle)
* Preis (in US-Dollar)
* x - Länge in mm
* y - Breite in mm
* z - Tiefe in mm

Bearbeiten Sie die folgenden Aufgaben. Vorab können Sie die folgenden Module
importieren. Schreiben Sie Ihre Antworten als Kommentar oder in eine
Markdown-Zelle. Lassen Sie das Jupyter Notebook am Ende noch einmal komplett
ausführen, bevor Sie es abgeben.

```{code-cell} ipython3
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
# from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
# from sklearn.svm  import SVC, SVR
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# pd.DataFrame.iteritems = pd.DataFrame.items
```

### Explorative Datenanalyse

#### Aufgabe 1: Import und Bereinigung der Daten

Importieren Sie die Daten 'diamonds_DE.csv'. Verschaffen Sie sich einen
Überblick und beantworten Sie folgende Fragen in einer Markdown-Zelle:

* Wie viele Diamanten enthält die Datei?
* Wie viele Merkmale/Attribute/Eigenschaften/Features sind in den Daten enthalten?
* Sind alle Einträge gültig? Wenn nein, wie viele Einträge sind nicht gültig?
* Welchen Datentyp haben die einzelnen Attribute/Eigenschaften/Features?

Falls der Datensatz ungültige Werte aufweist oder Unstimmigkeiten enthält,
bereinigen Sie ihn.

```{code-cell} ipython3
#
```

#### Aufgabe 2: Statistische Kennzahlen der numerischen Eigenschaften

* Ermitteln Sie von den numerischen Eigenschaften die statistischen Kennzahlen
  und visualisieren Sie sie. Verwenden Sie beim Plot eine aussagefähige
  Beschriftung.
* Interpretieren Sie jede Eigenschaft anhand der statistischen Kennzahlen und
  der Plots.
* Bereinigen Sie bei Ungereimtheiten den Datensatz weiter.
* Entfernen Sie Ausreißer.

```{code-cell} ipython3
#
```

#### Aufgabe 3: Statistische Kennzahlen (kategoriale Eigenschaften)

* Ermitteln Sie, wie häufig jeder Wert einer Kategorie in der jeweiligen Spalte
  vorkommt.
* Lassen Sie die Anzahl der Werte auch visualisieren. Beschriften Sie die
  Diagramme mit einem aussagefähigen Titel.
* Fassen Sie die Ergebnisse bzw. die Interpretation davon jeweils kurz zusammen
  (in einer Markdown-Zelle).

```{code-cell} ipython3
#
```

### ML-Modelle

#### Aufgabe 4: Regression

Ziel der Regressionsaufgabe ist es, den Preis der Diamanten zu prognostizieren.

* Wählen Sie zwei Regressionsmodelle aus.
* Wählen Sie für jedes der zwei Modelle eine oder mehrere Eigenschaften aus, die
  Einfluss auf den Preis haben könnten. Begründen Sie Ihre Auswahl.
* Adaptieren Sie die Daten jeweils passend zu den von Ihnen gewählten Modellen.
* Falls notwendig, skalieren Sie die Daten.
* Führen Sie einen Split der Daten in Trainings- und Testdaten durch.
* Trainieren Sie jedes ML-Modell.
* Validieren Sie jedes ML-Modell bzgl. der Trainingsdaten und der Testdaten.
* Bewerten Sie abschließend: welches der zwei Modelle würden Sie empfehlen?
  Begründen Sie Ihre Empfehlung.

```{code-cell} ipython3
#
```

#### Aufgabe 5: Klassifikation

Ziel der Klassifikationsaufgabe ist es, die Preisklasse "billig" oder "teuer"
der Diamanten zu prognostizieren.

* Vorbereitung: Filtern Sie die Daten nach den Diamanten, deren Preis kleiner
oder gleich dem Median aller Preise ist. Diese Diamanten sollen als "billig"
klassfiziert werden. Diamanten, deren Preis größer als der Median aller Preise
ist, sollen als "teuer" klassifiziert werden. Speichern Sie dieses neue Merkmal
in einer neuen Spalte "Preisklasse".
* Trainieren Sie einen Entscheidungsbaum (Decision Tree) mit den Merkmalen
  "Karat" und "Schliff".
* Adaptieren Sie die Daten.
* Falls notwendig, skalieren Sie die Daten.
* Führen Sie einen Split der Daten in Trainings- und Testdaten durch.
* Führen Sie eine Gittersuche für
  * die Baumtiefe (2, 3) und
  * die minimale Anzahl an Samples pro Blatt (1, 2, 5) durch.
* Lassen Sie das beste Modell als Baum visualisieren.
* Bewerten Sie abschließend: ist das Modell für den Produktiveinsatz geeignet?
  Begründen Sie Ihre Bewertung.

```{code-cell} ipython3
#
```
