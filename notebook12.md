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

# 12. Perzeptron

Durch neuronale Netze, die tief verschachtelt sind (= tiefe neuronale Netze =
deep neural network), gab es im Bereich des maschinellen Lernens einen
Durchbruch. Neuronale Netze sind eine Technik aus der Statistik, die bereits in
den 1940er Jahren entwickelt wurde. Seit ca. 10 Jahren erlebt diese Technik
verbunden mit Fortschritten in der Computertechnologie eine Renaissance.

Neuronale Netze bzw. Deep Learning kommen vor allem da zum Einsatz, wo es kaum
systematisches Wissen gibt. Damit neuronale Netze erfolgreich trainiert werden
können, brauchen sie sehr große Datenmengen. Nachdem in den letzten 15 Jahren
mit dem Aufkommen von Smartphones die Daten im Bereich Videos und Fotos massiv
zugenommen haben, lohnt sich der Einsatz der neuronalen Netze für
Spracherkennung, Gesichtserkennung oder Texterkennung besonders.

Beispielsweise hat ein junges deutsches Start-Up-Unternehmen 2017 aus einem
neuronalen Netz zum Übersetzen Englisch <-> Deutsch eine Webanwendung entwickelt
und ins Internet gestellt, die meinen Alltag massiv beeinflusst:

> DeepL.com

Auf der Seite

> [https://www.deepl.com/en/blog/how-does-deepl-work](https://www.deepl.com/en/blog/how-does-deepl-work)

finden Sie einen kurzen Übersichtsartikel dazu, wie DeepL funktioniert.

Die Grundlage der neuronalen Netze ist das Perzeptron, mit dem wir uns in diesem
Kapitel beschäftigen.

## 12.1 Grundbaustein neuronaler Netze

Neuronale Netze sind sehr beliebte maschinelle Lernverfahren. Das einfachste künstliche neuronale Netz ist das **Perzeptron**. In diesem Abschnitt werden wir das Perzeptron vorstellen.

### Lernziele Kapitel 12.1

* Sie können das Perzeptron als mathematische Funktion formulieren und in dem Zusammenhang die folgenden Begriffe erklären:
  * gewichtete Summe (Weighted Sum),
  * Bias oder Bias-Einheit (Bias),
  * Schwellenwert (Threshold)  
  * Heaviside-Funktion (Heaviside Function) und
  * Aktivierungsfunktion (Activation Function).
* Sie können das Perzeptron als ein binäres Klassifikationsproblem des überwachten Lernens einordnen.

### Die Hirnzelle dient als Vorlage für künstliche Neuronen

1943 haben die Forscher Warren McCulloch und Walter Pitts das erste Modell einer vereinfachten Hirnzelle präsentiert. Zu Ehren der beiden Forscher heißt dieses Modell MCP-Neuron. Darauf aufbauend publizierte Frank Rosenblatt 1957 seine Idee einer Lernregel für das künstliche Neuron. Das sogenannte Perzeptron bildet bis heute die Grundlage der künstlichen neuronalen Netze. Inspiriert wurden die Forscher dabei durch den Aufbau des Gehirns und der Verknüpfung der Nervenzellen.

![Darstellung einer Nervenzelle](https://gramschs.github.io/book_ml4ing/_images/neuron_wikipedia.svg)

([Quelle:](https://de.wikipedia.org/wiki/Künstliches_Neuron#/media/Datei:Neuron_(deutsch)-1.svg) "Schematische Darstellung einer Nervenzelle" von Autor unbekannt. Lizenz: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/))

Elektrische und chemische Eingabesignale kommen bei den Dendriten an und laufen im Zellkörper zusammen. Sobald ein bestimmter Schwellwert überschritten wird, wird ein Ausgabesignal erzeugt und über das Axon weitergeleitet. Mehr Details zu Nervenzellen finden Sie bei [Wikipedia/Nervenzelle](https://de.wikipedia.org/wiki/Nervenzelle).

### Eine mathematische Ungleichung ersetzt das logische Oder

Das einfachste künstliche Neuron besteht aus zwei Inputs und einem Output. Dabei sind für die beiden Inputs nur zwei Zustände zugelassen und auch der Output besteht nur aus zwei verschiedenen Zuständen. In der Sprache des maschinellen Lernens liegt also eine **binäre Klassifikationsaufgabe** innerhalb des **Supervised Learnings** vor.

Beispiel:

* Input 1: Es regnet oder es regnet nicht.
* Input 2: Der Rasensprenger ist an oder nicht.
* Output: Der Rasen wird nass oder nicht.

Den Zusammenhang zwischen Regen, Rasensprenger und nassem Rasen können wir in einer Tabelle abbilden:

Regnet es? | Ist Sprenger an? | Wird Rasen nass?
-----------|------------------|-----------------
nein       | nein             | nein
ja         | nein             | ja
nein       | ja               | ja
ja         | ja               | ja

Wir schreiben ein kurzes Python-Programm, das abfragt, ob es regnet und ob der
Rasensprenger eingeschaltet ist. Dann soll der Python-Interpreter ausgeben, ob
der Rasen nass wird oder nicht.

```{code-cell}
# Eingabe
x1 = input('Regnet es (j/n)?')
x2 = input('Ist der Rasensprenger eingeschaltet? (j/n)')

# Verarbeitung
y = (x1 == 'j') or (x2 == 'j')

# Ausgabe
if y == True:
    print('Der Rasen wird nass.')
else:
    print('Der Rasen wird nicht nass.')
```

Für das maschinelle Lernen müssen die Daten als Zahlen aufbereitet werde, damit
die maschinellen Lernverfahren in der Lage sind, Muster in den Daten zu
erlernen. Anstatt "Regnet es? Nein." oder Variablen mit True/False setzen wir
jetzt Zahlen ein. Die Inputklassen kürzen wir mit x1 für Regen und x2 für
Rasensprenger ab. Die 1 steht für ja, die 0 für nein. Den Output bezeichnen wir
mit y. Dann lautet die obige Tabelle für das "Ist-der-Rasen-nass-Problem":

x1 | x2 | y
---|----|---
0  | 0  | 0
1  | 0  | 1
0  | 1  | 1
1  | 1  | 1

wir schreiben das obige Python-Programm um und verwenden die Integers 0 und 1
für die Eingaben.

```{code-cell}
# Eingabe
x1 = int(input('Regnet es (ja = 1 | nein = 0)?'))
x2 = int(input('Ist der Rasensprenger eingeschaltet? (ja = 1 | nein = 0)'))

# Verarbeitung
y = (x1 == 1) or (x2 == 1)

# Ausgabe
if y == True:
    print('Der Rasen wird nass.')
else:
    print('Der Rasen wird nicht nass.')
```

Nun ersetzen wir das logische ODER durch ein mathematisches Konstrukt:
Wenn die Ungleichung

$$x_1 \omega_1  +  x_2 \omega_2 \geq \theta$$

erfüllt ist, dann ist $y = 1$ oder anders ausgedrückt, der Rasen wird nass. Und
ansonsten ist $y = 0$, der Rasen wird nicht nass. Allerdings müssen wir noch die
**Gewichte** $\omega_1$ und $\omega_2$ (auf Englisch: weights) geschickt wählen.
Die Zahl $\theta$ ist der griechische Buchstabe Theta und steht als Abkürzung
für den sogenannten **Schwellenwert** (auf Englisch: threshold).

Beispielsweise $\omega_1 = 0.3$, $\omega_2=0.3$ und $\theta = 0.2$ passen:

* $0 \cdot 0.3 + 0 \cdot 0.3 = 0.0 \geq 0.2$ nicht erfüllt
* $0 \cdot 0.3 + 1 \cdot 0.3 = 0.3 \geq 0.2$ erfüllt
* $1 \cdot 0.3 + 0 \cdot 0.3 = 0.3 \geq 0.2$ erfüllt
* $1 \cdot 0.3 + 1 \cdot 0.3 = 0.6 \geq 0.2$ erfüllt

Wir schreiben erneut das Python-Programm um und ersetzen das logische ODER durch
die linke Seite der Ungleichung. Dann vergleichen wir anschließend mit $0.2$, um
zu entscheiden, ob der Rasen nass wird oder nicht.

```{code-cell}
# Eingabe
x1 = int(input('Regnet es (ja = 1 | nein = 0)?'))
x2 = int(input('Ist der Rasensprenger eingeschaltet? (ja = 1 | nein = 0)'))

# Verarbeitung
y = 0.3 * x1 + 0.3 * x2

# Ausgabe
if y >= 0.2:
    print('Der Rasen wird nass.')
else:
    print('Der Rasen wird nicht nass.')
```

### Die Heaviside-Funktion ersetzt die Ungleichung

Noch sind wir aber nicht fertig, denn auch die Frage "Ist die Ungleichung
erfüllt oder nicht?" muss noch in eine mathematische Funktion umgeschrieben
werden. Dazu subtrahieren wir zuerst auf beiden Seiten der Ungleichung den
Schwellenwert $\theta$:

$$-\theta + x_1 \omega_1  +  x_2 \omega_2 \geq 0.$$

Damit haben wir jetzt nicht mehr einen Vergleich mit dem Schwellenwert, sondern
müssen nur noch entscheiden, ob der Ausdruck $-\theta + x_1 \omega_1 + x_2
\omega_2$ negativ oder positiv ist. Bei negativen Werten, soll $y = 0$ sein und
bei positiven Werten (inklusive der Null) soll $y = 1$ sein. Dafür gibt es in
der Mathematik eine passende Funktion, die sogenannte
[Heaviside-Funktion](https://de.wikipedia.org/wiki/Heaviside-Funktion) (manchmal
auch Theta-, Stufen- oder Treppenfunktion genannt).

![Schaubild der Heaviside-Funktion](https://gramschs.github.io/book_ml4ing/_images/heaviside_wikipedia.svg)

([Quelle:](https://de.wikipedia.org/wiki/Heaviside-Funktion#/media/Datei:Heaviside.svg) "Verlauf der Heaviside-Funktion auf $\mathbb{R}$" von Lennart Kudling. Lizenz: gemeinfrei)

Definiert ist die Heaviside-Funktion folgendermaßen:

$$\Phi(x) = \begin{cases}0:&x<0\\1:&x\geq 0\end{cases}$$

In dem Modul NumPy ist die Heaviside-Funktion schon hinterlegt, siehe
> <https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html>

Wir visualisieren die Heaviside-Funktion für das Intervall $[-3,3]$ mit 101
Punkten. Setzen Sie das zweite Argument einmal auf 0 und einmal auf 2. Was
bewirkt das zweite Argument? Sehen Sie einen Unterschied in der Visualisierung?
Erhöhen Sie auch die Anzahl der Punkte im Intervall. Wählen Sie dabei immer eine
ungerade Anzahl, damit die 0 dabei ist.

```{code-cell}
import pandas as pd
import plotly.express as px
import numpy as np

x = np.linspace(-3, 3, 101)
y0 = np.heaviside(x, 0)  # an der Stelle x=0 ist y=0
y1 = np.heaviside(x, 2)  # an der Stelle x=0 ist y=2

# Daten für Plotly Express vorbereiten
df = pd.DataFrame({'x': x, 'y0': y0, 'y1': y1})

# Visualisierung
fig = px.scatter(df, x='x', y=['y0', 'y1'], title='Heaviside-Funktion')
fig.update_layout(
    xaxis_title='x',
    yaxis_title='y'
)
fig.show()
```

Mit der Heaviside-Funktion können wir nun den Vergleich in der
Programmverzweigung mit $0.2$ durch eine direkte Berechnung ersetzen. Schauen
Sie sich im folgenden Programm-Code an, wie wir jetzt ohne logisches Oder und
ohne Programmverzweigung if-else auskommen.

```python
# Import der notwendigen Module
import numpy as np

# Eingabe
x1 = int(input('Regnet es (ja = 1 | nein = 0)?'))
x2 = int(input('Ist der Rasensprenger eingeschaltet? (ja = 1 | nein = 0)'))

# Verarbeitung
y = np.heaviside(-0.2 + 0.3 * x1 + 0.3 * x2, 1.0)

# Ausgabe
ergebnis_als_text = ['Der Rasen wird nicht nass.', 'Der Rasen wird nass.']
print(ergebnis_als_text[int(y)])
```

### Das Perzeptron mit mehreren Eingabewerten

Das Perzeptron für zwei Eingabewerte lässt sich in sehr natürlicher Weise auf
viele Eingabewerte verallgemeinern, die auch mehrere Zustände annehmen können.
Bei den Outputs bleiben wir jedoch dabei, dass nur zwei Zustände angenommen
werden können, die wir mit 0 und 1 bezeichnen. Wir betrachten also weiterhin
binäre Klassifikationsaufgaben.

Wenn wir nicht nur zwei, sondern $n$ Eingabewerte $x_i$ haben, brauchen wir
entsprechend auch $n$ Gewichte $\omega_i$. Die Eingabewerte können wir in einem
Spaltenvektor zusammenfassen, also

$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}.$$

Auch die Gewichte können wir in einem Spaltenvektor zusammenfassen, also

$$\boldsymbol{\omega} = \begin{pmatrix} \omega_1 \\ \omega_2 \\ \vdots \\
\omega_n\end{pmatrix}.$$

Nun lässt sich die Ungleichung recht einfach durch das Skalarprodukt abkürzen:

$$\mathbf{x}^{T}\boldsymbol{\omega} = x_1 \omega_1 +  x_2 \omega_2 + \ldots +
x_n \omega_n \geq \theta.$$

Wie bei dem Perzeptron mit zwei Eingängen wird der Schwellenwert $\theta$ durch
Subtraktion auf die linke Seite gebracht. Wenn wir jetzt bei dem Vektor
$\boldsymbol{\omega}$ mit den Gewichten vorne den Vektor um das Element
$\omega_0 = -\theta$ ergänzen und den Vektor $\mathbf{x}$ mit $x_0 = 1$
erweitern, dann erhalten wir

$$\mathbf{x}^{T}\boldsymbol{\omega} = 1 \cdot (-\theta) + x_1 \omega_1 + x_2
\omega_2 + \ldots + x_n \omega_n \geq 0.$$

Genaugenommen hätten wir jetzt natürlich für die Vektoren $\boldsymbol{\omega}$
und $\mathbf{x}$ neue Bezeichnungen einführen müssen, aber ab sofort gehen wir
immer davon aus, dass die mit dem negativen Schwellenwert $-\theta$ und $1$
erweiterten Vektoren gemeint sind. Der negative Schwellenwert wird übrigens in
der ML-Community **Bias** oder **Bias-Einheit (Bias Unit)** genannt.

Um jetzt klassfizieren zu können, wird auf die gewichtete Summe
$\mathbf{x}^{T}\boldsymbol{\omega}$ die Heaviside-Funktion angewendet. Manchmal
wird anstatt der Heaviside-Funktion auch die Signum-Funktion verwendet. Im
Folgenden nennen wir die Funktion, die auf die gewichtete Summe angewendet wird,
**Aktivierungsfunktion**.

**Was ist ... ein Perzeptron?**

Das Perzeptron ist ein Modell, das Eingaben verarbeitet, indem es erst eine
gewichtete Summe der Eingaben bildet und dann darauf eine Aktivierungsfunktion
anwendet.

Eine typische Visualisierung des Perzeptrons ist in der folgenden Abbildung
{ref}`fig_perzeptron` gezeigt. Die Eingaben werden durch Kreise symbolisiert.
Die Multiplikation der Inputs $x_i$ mit den Gewichten $\omega_i$ wird durch
 Kanten dargestellt. Die einzelnen Summanden $x_i \omega_i$ treffen sich
sozusagen im mittleren Kreis, wo auf die gewichtete Summe dann eine
Aktivierungsfunktion angewendet wird. Das Ergebnis, der Output $\wedge{y}$ wird
dann berechnet und wiederum als Kreis gezeichnet.

![Darstellung Perzeptron](https://gramschs.github.io/book_ml4ing/_images/topology_perceptron.svg)

### Zusammenfassung und Ausblick Kapitel 12.1

In diesem Abschnitt haben wir gelernt, wie ein Perzeptron aufgebaut ist und wie
aus den Daten mit Hilfe von Gewichten und einer Aktivierungsfunktion der binäre
Zustand prognostiziert wird. Im nächsten Abschnitt beschäftigen wir uns mit der
Frage, wie die Gewichte gefunden werden.

## 12.2 Die Perzeptron-Lernregel

In dem Abschnitt über das Perzeptron waren die Gewichte und der Schwellenwert vorgegeben. Aber wie kommt man dazu? In diesem Abschnitt beschäftigen wir uns damit, wie die Gewichte und der Schwellenwert gewählt werden müssen, damit das Perzeptron seine binäre Klassifikationsaufgabe erfüllen kann.

### Lernziele Kapitel 12.2

* Sie kennen die drei Phasen, in denen ein Perzeptron trainiert wird:
  * Initialisierung der Gewichte und Festlegung der Lernrate;
  * Berechnung des prognostizierten Outputs und Aktualisierung der Gewichte sowie
  * Terminierung des Trainings.
* In Zusammenhang mit dem Training von ML-Verfahren kennen Sie die Fachbegriffe Lernrate und Epoche.

### Hebbsche Regel

Kaum zu glauben, aber die Idee zum Lernen der Gewichte eines Perzeptrons stammt
nicht von Informatiker:innen, sondern von einem Psychologen namens [Donald
Olding Hebb](https://de.wikipedia.org/wiki/Donald_O._Hebb). Im Englischen wird
seine Arbeit meist durch das Zitat

>"what fires together, wires together"

kurz zusammengefasst. Hebb hat die Veränderung der synaptischen Übertragung von
Neuronen untersucht und dabei festgestellt, dass je häufiger zwei Neuronen
gemeinsam aktiv sind, desto eher werden die beiden aufeinander reagieren.

Die Hebbsche Regel wird beim maschinellen Lernen dadurch umgesetzt, dass der
Lernprozess mit zufälligen Gewichten startet und dann der prognostizierte Output
mit dem echten Output verglichen wird. Je nachdem, ob der echte Output erreicht
wurde oder nicht, werden nun die Gewichte und damit der Einfluss eines einzelnen
Inputs verstärkt oder nicht. Dieser Prozess — Vergleichen und Abändern der
Gewichte — wird solange wiederholt, bis die passenden Gewichte gefunden sind.

Angenommen, in unserem "Ist-der-Rasen-nass-Problem" sind die Gewichte alle Null,
also $\omega_0 = \omega_1 = \omega_2 = 0$. Was prognostiziert das Perzeptron für
"es regnet nicht" ($x_1=0$) und "der Rasensprenger ist aus" ($x_2=0$)?

Das Perzeptron prognostiziert fälschlicherweise, dass der Rasen nass ist. Die
gewichtete Summe wird zu

$$\mathbf{x}^{T} \boldsymbol{\omega} = \begin{pmatrix} 0, 0, 0 \end{pmatrix}
\cdot \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix} = 0$$

berechnet. Da aber dann noch die Aktivierungsfunktion (Heaviside-Funktion)
angewendet werden muss, erhalten wir

$$\Phi(0)=1,$$

also der Rasen ist nass.

### Lernregel für das Perzeptron

Wie werden die Gewichte konkret verstärkt oder abgeschwächt, wenn der
prognostizierte Output nicht mit dem echten Output übereinstimmt? Die Lernregel
für das Perzeptron sieht zunächst einmal kompliziert aus:

$$\omega_i^{\text{neu}} = \omega_i^{\text{aktuell}} + \alpha \cdot(y -
\hat{y}^{\text{aktuell}}) \cdot x_i.$$

Gehen wir die Rechenvorschrift Stück für Stück durch. Zunächst einmal fällt auf,
dass ein Index $i$ auftaucht. Das liegt daran, dass wir mehrere Eingabewerte
haben und damit mehrere Gewichte — ein Gewicht pro Eingabewert. Da die
Lernvorschrift allgemeingültig formuliert werden soll, gehen wir jetzt einfach
mal davon aus, dass wir $m$ verschiedene Eingabewerte haben. $x_i$ meint also
den i-ten Eingabewert und mit $\omega_i$ bezeichnen wir das dazugehörige
Gewicht. Dabei dürfen wir den Bias nicht vergessen.

Bisher hatten wir den Output einfach mit $y$ gekennzeichnet. Jetzt müssen wir
aber etwas sorgfältiger vorgehen und genau unterscheiden, ob wir den Output
meinen, den das Perzeptron prognostiziert oder den echten (gemessenen) Output.
Über berechnete bzw. prognostizierte Outputs setzen wir ein kleines Dachsymbol
$\wedge$. Etwas präziser bezeichnen wir den prognostizierten Output, den das
Perzeptron mit den aktuellen Gewichten $(\omega_0^{\text{aktuell}},
\omega_1^{\text{aktuell}}, \ldots, \omega_m^{\text{aktuell}})$ berechnen würde,
mit der Abkürzung $\hat{y}^{\text{aktuell}}$ . Für den echten Output bleiben wir
einfach bei der Bezeichnung $y$.

Fehlt noch das $\alpha$, doch dazu kommen wir gleich. Schauen wir uns erst
einmal an, wie sich die Differenz $y - \hat{y}^{\text{aktuell}}$ auf die
Verstärkung oder Abschwächung der Gewichte auswirkt.

Wenn der echte Output und der prognostizierte Output gleich sind, ist deren
Differenz Null und es ändert sich nichts. Ansonsten gibt es zwei Möglichkeiten:

* Wenn der *echte Output größer ist als der prognostizierte Output*, dann ist
  $y - \hat{y}^{\text{aktuell}} > 0$. Indem wir nun zu den alten Gewichten den Term
  $ \alpha \cdot(y - \hat{y}^{\text{aktuell}})$ addieren, verstärken wir die
  alten Gewichte. Dabei berücksichtigen wir, ob der Input überhaupt einen
  Beitrag zum Output liefert, indem wir zusätzlich mit $x_i$ multiplizieren. Ist
  nämlich der Input $x_i=0$, wird so nichts an den Gewichten geändert.
* Ist jedoch *der echte Output kleiner als der prognostizierte Output*, dann ist
  $y - \hat{y}^{\text{aktuell}} < 0$. Daher werden nun die alten Gewichte durch
  die Addition des negativen Terms $\alpha \cdot(y - \hat{y}^{\text{aktuell}})
  \cdot x_i$ abgeschwächt.

Damit die Schritte zwischen der Abschwächung und der Verstärkung nicht zu groß
werden, werden sie noch mit einem Vorfaktor $\alpha$ multipliziert, der zwischen
0 und 1 liegt. Ein typischer Wert von $\alpha$ ist $0.0001$. Dieser Vorfaktor
$\alpha$ wird **Lernrate** genannt.

**Was ist ... die Lernrate?**

Die Lernrate ist eine Zahl, die zu Beginn des ML-Trainings gesetzt wird (ein
sogenannter Hyperparameter). Sie bestimmt, wie stark die neuen Gewichte auf
Fehler zwischen Prognose und tatsächlichem Output des aktuellen Durchgangs
reagieren.

### Perzeptron-Training am Beispiel des logischen ODER

Das logische Oder ist bereits durch die Angabe der folgenden vier Datensätzen
komplett definiert. Dabei haben wir noch die Bias-Einheit $x_0=1$ ergänzt.

x0 | x1 | x2 | y
---|---|----|---
1  | 0 | 0  | 0
1  | 0 | 1  | 1
1  | 1 | 0  | 1
1  | 1 | 1  | 1

Im Folgenden wird das Training eines Perzeptrons Schritt für Schritt vorgerechnet.

#### Schritt 1: Initialisierung der Gewichte und der Lernrate

Wir brauchen für die drei Inputs drei Gewichte und setzen diese drei Gewichte
jeweils auf 0. Wir sammeln die Gewichte als Vektor, also

$$\boldsymbol{\omega} = \begin{pmatrix}\omega_0 \\ \omega_1 \\ \omega_2\end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0\end{pmatrix}.$$

Darüber hinaus müssen wir uns für eine Lernrate $\alpha$ entscheiden. Obwohl
normalerweise ein kleiner (aber positiver) Wert gewählt wird, setzen wir der
Einfachheit halber die Lernrate auf 1, also $\alpha = 1$.

#### Schritt 2: Berechnung des Outputs und ggf. Anpassung der Gewichte

Wir setzen nun solange nacheinander den ersten, zweiten, dritten und vierten
Trainingsdatensatz in die Berechnungsvorschrift unseres Perzeptrons ein, bis die
Gewichte für alle vier Trainingsdatensätze zu einer korrekten Prognose führen.
Zur Erinnerung, wir berechnen den aktuellen Output des Perzeptrons mit der
Formel

$$\hat{y}^{aktuell} = \Phi(\mathbf{x}^{T}\boldsymbol{\omega}) = \Phi(x_0
\omega_0 + x_1 \omega_1 + x_2 \omega_2 ).$$

Blättern Sie Seite für Seite durch. Jede Seite entspricht einem Durchgang. Ein
Durchgang wird im ML (wie auch in der Mathematik) als eine **Iteration**
bezeichnet.

![Lernregel am Beispiel](https://gramschs.github.io/book_ml4ing/chapter12/sec02.html#schritt-2-berechnung-des-outputs-und-ggf-anpassung-der-gewichte)

#### Schritt 3: Terminierung

Die letzten vier Iterationen mit den Gewichten $(-1,1,1)$ prognostizierten
jeweils das richtige Ergebnis. Daher können wir nun mit den Iterationen stoppen.

Insgesamt brauchten wir 13 Iterationen, bis wir die Gewichte für unser
Perzeptron gefunden haben. Die finalen Gewichte haben wir bereits nach neun
Iterationen gefunden. Weitere vier Iterationen brauchten wir, um zu überprüfen,
ob das Perzeptron die vier Datensätze korrekt prognostiziert. Oder anders
ausgedrückt, mussten alle vier Datensätze noch einmal durchlaufen werden. Das
Durchlaufen aller Datensätze kommt beim mschinellen Lernen häufig vor, so dass
es dafür einen eigenen Fachbegriff gibt, nämlich die Epoche.

**Was ist ... eine Epoche?**

Das komplette Durchlaufen aller Trainingsdaten wird eine Epoche genannt.

### Zusammenfassung und Ausblick Kapitel 12.2

In diesem Abschnitt haben wir uns mit dem händischen Training eines Perzeptrons
beschäftigt. Als nächstes werden wir dazu eine Bibliothek kennenlernen, die
diese Arbeit für uns übernimmt.

## 12.3 Training eines Perzeptrons mit Scikit-Learn

Nachdem wir im letzten Abschnitt ein Perzeptron händisch für die
Klassifikationsaufgabe des logischen Oders trainiert haben, benutzen wir nun
Scikit-Learn.

### Lernziele Kapitel 12.3

* Sie können das Perzeptron-Modell von Scikit-Learn laden und mit den gegebenen
  Trainingsdaten trainieren.
* Sie wissen, wie Sie auf die Gewichte des gelernten Modells zugreifen.

### Das logische Oder Klassifikationsproblem - diesmal mit Scikit-Learn

Im letzten Abschnitt haben wir händisch ein Perzeptron trainiert. Zur
Erinnerung, wenn wir die Bias-Einheit weglassen, lautet das logische Oder in
Tabellenform wie folgt:

x1 | x2 | y
---|----|---
 0 | 0  | 0
 0 | 1  | 1
 1 | 0  | 1
 1 | 1  | 1

Diese Daten packen wir in ein DataFrame.

```{code-cell} ipython3
import pandas as pd

data =  pd.DataFrame({'x1' : [0, 0, 1, 1], 'x2'  : [0, 1, 0, 1], 'y' : [0, 1, 1, 1]})
data.head()
```

Nun wählen wir das Perzeptron als das zu trainierende ML-Modell aus. Direkt beim
Laden des Modells legen wir die Hyperparameter des Modells fest. Welche
Hyperparameter ein Modell hat, steht in der
[Perzeptron-Dokumentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html?highlight=perceptron#sklearn.linear_model.Perceptron).
In diesem Fall wäre beispielsweise die Lernrate ein Hyperparameter. Laut
Dokumentation wird die Lernrate beim Scikit-Learn-Perzeptron mit `eta0`
bezeichnet. Der Python-Code, um das Perzeptron-Modell mit einer Lernrate von 1
zu laden, lautet also wie folgt:

```{code-cell} ipython3
from sklearn.linear_model import Perceptron
model = Perceptron(eta0 = 1.0)
```

Nun können wir das Perzeptron-Modell mit den Input- und Outputdaten trainieren,
indem wir die `.fit()`-Methode aufrufen. Zuvor bereiten wir die Daten noch
passend für das Perzeptron auf.

```{code-cell} ipython3
X = data[['x1', 'x2']]
y = data['y']

model.fit(X,y)
```

Nachdem wir den letzten Python-Befehl ausgeführt haben, passiert scheinbar
nichts. Nur der Klassenname `Perceptron()` des Objekts `model` wird ausgegeben
(wenn Sie den Code interaktiv ausführen). Intern wurde jedoch das
Perzeptron-Modell trainiert, d.h. die Gewichte des Perzeptrons wurden iterativ
bestimmt. Die Gewichte sind nun in dem Objekt `model` gespeichert. Davon können
wir uns überzeugen, indem wir auf die Attribute des Objekts zugreifen und diese
anzeigen lassen. Die Gewichte sind in dem Attribut `.coef_` gespeichert, während
das Gewicht der Bias-Einheit sich im Attribut `.intercept_` befindet.

```{code-cell} ipython3
print(model.coef_)
print(model.intercept_)
```

Zuletzt können wir das trainierte Perzeptron-Modell Prognosen treffen lassen.
Was prognostiziert das Modell beispielsweise für $x_1=0$ und $x_2=1$? Das
tatsächliche Ergebnis der logischen Oder-Verknüpfung ist $y=1$, was liefert das
Perzeptron?

```{code-cell} ipython3
y_prognose = model.predict([[0, 1]])
print(y_prognose)
```

Wir können auch gleich für alle Datensätze eine Prognose erstellen.

```{code-cell} ipython3
y_prognose = model.predict(X)
print(y_prognose)
```

Tatsächlich funktioniert unser trainiertes Perzeptron zu 100 % korrekt und ist
damit validiert. Bei nur vier Datensätzen lässt sich relativ leicht überblicken,
dass alle vier Prognosen korrekt sind. Sobald die Datenanzahl zunimmt, wird es
schwieriger, den Überblick zu behalten. Daher stellt Scikit-Learn die Methode
`.score()` zur Verfügung, die bei Klassifikatoren die Anzahl der korrekt
prognostizierten Outputs im Verhältnis zur Gesamtanzahl berechnet. Das Ergbnis
ist also eine Bewertung zwischen 0 (keine einzige korrekte Prognose) und 1
(perfekt Prognose).

```{code-cell} ipython3
genauigkeit = model.score(X, y)
print(genauigkeit)
```

### Erkennung von Brustkrebs

Als nächstes betrachten wir einen sehr bekannten ML-Datensatz, nämlich Daten zur
Erkennung von Brustkrebs, siehe
<https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset>.

```{code-cell} ipython3
# Importieren des Breast Cancer Datensatzes aus Scikit-Learn
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
data = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns= np.append(cancer['feature_names'], ['target']))
data.info()
```

Wie immer berschaffen wir uns einen Überblick über die statistischen Kennzahlen.

```{code-cell} ipython3
data.describe()
```

Für das Training des Perzeptrons teilen wir die Daten in Trainings- und Testdaten auf.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

X_train = data_train.loc[:, 'mean radius' : 'worst fractal dimension']
X_test  = data_test.loc[:, 'mean radius' : 'worst fractal dimension']

y_train = data_train['target']
y_test = data_test['target']
```

Nun laden wir das Perzeptron-Modell und trainieren es mit den Trainingsdaten.

```{code-cell} ipython3
# Create a Perceptron model
model = Perceptron(eta0=0.1)

# Train the model
model.fit(X_train, y_train)
```

Wie üblich können wir es nun zu Prognosen nutzen.

```{code-cell} ipython3
# Make predictions
y_test_prognose = model.predict(X_test)
print(y_test_prognose)
```

Vor allem aber die systematische Bestimmung der Scores für Trainingsdaten und
Testdaten ist interessant:

```{code-cell} ipython3
score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)

print(f'Score Trainingsdaten: {score_train}')
print(f'Score Testdaten: {score_test}')
```

Wir könnten vermuten, dass wir bereits im Bereich des Overfittings sind.
Allerdings ist auch die Initialisierung der Zufallszahlen fixiert. Ohne
`random_state=42` kommen andere Scores für Trainings- und Testdaten heraus, so
dass wir das Perzeptron-Modell zunächst für eine erste Schätzung nehmen dürfen.

### Zusammenfassung und Ausblick Kapitel 12.3

Mit Scikit-Learn steht schon eine Implementierung des Perzeptrons zur Verfügung,
die auch bei größeren Datenmengen eine binäre Klassifikation erlaubt. Welche
Daten dabei überhaupt binär klassifiziert können, klären wir in einem der
folgenden Abschnitte.

## Übung

Der folgende Datensatz `automarkt_polen_DE.csv` enthält die Preise und
Eigenschaften von Autos aus Polen. Das Jahr bezieht sich auf die Erstzulassung
der Autos. Stadt bzw. Region beziehen sich auf den Verkaufsort des Autos. Die
übrigen Eigenschaften sind selbsterklärend und ggf. mit ihren Einheiten
angegeben.

Bearbeiten Sie die folgenden Aufgaben. Vorab können Sie die folgenden Module
importieren. Schreiben Sie Ihre Antworten in eine Markdown-Zelle.

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

### Aufgabe 1: Import und Bereinigung der Daten

Importieren Sie die Daten 'automarkt_polen_DE.csv'. Verschaffen Sie sich einen
Überblick und beantworten Sie folgende Fragen:

* Wie viele Autos enthält die Datei?
* Wie viele Attribute/Eigenschaften/Features sind in den Daten enthalten?
* Sind alle Einträge gültig? Wenn nein: welche Eigenschaften sind unvollständig
  und wie viele Einträge dieser Eigenschaft sind nicht gültig?
* Welches sind die kategorialen/diskreten/qualitativen Eigenschaften und welches
  die numerischen/quantititaven Eigenschaften?
* Welchen Datentyp haben die einzelnen Attribute/Eigenschaften/Features?

Falls der Datensatz ungültige Werte aufweist oder Unstimmigkeiten enthält,
bereinigen Sie ihn.

```{code-cell} ipython3
#
```

### Aufgabe 2: Statistische Kennzahlen der numerischen Eigenschaften

* Ermitteln Sie von den numerischen Eigenschaften die statistischen Kennzahlen
  und visualisieren Sie sie. Verwenden Sie beim Plot eine aussagefähige
  Beschriftung.
* Interpretieren Sie jede Eigenschaft anhand der statistischen Kennzahlen und
  der Plots.
* Bereinigen Sie bei Ungereimtheiten den Datensatz weiter.
* Entfernen Sie Ausreißer.
* Beantworten Sie folgende Fragen:
  * In welchem Jahr wurden die meisten Autos zugelassen?
  * Wie viele Autos wurden in diesem Jahr mit den meisten Zulassungen zugelassen?

```{code-cell} ipython3
#
```

### Aufgabe 3: Statistische Kennzahlen (kategoriale Eigenschaften)

Beantworten Sie durch Datenanalyse die folgenden Fragen. Fassen Sie die
Ergebnisse bzw. die Interpretation davon jeweils kurz zusammen (als Kommentar in
der Code-Zelle oder in einer Markdown-Zelle).

* Welche Automarke wird momentan in Polen am häufigsten und welche Automarke am
  seltesten zum Verkauf angeboten? Geben Sie jeweils die Anzahl an.
* Visualisieren Sie die Anzahl der Autos pro Marke. Beschriften Sie das Diagramm
  mit einem aussagefähigen Titel.
* Analysieren Sie die Regionen. Gibt es Regionen, die Sie überraschen? Wenn ja,
  warum?
* Wie viel Prozent der Autos haben ein Automatik-Getriebe?

```{code-cell} ipython3
#
```

### Aufgabe 4: Regression

Ziel der Regressionsaufgabe ist es, den Preis der Autos zu prognostizieren.

* Wählen Sie zwei Regressionsmodelle aus. Begründen Sie Ihre Auswahl mit einer
  Scattermatrix.
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

### Aufgabe 5

Ziel der Klassifikationsaufgabe ist es, die Preisklasse "billig" oder "teuer"
der Autos zu prognostizieren.

Achtung Vorbereitung:
Filtern Sie die Daten nach den Autos, deren Preis kleiner oder gleich dem
Median aller Preise ist. Diese Autos sollen als "billig" klassfiziert
werden. Autos, deren Preis größer als der Median aller Preise ist, sollen
als "teuer" klassifiziert werden.

* Wählen Sie die folgenden Eigenschaften aus: **Jahr** und **Kilometerstand
  [km]**.
* Adaptieren Sie die Daten passend.
* Falls notwendig, skalieren Sie die Daten.
* Trainieren Sie einen Entscheidungsbaum (Decision Tree). Begrenzen Sie dabei
  die maximale Tiefe des Baumes auf 2.
* Visualisieren Sie den Entscheidungsbaum (Decision Tree) inklusive Beschriftung
  der Labels.
* Bewerten Sie abschließend: Ist das Jahr oder der Kilometerstand wichtiger für
  die Einstufung als billiges oder teures Auto? Begründen Sie Ihre Entscheidung
  anhand des Entscheidungsbaumes (Decision Trees).

```{code-cell} ipython3
#
```
