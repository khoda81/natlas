# Browser

[Project setup](../README.md) · [Rendering performance](../docs/performance.md)

## The picture

Pick one measure $p$ and one measure $q$. They are the same by default. A
model is weights loaded on a device at a precision, such as
`large/cuda/float16`. A measure reads that model with a chosen conditioning
and temperature, such as `large/cuda/float16/audio=3f2a9c/temperature=1.6`.
This identifier is its *spec*. Measures from the same server share branch
labels, so you can compare two models with the same tokenizer, two precisions,
or the same model hearing different recordings.

On a live source, open **models…** beside the $p$ and $q$ selectors to load
or unload a model and create measures from a loaded model. The selectors
list the measures the server holds. See [live server setup](../export/README.md)
and the [music transcription guide](../export/scribe.md) for the options each
server offers.

Each child of a node is a box.

- **Height** follows $p$. The children split their parent's interval in $p$'s
  rank order. A box's height is proportional to the $p$-probability of its
  whole path.
- **Width** is $-\ln q$ nats for the branch conditioned on its parent. The
  box begins where its parent's box ends, so its left edge marks the
  $q$-surprisal accumulated through the parent, measured from the current
  origin.

At a node with $p$-mass $m$, the area of the child boxes represents
$m \sum_i p_i(-\ln q_i)$. This is mass times conditional entropy when
$p = q$, or mass times conditional cross-entropy when the measures differ.
The excess over the $p = q$ area is $m D_{\mathrm{KL}}(p\|q)$.

For ChatML token paths, blue fills show the user turn, green the assistant
turn, and purple the assistant's thinking. A turn separator brings the fill
back to neutral until the next role header.

## Canvas navigation

The view is measured from an **origin** node, usually the root. At a left
edge $x$ nats from the origin, the view spans $e^{-x}$ of the origin's
interval. Moving deeper therefore also zooms in. The same node keeps its
screen height as you follow it to the left edge when $p = q$. When the
measures differ, the picture depends on the origin because horizontal depth
uses $q$ and height uses $p$. Horizontal scale can be adjusted independently.

Opening text flies to its path. The server places branches as it computes
them, and the camera descends along the part already placed. The bar along
the top shows progress. While the path is being placed, it takes priority
over the rest of the view. Only the root bounds the view. If you pan past
another origin, the browser fetches its parent. The space stays empty until
the parent arrives, then the parent becomes the origin. With $p = q$, that
change keeps the picture in place.

| Input | Action |
| --- | --- |
| Wheel or <kbd>←</kbd> / <kbd>→</kbd> | Move deeper or shallower, zooming about the pointer |
| Drag | Pan |
| <kbd>Shift</kbd> + wheel or <kbd>↑</kbd> / <kbd>↓</kbd> | Move along the vertical interval |
| <kbd>Ctrl</kbd> + wheel, pinch, or <kbd>+</kbd> / <kbd>−</kbd> | Change horizontal scale about the left edge |
| Click a box | Glide to its children |
| Hover | Pick a branch by height and depth. Empty space keeps the deepest known subtree selected. |
| <kbd>0</kbd> or <kbd>Home</kbd> | Return to the root |
| <kbd>S</kbd> | Swap $p$ and $q$ |
| <kbd>B</kbd> | Switch between nats and bits |
| <kbd>?</kbd> | Show help |

Click the text in the top path inspector to edit and search. Select a result
or press <kbd>Enter</kbd> to fly there. Leaving the editor shows hovered paths
again and keeps the draft. Press <kbd>Esc</kbd> in the editor or click the
canvas to discard it.

## Snapshots and saved views

Drop one or more `.pb` snapshots onto the page, optionally with a labels
file. Recorded text search includes listed leaves and expanded nodes. It
returns an exact match, a completion, or the deepest recorded prefix, and
reports an error when there is no match. A token can contain part of a character,
so the client decodes the path's bytes together. A box shows the characters
its token finishes, or the token's bytes in hex when it finishes none.

The browser saves the origin, camera path and position, horizontal scale,
and chosen measures in local storage. The key includes the source URLs or
the dropped files' contents. On reload, the browser walks the saved path
from the origin and shows the saved view once the path is placed. The browser
chooses the initial view from its own state.

## Source

- [UI](ui/) holds the camera controls, canvas drawing, path editor, and saved views.
- [Data sources](data/) load snapshots and connect to live servers.
- [Readings](readings/) turn branches into text or music.
- [Core adapters](core/) call the [shared Rust core](../core/README.md) through WebAssembly.
