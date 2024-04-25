import leafmap
from IPython.display import display
import jupyter_leaflet

m = leafmap.Map(center=(35, 140), zoom=4)
display(m)

m = leafmap.Map()
m.to_html('./map.html')