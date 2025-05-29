import h3
from shapely.geometry import Polygon

print(h3.__version__) # Check version
# A simple square polygon
simple_poly = Polygon([(0,0), (0,1), (1,1), (1,0), (0,0)])
try:
    cells = h3.polygon_to_cells(simple_poly, 9)
    print("Success! Cells:", cells)
except Exception as e:
    print("Error with simple polygon:", e)