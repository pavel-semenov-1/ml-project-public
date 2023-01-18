import xml.etree.ElementTree as ET
import random

data = ET.parse("export.xml")
root = data.getroot()
element = root.findall("workItem")[random.randint(0, 28791)]
print([i for i in element.iter()])