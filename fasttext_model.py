import xml.etree.ElementTree as ET
import numpy as np
import sys
import re

def clean(text):
    return re.sub("[^a-zA-Z ]+", '', text)

def parse(input_file, output_file):
    data = ET.parse(input_file)
    root = data.getroot()
    wis = root.findall("workItem")

    with open(output_file, 'w') as file:
        for wi in wis:
            print(clean(wi.findtext("summary")), end=' ', file=file)
            print(clean(wi.findtext("description")), end=' ', file=file)
            print(file=file)

if __name__=="__main__":
    parse('export.xml', 'test.txt')