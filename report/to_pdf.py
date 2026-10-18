"""Convert report_pdf.docx to report.pdf with LibreOffice, updating the table of
contents and the lists of tables / figures first (a plain `soffice --convert-to`
leaves them without page numbers).

    /usr/bin/python3 report/to_pdf.py
"""
import os
import subprocess
import sys
import time

import uno
from com.sun.star.beans import PropertyValue

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "report_pdf.docx")
OUT = os.path.join(HERE, "report.pdf")
PORT = 2099


def prop(name, value):
    p = PropertyValue()
    p.Name, p.Value = name, value
    return p


def main():
    office = subprocess.Popen(["soffice", "--headless", "--invisible", "--norestore",
                               f"--accept=socket,host=127.0.0.1,port={PORT};urp;"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        resolver = uno.getComponentContext().ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", uno.getComponentContext())
        for _ in range(60):
            try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={PORT};urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(0.5)
        else:
            sys.exit("LibreOffice did not start")
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(SRC), "_blank", 0, (prop("Hidden", True),))
        for _ in range(2):    # second pass: page numbers settle after the lists themselves take space
            idx = doc.getDocumentIndexes()
            for i in range(idx.getCount()):
                idx.getByIndex(i).update()
            doc.refresh()
        doc.storeToURL(uno.systemPathToFileUrl(OUT), (prop("FilterName", "writer_pdf_Export"),))
        doc.close(True)
    finally:
        office.terminate()
        office.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
