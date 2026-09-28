"""Generate a standalone, offline dashboard from saved results."""
import json
from prepare_data import ROOT

def main():
    binary=json.loads((ROOT/'results/binary_raw.json').read_text())
    balanced=json.loads((ROOT/'results/balanced_results.json').read_text())
    manifest=json.loads((ROOT/'results/data_manifest.json').read_text())
    # Escape '<' so review text cannot close the script element.
    data=json.dumps({'binary':binary,'balanced':balanced,'manifest':manifest},ensure_ascii=False).replace('<','\\u003c')
    template=(ROOT/'dashboard_template.html').read_text()
    (ROOT/'dashboard.html').write_text(template.replace('__DATA__',data),encoding='utf-8')
    print('Wrote dashboard.html')
if __name__=='__main__': main()
