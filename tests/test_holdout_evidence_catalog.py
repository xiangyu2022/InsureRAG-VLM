import importlib.util
from pathlib import Path
import unittest
from bs4 import BeautifulSoup
from src.insurerag_vlm.source_faq_extraction import clean,content_root

spec=importlib.util.spec_from_file_location('evidence_catalog',Path(__file__).resolve().parents[1]/'scripts/build_holdout1000_evidence_catalog.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class EvidenceCatalogTests(unittest.TestCase):
 def catalog(self,html):
  text=clean(content_root(BeautifulSoup(html,'html.parser')).get_text(' ',strip=True))
  doc=dict(id='fixture',publisher='synthetic',jurisdiction='XX',source_title='Synthetic insurance rules',source_url='https://example.test/rules',text=text)
  return module.catalog_document(doc,html),text

 def test_short_limit_and_neighbor_qualifier_survive(self):
  html='<main><h1>Auto coverage</h1><h2>Injury limits</h2><p>These limits apply only to covered bodily injury claims.</p><ul><li>$25,000 per injured person.</li><li>$50,000 total per accident.</li></ul></main>'
  result,text=self.catalog(html)
  self.assertEqual(len(result['blocks']),3)
  self.assertIn('$25,000 per injured person.',result['sections'][0]['text'])
  self.assertIn('only to covered bodily injury',result['sections'][0]['text'])
  for b in result['blocks']:self.assertEqual(text[b['start']:b['end']],b['text'])

 def test_whole_table_retains_headers_without_duplicate_cells(self):
  html='<main><h1>Plan costs</h1><table><tr><th>Plan</th><th>Network deductible</th></tr><tr><td>Plan A</td><td>$100 per year</td></tr><tr><td>Plan B</td><td>$200 per year</td></tr></table></main>'
  result,_=self.catalog(html)
  self.assertEqual(len(result['blocks']),1)
  self.assertEqual(result['blocks'][0]['element'],'table')
  self.assertIn('Network deductible',result['blocks'][0]['text'])
  self.assertIn('Plan B $200 per year',result['blocks'][0]['text'])

 def test_va_table_retains_title_headers_and_exact_source_span(self):
  html='<main><h1>Life insurance premiums</h1><va-table table-title="Monthly premiums"><table><tr><th>Age</th><th>$10,000 coverage</th></tr><tr><td>60</td><td>$50.00</td></tr></table></va-table><p>Rates apply to the age when coverage begins.</p></main>'
  result,text=self.catalog(html)
  self.assertEqual(len(result['blocks']),2)
  table=result['blocks'][0]
  self.assertEqual(table['element'],'va-table')
  self.assertEqual(table['table_title'],'Monthly premiums')
  self.assertIn('Age $10,000 coverage 60 $50.00',table['text'])
  self.assertEqual(text[table['start']:table['end']],table['text'])
  self.assertIn('Rates apply to the age',result['sections'][0]['text'])

if __name__=='__main__':unittest.main()
