import unittest
from scripts.build_policy_wordings import literal_references


class LiteralPageReferenceTests(unittest.TestCase):
    def test_printed_offset_not_physical_number(self):
        accepted,rejected=literal_references('See page 8 for the definition.',{'8':[10]},3)
        self.assertEqual(accepted[0]['target'],10)
        self.assertEqual(accepted[0]['label'],'8')
        self.assertEqual(rejected,[])

    def test_never_guess_missing_ambiguous_self_or_year(self):
        text='See page 8, see page 9, see page 10. Continued on next page 2026 Plan.'
        accepted,rejected=literal_references(text,{'8':[10,11],'10':[3]},3)
        self.assertEqual(accepted,[])
        self.assertEqual(len(rejected),3)

    def test_multi_target_reference_is_skipped(self):
        for text in ['page 8-10','page 8–10','page 8 and 10','page 8, 10']:
            accepted,rejected=literal_references(text,{'8':[10]},3)
            self.assertFalse(accepted)
            self.assertEqual(rejected[0]['reason'],'multi_target_list_or_range')

    def test_literal_quote_is_exact_source_substring(self):
        text='The deductible changes. See PAGE 8 for details of the exception.'
        accepted,_=literal_references(text,{'8':[10]},3)
        self.assertIn(accepted[0]['quote'],text)
