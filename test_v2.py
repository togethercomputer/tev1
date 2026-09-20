import itertools,unittest
from build_v2 import evaluate,table,label,independent_label,check_atom,synthetic,validate_synthetic,shape
class SemanticsTests(unittest.TestCase):
 def test_hand_written_truth_tables(self):
  for a,b,c in itertools.product([False,True],repeat=3):
   facts=dict(a=a,b=b,c=c)
   cases=[(('and','a',('not','b')),a and not b),(('or','a',('and','b','c')),a or (b and c)),(('atleast',2,'a','b','c'),sum([a,b,c])>=2),(('exactly',1,'a','b','c'),sum([a,b,c])==1),(('not',('exactly',2,'a','b','c')),sum([a,b,c])!=2)]
   for t,want in cases:self.assertEqual(evaluate(t,facts),want)
 def test_unknown_is_not_missing(self):
  names=('a','b','c');trees=(('or',('and','a','b'),'c'),);worlds=table(trees,names,False)
  for facts,want in [({'c':True},'eligible'),({'a':False,'c':False},'ineligible'),({'a':True,'c':False},'unknown'),({},'unknown')]:
   self.assertEqual(label(worlds,names,facts),want)
   self.assertEqual(independent_label(trees,names,facts,False),want)
 def test_repeated_atoms_are_correlated(self):
  for t,want in [(('or','a',('not','a')),'eligible'),(('and','a',('not','a')),'ineligible')]:
   self.assertEqual(label(table((t,),('a',),False),('a',),{}),want)
   self.assertEqual(independent_label((t,),('a',),{},False),want)
 def test_priority_rules(self):
  trees=('a','b');names=('a','b');worlds=table(trees,names,True)
  for facts,want in [({'a':True},'route_1'),({'a':False,'b':True},'route_2'),({'b':True},'unknown'),({'a':False,'b':False},'default')]:
   self.assertEqual(label(worlds,names,facts),want)
   self.assertEqual(independent_label(trees,names,facts,True),want)
 def test_numeric_boundaries(self):
  for op,want in [('<',[True,False,False]),('<=',[True,True,False]),('>',[False,False,True]),('>=',[False,True,True]),('==',[False,True,False]),('!=',[True,False,True])]:
   self.assertEqual([check_atom({'op':op,'value':10},n) for n in [9,10,11]],want)
 def test_generated_facts_and_disjoint_shapes(self):
  signatures={}
  for split in ['train','dev','transfer']:
   rows=synthetic(split,12)+synthetic(split,12,True)
   signatures[split]={r['provenance']['shape_signature'] for r in rows}
   for r in rows:validate_synthetic(r)
  self.assertFalse((signatures['train']|signatures['dev'])&signatures['transfer'])
  self.assertEqual(synthetic('train',6),synthetic('train',6))
if __name__=='__main__':unittest.main()
