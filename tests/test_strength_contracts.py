import unittest
from leviathan.strength.contracts import ArcTask,Example,SearchConfig,grid
from leviathan.strength.programs import Program,parse

class StrengthContracts(unittest.TestCase):
    def test_query_labels_rejected(self):
        with self.assertRaises(ValueError):ArcTask.from_public('x',{'train':[{'input':[[1]],'output':[[2]]}], 'test':[{'input':[[3]],'output':[[4]]}]})
    def test_grid_domain(self):
        for bad in ([[True]],[[10]],[[1],[2,3]],[],[[1]]*31):
            with self.assertRaises(ValueError):grid(bad)
    def test_grid_immutable_snapshot(self):
        raw=[[1,2]];g=grid(raw);raw[0][0]=9;self.assertEqual(g,((1,2),))
    def test_parser_blocks_code(self):
        for text in ('__import__("os").system("id")','x.__class__','open("file")','[v for v in x]',
                     'rot90(x); print(1)','recolor(x, True, 2)','scale(x,1000000000)','rot90(x, foo=1)'):
            with self.assertRaises((ValueError,SyntaxError)):parse(text)
    def test_ast_roundtrip(self):
        p=parse('recolor(rot90(crop(x, 0)), 1, 2)')
        self.assertEqual(p,Program.from_dict(p.as_dict()));self.assertEqual(p,parse(str(p)))
    def test_budget_is_finite(self):
        for kwargs in ({'max_seconds':float('inf')},{'max_candidates':0},{'max_depth':99}):
            with self.assertRaises(ValueError):SearchConfig(**kwargs)
    def test_max_grid_inflation_guard(self):
        with self.assertRaises(ValueError):parse('tile(x, 5, 5)').run(tuple((1,)*10 for _ in range(10)))
