import unittest
from leviathan.strength.safe_python import run,syntax,Interpreter
from leviathan.strength.programs import parse

class SafeGridLanguageTests(unittest.TestCase):
    def test_real_python_shaped_algorithm_without_exec(self):
        code='def transform(x):\n    return [list(row) for row in zip(*x)]'
        self.assertEqual(run(code,((1,2,3),(4,5,6))),((1,4),(2,5),(3,6)))
    def test_conditional_loop_and_write(self):
        code='def transform(x):\n    y = [list(r) for r in x]\n    for i in range(len(x)):\n        for j in range(len(x[0])):\n            if x[i][j] == 2:\n                y[i][j] = 3\n    return y'
        self.assertEqual(run(code,((1,2),(2,1))),((1,3),(3,1)))
    def test_no_input_mutation(self):
        x=((1,2),);run('def transform(x):\n    x[0][0] = 9\n    return x',x)
        self.assertEqual(x,((1,2),))
    def test_no_import_files_attributes_recursion(self):
        for body in ('import os\n    return x','return open(1)','return x.__class__','return transform(x)',
                     'while True:\n        pass\n    return x','return eval(1)'):
            with self.assertRaises(ValueError):syntax('def transform(x):\n    '+body)
    def test_resource_limits(self):
        for code in ('def transform(x):\n    return [[0] * 100000000]',
                     'def transform(x):\n    return [[0] for i in range(100000000)]'):
            with self.assertRaises(ValueError):run(code,((1,),))
    def test_program_composes_with_grid_dsl(self):
        p=parse('def transform(x):\n    return x[::-1]')
        g=((1,2),(3,4));self.assertEqual(p.then('rot90').run(g),((4,2),(3,1)))
    def test_negative_slice_and_boolean(self):
        code='def transform(x):\n    return [[3 if c == 2 else c for c in row[::-1]] for row in x]'
        self.assertEqual(run(code,((1,2),)),((3,1),))


class WitnessHoleTests(unittest.TestCase):
    def test_neural_literal_sketch_can_be_repaired_without_weights(self):
        from leviathan.strength.contracts import Example
        from leviathan.strength.programs import constant_hole_repairs
        p=parse('def transform(x):\n    return [[3 if c == 1 else c for c in row] for row in x]')
        examples=(Example(((1,0),),((2,0),)),Example(((0,1),),((0,2),)))
        repairs=list(constant_hole_repairs(p,examples))
        self.assertTrue(any(all(q.run(e.input)==e.output for e in examples) for q in repairs))
        self.assertEqual(p.run(((1,0),)),((3,0),))
    def test_repairs_have_explicit_limit(self):
        from leviathan.strength.contracts import Example
        from leviathan.strength.programs import constant_hole_repairs
        p=parse('recolor(x, 1, 2)')
        ex=(Example(((0,1,2,3,4,5,6,7,8,9),),((0,1,2,3,4,5,6,7,8,9),)),)
        self.assertEqual(len(list(constant_hole_repairs(p,ex,limit=2))),2)


class InterpretedLanguageGuardTests(unittest.TestCase):
    def test_bounded_list_building_and_count(self):
        code='def transform(x):\n    y = []\n    for row in x:\n        if row.count(2) > 0:\n            y.append(row.copy())\n    return y'
        self.assertEqual(run(code,((1,2),(3,4))),((1,2),))
    def test_generator_is_an_eager_bounded_comprehension(self):
        code='def transform(x):\n    return [[sum(c == 2 for row in x for c in row)]]'
        self.assertEqual(run(code,((1,2),(2,3))),((2,),))
    def test_method_values_and_unknown_receivers_rejected(self):
        with self.assertRaises(ValueError):syntax('def transform(x):\n    y=x.append\n    return x')
        with self.assertRaises(ValueError):run('def transform(x):\n    y=1\n    y.append(2)\n    return x',((1,),))
    def test_recursive_containers_do_not_escape_operation_budget(self):
        code='def transform(x):\n    a = [0]\n    b = [0]\n    for i in range(30):\n        a = [a, a]\n        b = [b, b]\n    return [[1 if a == b else 0]]'
        with self.assertRaises(ValueError):run(code,((1,),))
    def test_list_sorting_is_operation_counted(self):
        code='def transform(x):\n    return sorted(x)'
        self.assertEqual(run(code,((3,4),(1,2))),((1,2),(3,4)))
