import unittest
from leviathan.strength.evaluation import score,select_ids
class StrengthEvaluationTests(unittest.TestCase):
    def test_wrong_shape_wrong_cell_are_not_solved(self):
        labels={'t':[[[1,2]]]}
        p={'t':{'attempts':[[[[1],[2]],[[1,3]]]]}}
        self.assertEqual(score(p,labels,['t'])['task_pass2'],0)
    def test_whole_task_all_queries_required(self):
        labels={'t':[[[1]],[[2]]]};p={'t':{'attempts':[[[[1]]],[[[3]]]]}}
        r=score(p,labels,['t']);self.assertEqual(r['query_pass1'],1);self.assertEqual(r['task_pass2'],0)
    def test_max_two_attempts(self):
        labels={'t':[[[3]]]};p={'t':{'attempts':[[[[1]],[[2]],[[3]]]]}}
        self.assertEqual(score(p,labels,['t'])['task_pass2'],0)
    def test_second_attempt_counts_separately(self):
        r=score({'t':{'attempts':[[[[1]],[[2]]]]}},{'t':[[[2]]]},['t'])
        self.assertEqual((r['task_pass1'],r['task_pass2']),(0,1))
    def test_selection_independent_of_labels_order(self):
        self.assertEqual(select_ids({'a':{},'b':{},'c':{}},2),select_ids({'c':{},'b':{},'a':{}},2))
