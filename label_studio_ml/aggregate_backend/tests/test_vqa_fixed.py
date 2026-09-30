import unittest
from types import SimpleNamespace
from unittest.mock import patch

from label_studio_ml.aggregate_backend.template_router import resolve_vqa_route
from label_studio_ml.aggregate_backend.result_converter import convert_vqa
from label_studio_ml.aggregate_backend.templates.vqa import DoubaoVQA


class FixedQuestionTests(unittest.TestCase):
    def test_fixed_and_dynamic_questions(self):
        route = resolve_vqa_route(SimpleNamespace(label_config='''<View>
          <Image name="image" value="$image"/>
          <Text name="q1" value="图片里有什么？"/>
          <TextArea name="a1" toName="q1"/>
          <Text name="q2" value="$question2"/>
          <TextArea name="a2" toName="q2"/>
        </View>'''))
        self.assertEqual(route.fixed_questions, {'q1': '图片里有什么？'})
        self.assertEqual(route.questions, [('q1', 'a1', 'q1'), ('question2', 'a2', 'q2')])
        results = convert_vqa(route, {'q1': '猫', 'question2': '两只'}, None, 1)
        self.assertEqual([r['from_name'] for r in results], ['a1', 'a2'])
        self.assertEqual([r['value']['text'] for r in results], [['猫'], ['两只']])

    def test_empty_questions_fail_without_network(self):
        with patch('requests.post') as post:
            with self.assertRaisesRegex(ValueError, 'questions are empty'):
                DoubaoVQA(api_key='test').answer_file('unused', {'q': ''})
            post.assert_not_called()
